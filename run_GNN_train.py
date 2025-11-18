import torch
import os
import numpy as np
import gc
import json
from torch_geometric.data import HeteroData
from GNN_model.train_GNN import GNNTrainer
from GNN_model.GNN_class import LightGCN, EmbeddingLayer
from GNN_model.eval_GNN import GNNEvaluator
from config import config
from GNN_model.diagnostics import diagnose_embedding_scales
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType, ShardingStrategy, CPUOffload
from torch.distributed.fsdp.wrap import wrap as fsdp_wrap
from functools import partial
import torch.multiprocessing as mp


def save_fsdp_model(model: FSDP, rank: int, save_path: str):
    """Saves the FSDP model state dict using FULL_STATE_DICT type on rank 0."""
    with FSDP.state_dict_type(model, statedict_type=StateDictType.FULL_STATE_DICT):
        cpu_state = model.state_dict()
    if rank == 0:
        torch.save(cpu_state, save_path)
        print(f"> FSDP model saved successfully to {save_path}")
        return cpu_state
    return None


def check_prev_files():
    """
    check for the files created in the previous stage.
    if at least one file is missing raises FileNotFoundError
    """
    needed = [config.paths.audio_embeddings_file, config.paths.train_graph_file,
              config.paths.test_scores_file, config.paths.negative_train_in_graph_file,
              config.paths.negative_train_cold_start_file]
    fail = False
    for file in needed:
        if not os.path.exists(file):
            print("Couldn't find file: {}".format(file))
            fail = True
    if fail:
        raise FileNotFoundError("Needed files are missing, run previous stage to create the needed files!")
    else:
        print("All needed files are present! starting GNN training ... ")


def test_evaluation(model: LightGCN, train_graph: HeteroData):
    print("evaluating best model on test set...")
    test_evaluator = GNNEvaluator(model, train_graph, "test", config)
    test_metrics = test_evaluator.evaluate()
    k_hit = config.gnn.k_hit

    print(f"Test set metrics @K={k_hit}:")
    print(f"  NDCG@{k_hit}: {test_metrics['ndcg@k']:.4f}")
    print(f"  NDCG_raw@{k_hit}: {test_metrics['ndcg_raw@k']:.4f}")
    print(f"  Hit@{k_hit} (like only): {test_metrics['hit_like@k']:.4f}")
    print(f"  Hit@{k_hit} (like+listen): {test_metrics['hit_like_listen@k']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  Dislike-FPR@{k_hit}: {test_metrics['dislike_fpr@k']:.4f}")
    print(f"  Novelty@{k_hit}: {test_metrics['novelty@k']:.4f}")

    with open(config.paths.test_eval, "w") as f:
        json.dump(test_metrics, f, indent=4)


def save_final_embeddings(model: LightGCN, user_embed_path: str, song_embed_path: str):
    """
    Saves final user and song embeddings to disk.

    This function sets the model to evaluation mode and calls the
    memory-efficient `forward_cpu` method to get embeddings
    without causing GPU OOM errors.
    """
    print("Starting to save final embeddings...")
    torch.cuda.empty_cache()
    model.eval()

    with torch.no_grad():
        # Call the new CPU-based forward method.
        # This returns user and item embeddings as CPU tensors.
        user_emb, item_emb, _ = model.forward_cpu()

        print("Converting final embeddings to NumPy...")
        # Convert to NumPy
        # Tensors are already on CPU, so .numpy() is direct and fast
        user_emb_np = user_emb.numpy().astype(np.float32)
        item_emb_np = item_emb.numpy().astype(np.float32)

        # Get original IDs (assuming these are already on CPU or small)
        user_ids_np = model.user_original_ids.cpu().numpy()
        item_ids_np = model.item_original_ids.cpu().numpy()

        # Save to .npz files
        print(f"Saving user embeddings to {user_embed_path}...")
        np.savez(user_embed_path, embeddings=user_emb_np, original_ids=user_ids_np)

        print(f"Saving song embeddings to {song_embed_path}...")
        np.savez(song_embed_path, embeddings=item_emb_np, original_ids=item_ids_np)

    print("-------------------------------------------------")
    print(f"User embeddings saved to {user_embed_path}")
    print(f"Song embeddings saved to {song_embed_path}")
    print("Embedding saving process complete.")


def ddp_main(rank, world_size):
    # Set the device for this process and initialize distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Configure the rank's device
    device = torch.device(f"cuda:{rank}")
    config.gnn.device = device

    if rank == 0:
        print(f"World Size: {world_size}. Current Rank: {rank}. Device: {device}")

    # Load data onto CPU (it is too large to fully replicate on each GPU)
    train_graph = torch.load(config.paths.train_graph_file, map_location='cpu')

    # --- Model Initialization & Diagnostics ---
    model = LightGCN(train_graph, config)

    if rank == 0:
        audio_scale, metadata_scale = diagnose_embedding_scales(model)
    else:
        audio_scale, metadata_scale = 0.0, 0.0

        # Broadcast scales
    broadcast_tensors = [torch.tensor(audio_scale, device=device), torch.tensor(metadata_scale, device=device)]
    dist.broadcast(broadcast_tensors[0], src=0)
    dist.broadcast(broadcast_tensors[1], src=0)
    model.audio_scale = broadcast_tensors[0].item()
    model.metadata_scale = broadcast_tensors[1].item()

    model.to(device)

    fsdp_auto_wrap_policy = partial(
        fsdp_wrap,
        module_to_wrap={EmbeddingLayer},
        min_num_params=0  # Ensure wrapping occurs even if size is small, as we target by type
    )

    # --- FSDP WRAPPER: The critical change for multi-GPU training ---
      # Move initial embeddings (user/item meta) to the GPU before wrapping
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        # AGGRESSIVE OOM FIX: Offload parameters and optimizer state to CPU RAM
        cpu_offload=CPUOffload(offload_params=True),
        device_id=device,
        auto_wrap_policy=fsdp_auto_wrap_policy
    )

    trainer = GNNTrainer(model, train_graph, config)
    trainer.train(trial=False)  # Run training on all GPUs

    # --- FSDP Model Save and Evaluation (only on Rank 0) ---
    if rank == 0:
        # Save the full state dict by gathering parameters onto rank 0
        saved_state_dict = save_fsdp_model(model, rank, config.paths.trained_gnn)

        # Load the saved state dict back into a new LightGCN model for evaluation
        # We use the CPU state because evaluation is often done offline/on CPU
        eval_model = LightGCN(train_graph, config)
        eval_model.audio_scale = model.module.audio_scale  # FSDP module
        eval_model.metadata_scale = model.module.metadata_scale
        eval_model.load_state_dict(saved_state_dict)

        # Cleanup
        del saved_state_dict, trainer, model
        gc.collect()
        torch.cuda.empty_cache()

        # Run test evaluation and save embeddings (original logic)
        test_evaluation(eval_model, train_graph)
        save_final_embeddings(eval_model, config.paths.user_embeddings_gnn, config.paths.song_embeddings_gnn)

    dist.barrier()  # Wait for rank 0 to finish evaluation
    dist.destroy_process_group()  # Cleanup


def main():
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Warning: Fewer than 2 GPUs detected. Launching single-GPU FSDP (still shards embeddings).")
        # For development/testing, we can run with a world size of 1

    # Use torch.multiprocessing.spawn to launch the workers
    print(f"Launching distributed FSDP training on {world_size} GPUs...")
    mp.spawn(
        ddp_main,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    check_prev_files()
    main()
