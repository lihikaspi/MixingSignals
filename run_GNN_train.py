import torch
import os
import numpy as np
import gc
import json
from torch_geometric.data import HeteroData
from GNN_model.train_GNN import GNNTrainer
from GNN_model.GNN_class import LightGCN
from GNN_model.eval_GNN import GNNEvaluator
from config import config
from GNN_model.song_embed_diagnostics import diagnose_embedding_scales


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


def main():
    torch.cuda.empty_cache()
    train_graph = torch.load(config.paths.train_graph_file)

    model = LightGCN(train_graph, config)

    audio_scale, metadata_scale = diagnose_embedding_scales(model)
    model.audio_scale = audio_scale
    model.metadata_scale = metadata_scale

    trainer = GNNTrainer(model, train_graph, config)
    trainer.train()

    model = trainer.model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    model.load_state_dict(torch.load(config.paths.trained_gnn, map_location=config.gnn.device))
    test_evaluation(model, train_graph)

    save_final_embeddings(model, config.paths.user_embeddings_gnn, config.paths.song_embeddings_gnn)


if __name__ == "__main__":
    check_prev_files()
    main()