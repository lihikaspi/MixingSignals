import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import json
import math
from GNN_model.GNN_class import LightGCN
from GNN_model.eval_GNN import GNNEvaluator
from config import Config
from collections import defaultdict
import pandas as pd


class BPRDataset(Dataset):
    """
    UNIFIED (Request 2): Advanced BPR Dataset
    This version uses the pre-computed hybrid ratio from the graph
    (edge_attr column 2) as the positive weight.
    """

    def __init__(self, train_graph, config: Config, raw_user2neg: dict):
        print(">>> Initializing Advanced BPRDataset...")

        # All edges are now positive in the graph
        edge_index = train_graph['user', 'interacts', 'item'].edge_index
        edge_attr = train_graph[
            'user', 'interacts', 'item'].edge_attr

        user_idx, item_idx = edge_index.cpu()
        # edge_attr[:, 0] is the 'edge_type' (e.g., 1=listen, 2=like, 5=undislike)
        edge_types = edge_attr[:, 0].cpu()
        # UNIFIED: Get edge_avg_played_ratio (This is the hybrid weight)
        # This is column 2 (index 2) from build_graph.py
        edge_ratios = edge_attr[:, 2].cpu()

        self.user2pos = {}
        # UNIFIED: Store ratios instead of types
        self.user2pos_ratios = {}
        # self.user2neg stores list of tuples: (item_train_idx, audio_embed)
        self.user2neg = {}  # For "hard" negatives (dislikes)
        self.num_items = int(train_graph['item'].num_nodes)
        self.all_users_pos_sets = defaultdict(set)  # For "easy" negative sampling

        self.listen_weight = config.gnn.listen_weight  # No longer used for pos_weight
        self.neutral_neg_weight = config.gnn.neutral_neg_weight
        self.neg_samples_per_pos = config.gnn.neg_samples_per_pos
        self.edge_type_mapping = config.preprocessing.edge_type_mapping

        # Define positive event types for filtering edge_attr value just in case of graph artifacts
        pos_types = [self.edge_type_mapping[k] for k in ["listen", "like", "undislike"]]

        # 1. Extract POSITIVE interactions from the graph
        print("    Building user-to-item positive map from graph...")
        # UNIFIED: Add 'r' (ratio) to the loop
        for u, i, t, r in zip(user_idx.tolist(), item_idx.tolist(), edge_types.tolist(), edge_ratios.tolist()):
            t_int = int(round(t))
            if t_int in pos_types:
                self.user2pos.setdefault(u, []).append(i)
                # UNIFIED: Store the ratio
                self.user2pos_ratios.setdefault(u, []).append(r)
                self.all_users_pos_sets[u].add(i)  # Add to set for fast lookup

        # 2. Process the pre-loaded RAW hard negative map
        print("    Cleaning hard negative lists...")
        for u, neg_tuples in raw_user2neg.items():
            pos_set_for_user = self.all_users_pos_sets.get(u, set())
            cleaned_negs = []
            for (item_train_idx, audio_embed) in neg_tuples:
                # Only clean if it's an in-graph item
                if item_train_idx != -1:
                    if item_train_idx not in pos_set_for_user:
                        cleaned_negs.append((item_train_idx, audio_embed))
                else:
                    # Always keep cold-start negatives (they can't be in the pos set)
                    cleaned_negs.append((item_train_idx, audio_embed))

            if cleaned_negs:
                self.user2neg[u] = cleaned_negs

        # self.users is the list of users who have at least one positive item
        self.users = sorted(list(self.user2pos.keys()))
        print(f"Advanced BPRDataset: Loaded {len(self.users)} users with positive interactions.")

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        pos_items = self.user2pos[u]
        # UNIFIED: Get pos_ratios
        pos_ratios = self.user2pos_ratios[u]

        # Get a mutable list of hard negatives for this user
        hard_negs_list = self.user2neg.get(u, []).copy()

        # Get the set of all positive items for this user for fast lookup
        known_pos_set = self.all_users_pos_sets[u]

        # --- 1. Sample one positive item ---
        i_pos_idx = np.random.randint(len(pos_items))
        i_pos = pos_items[i_pos_idx]

        # --- UNIFIED (Request 2): Use the pre-computed hybrid ratio as the weight ---
        # This ratio is 1.0 for 'like', 0.5 for 'undislike',
        # and played_ratio_pct/100.0 for 'listen'.
        pos_weight = pos_ratios[i_pos_idx]
        # --- END UNIFIED ---

        # --- 2. Sample K negative items ---
        neg_graph_indices = []
        neg_audio_embeds = []
        neg_weights = []

        for _ in range(self.neg_samples_per_pos):

            # First, try to sample a "hard" negative (explicit dislike)
            if len(hard_negs_list) > 0:
                i_neg_sample_idx = np.random.randint(len(hard_negs_list))
                # Sample without replacement
                i_neg_tuple = hard_negs_list.pop(i_neg_sample_idx)

                neg_graph_indices.append(i_neg_tuple[0])  # item_train_idx (-1 for cold-start)
                neg_audio_embeds.append(i_neg_tuple[1])  # audio_embed (None for in-graph)
                neg_weights.append(1.0)  # Hard negatives get full weight

            # If no hard negatives left, sample an "easy" negative (random unseen)
            else:
                while True:
                    # Sample a random item from the *entire catalog*
                    i_neg_idx = np.random.randint(self.num_items)
                    # Keep sampling until we find one that is NOT positive
                    if i_neg_idx not in known_pos_set:
                        break  # Found a valid random negative

                neg_graph_indices.append(i_neg_idx)  # This is an in-graph item
                neg_audio_embeds.append(None)  # It's in-graph, so no separate embed
                neg_weights.append(self.neutral_neg_weight)  # Easy negatives get reduced weight

        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(i_pos, dtype=torch.long),
            neg_graph_indices,  # list of k longs
            neg_audio_embeds,  # list of k (Nones or lists)
            torch.tensor(pos_weight, dtype=torch.float),  # scalar
            torch.tensor(neg_weights, dtype=torch.float)  # [k]
        )


def collate_bpr_advanced(batch):
    """
    Collates the output of the Advanced BPRDataset.
    Handles None values in audio embeddings for in-graph items.
    """
    batch_size = len(batch)
    # Find embed_dim from the first non-None audio embed
    embed_dim = -1
    for b in batch:
        for embed in b[3]:  # b[3] is neg_audio_embeds
            if embed is not None:
                embed_dim = len(embed)
                break
        if embed_dim != -1:
            break

    # Fallback if no audio embeds were found in batch (e.g., all easy negatives)
    if embed_dim == -1:
        embed_dim = 128  # Should match config.gnn.embed_dim, but hardcoding is safer here

    k = len(batch[0][2])  # neg_samples_per_pos

    # Unpack simple tensors
    u_idx = torch.stack([b[0] for b in batch])
    i_pos_idx = torch.stack([b[1] for b in batch])
    pos_weights = torch.stack([b[4] for b in batch])
    neg_weights = torch.stack([b[5] for b in batch])  # [batch_size, k]

    # Process complex lists
    i_neg_idx_tensor = torch.zeros((batch_size, k), dtype=torch.long)
    i_neg_audio_embeds_tensor = torch.zeros((batch_size, k, embed_dim), dtype=torch.float)
    i_neg_is_cold_start_tensor = torch.zeros((batch_size, k), dtype=torch.bool)

    for i, (u, i_pos, neg_indices, neg_embeds, pos_w, neg_w) in enumerate(batch):
        i_neg_idx_tensor[i] = torch.tensor(neg_indices, dtype=torch.long)
        for j, embed in enumerate(neg_embeds):
            if embed is not None:
                i_neg_audio_embeds_tensor[i, j] = torch.tensor(embed, dtype=torch.float)
                i_neg_is_cold_start_tensor[i, j] = True

    return (
        u_idx,
        i_pos_idx,
        i_neg_idx_tensor,  # [B, k] (has -1 for cold-start)
        i_neg_audio_embeds_tensor,  # [B, k, D] (zeros for in-graph)
        i_neg_is_cold_start_tensor,  # [B, k] (bool mask)
        pos_weights,
        neg_weights
    )


class GNNTrainer:
    """
    UNIFIED (Request 1): Trainer for BPR loss,
    compatible with LightGCN class.

    This trainer forces the model and graph to the CPU to avoid OOM errors.
    It performs one slow, full-graph `forward_cpu()` pass at the start of each
    epoch and then calculates BPR loss in batches using the pre-computed
    CPU embeddings. This is the logic from the "committed" code.
    """

    def __init__(self, model: LightGCN, train_graph, config: Config):
        self.config = config

        # --- UNIFIED (Request 1): CPU-ONLY Strategy ---
        # Force to CPU to avoid OOM errors on the full-graph pass
        self.device = torch.device('cpu')
        print(f"--- GNNTrainer: Forcing all training to CPU to avoid OOM. ---")
        self.model = model.to(self.device)
        self.train_graph = train_graph.to(self.device)
        # --- END FIX ---

        self.batch_size = config.gnn.batch_size
        self.num_epochs = config.gnn.num_epochs
        self.save_path = config.paths.trained_gnn
        self.neg_samples_per_pos = config.gnn.neg_samples_per_pos  # Used by dataset
        self.embed_dim = config.gnn.embed_dim  # Store embed_dim

        # --- Load and process negative interactions ONCE ---
        print(">>> GNNTrainer: Building item ID to graph index map...")
        # .cpu() is important if graph was loaded to GPU
        item_original_ids = train_graph['item'].item_id.cpu().numpy()
        item_id_to_graph_idx = {item_id: i for i, item_id in enumerate(item_original_ids)}
        del item_original_ids  # Free memory

        raw_user2neg_map = defaultdict(list)

        # 1. Load IN-GRAPH negatives (lightweight)
        in_graph_neg_file = config.paths.negative_train_in_graph_file
        print(f">>> GNNTrainer: Loading IN-GRAPH negatives from {in_graph_neg_file}...")
        neg_in_graph_df = pd.read_parquet(in_graph_neg_file, columns=['user_id', 'item_id'])

        print(">>> GNNTrainer: Processing IN-GRAPH negatives...")
        for row in neg_in_graph_df.itertuples(index=False):
            item_train_idx = item_id_to_graph_idx.get(row.item_id, -1)
            # Should always be found, but check just in case
            if item_train_idx != -1:
                raw_user2neg_map[row.user_id].append((item_train_idx, None))
        del neg_in_graph_df  # Free memory

        # 2. Load COLD-START negatives (heavy but smaller)
        cold_start_neg_file = config.paths.negative_train_cold_start_file
        print(f">>> GNNTrainer: Loading COLD-START negatives from {cold_start_neg_file}...")
        neg_cold_start_df = pd.read_parquet(cold_start_neg_file, columns=['user_id', 'normalized_embed'])

        print(">>> GNNTrainer: Processing COLD-START negatives...")
        for row in neg_cold_start_df.itertuples(index=False):
            # item_train_idx is -1, and we pass the embedding
            raw_user2neg_map[row.user_id].append((-1, row.normalized_embed))
        del neg_cold_start_df  # Free memory

        print(f">>> GNNTrainer: Loaded {len(raw_user2neg_map)} users with hard negatives.")
        # --- END NEW ---

        # --- UNIFIED (Request 2): Use the Advanced BPRDataset ---
        print(">>> GNNTrainer: Initializing BPRDataset...")
        self.dataset = BPRDataset(train_graph, config, raw_user2neg_map)

        # Get node counts from model and dataset
        self.num_users = self.model.num_users
        self.num_items = self.dataset.num_items
        self.total_nodes = self.num_users + self.num_items

        # --- DATALOADER ---
        # pin_memory = 'cuda' in str(self.device) # <-- Cannot pin memory for CPU
        pin_memory = False
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=config.gnn.num_workers,
            collate_fn=collate_bpr_advanced,  # Use the new collate fn
            pin_memory=pin_memory,
            persistent_workers=True if config.gnn.num_workers > 0 else False,
            drop_last=True
        )
        # --- END DATALOADER ---

        # Get data from the *model* buffers, which are already homogeneous
        self.edge_index_full = self.model.edge_index.cpu()
        self.edge_weight_full = self.model.edge_weight_init.cpu()

        self.lr_base = config.gnn.lr
        self.lr_decay = config.gnn.lr_decay
        self.weight_decay = config.gnn.weight_decay
        self.max_grad_norm = config.gnn.max_grad_norm

        self.dropout = config.gnn.dropout
        self.margin = config.gnn.margin

        self.step_count = 0
        self.warmup_steps = len(self.loader)
        self.accum_steps = config.gnn.accum_steps

        self.best_ndcg = 0.0
        self.best_metrics = None

    def _get_lr(self, epoch):
        """
        Learning rate schedule with warmup and CYCLICAL decay (Cosine Annealing).
        """
        # --- WARMUP (same as before) ---
        if self.step_count == 0:
            return self.lr_base * (1 / self.warmup_steps)
        if self.step_count < self.warmup_steps:
            return self.lr_base * (self.step_count / self.warmup_steps)

        # --- CYCLICAL DECAY (NEW) ---
        epochs_per_cycle = 5
        current_epoch_in_cycle = (epoch - (self.warmup_steps / len(self.loader))) % epochs_per_cycle
        min_lr = 1e-6  # A very small minimum LR
        cos_inner = (math.pi * current_epoch_in_cycle) / epochs_per_cycle
        current_lr = min_lr + (self.lr_base - min_lr) * (1 + math.cos(cos_inner)) / 2
        return current_lr

    def _update_parameters(self, lr):
        """
        Pure SGD update with per-parameter LR and weight decay.
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue

                if 'user_emb' in name:
                    param_lr = lr * 1.0
                elif 'artist_emb' in name or 'album_emb' in name:
                    param_lr = lr * 2.0
                else:
                    param_lr = lr

                grad = param.grad.data
                if self.weight_decay > 0 and 'emb' in name:
                    grad = grad.add(param.data, alpha=self.weight_decay)

                param.data.add_(grad, alpha=-param_lr)

    def train(self, trial=False):
        print(f">>> starting training with ADVANCED BPR ranking loss (on CPU)")
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.model.to(self.device)  # Ensure model is on CPU

        patience = 0
        max_patience = 10

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()

            epoch_loss = 0.0
            epoch_grad_norm = 0.0
            num_batches = 0

            current_lr = self._get_lr(epoch)

            # --- UNIFIED (Request 1): EPOCH-LEVEL FORWARD PASS ---
            print(f">>> Epoch {epoch}: Running full-graph CPU forward pass...")
            with torch.no_grad():  # No gradients needed for this part
                # This calls the GNN-propagation
                all_user_embs_final, all_item_embs_final, _ = self.model.forward_cpu()
            print(f">>> CPU forward pass complete. Starting BPR batches...")
            # all_user_embs_final and all_item_embs_final are now on CPU
            # --- END UNIFIED ---

            progress = tqdm(self.loader, desc=f"Epoch {epoch} (LR={current_lr:.6f})", leave=True)

            for batch_idx, batch in enumerate(progress):
                # --- 1. Unpack batch ---
                (u_idx, i_pos_idx,
                 i_neg_idx_tensor, i_neg_audio_embeds_tensor, i_neg_is_cold_start_tensor,
                 pos_weights, neg_weights) = batch

                # --- 2. Build Subgraph (REMOVED) ---
                # All subgraph logic is removed, as we now use the pre-computed embeddings

                # --- 3. Forward Pass (REMOVED) ---
                # We no longer call forward_subgraph

                # --- 4. Map nodes and Calculate BPR Loss (Using pre-computed embs) ---

                # Move all batch tensors to device (which is CPU)
                u_idx_batch = u_idx.to(self.device)
                i_pos_idx_batch = i_pos_idx.to(self.device)
                i_neg_idx_batch = i_neg_idx_tensor.to(self.device)
                neg_audio_embeds_batch = i_neg_audio_embeds_tensor.to(self.device)
                neg_is_cold_start_batch = i_neg_is_cold_start_tensor.to(self.device)
                pos_weights_batch = pos_weights.to(self.device)
                neg_weights_batch = neg_weights.to(self.device)

                # --- Get Positive Embeddings ---
                # We lookup from the pre-computed full-graph embeddings
                u_emb = all_user_embs_final[u_idx_batch]  # [B, D]
                pos_i_emb = all_item_embs_final[i_pos_idx_batch]  # [B, D]

                # --- Get Negative Embeddings (All 3 Cases) ---
                neg_i_emb = torch.zeros(
                    self.batch_size,
                    self.neg_samples_per_pos,
                    self.embed_dim,
                    device=self.device
                )

                # Case 3: Cold-Start (Not in graph)
                # Use pre-computed audio embedding
                neg_i_emb[neg_is_cold_start_batch] = neg_audio_embeds_batch[neg_is_cold_start_batch]

                # --- Handle In-Graph Negatives (Case 1 & 2) ---
                # This logic is now much simpler.
                in_graph_mask = ~neg_is_cold_start_batch
                in_graph_nodes_flat = i_neg_idx_batch[in_graph_mask]  # 1D tensor of in-graph node indices

                if in_graph_nodes_flat.numel() > 0:
                    # All in-graph nodes have a pre-computed final embedding.
                    # We just look them up from the full item embedding table.
                    in_graph_embeds_flat = all_item_embs_final[in_graph_nodes_flat]

                    # Scatter flat results back to [B, k, D] shape
                    neg_i_emb[in_graph_mask] = in_graph_embeds_flat

                # --- ADD NODE/EMBEDDING DROPOUT ---
                if self.model.training:
                    u_emb = F.dropout(u_emb, p=self.dropout, training=True)
                    pos_i_emb = F.dropout(pos_i_emb, p=self.dropout, training=True)
                    neg_i_emb = F.dropout(neg_i_emb, p=self.dropout, training=True)
                # --- [ END NEW ] ---

                # --- Calculate Weighted BPR Loss ---
                pos_scores = (u_emb * pos_i_emb).sum(dim=-1, keepdim=True)  # [B, 1]
                neg_scores = (u_emb.unsqueeze(1) * neg_i_emb).sum(dim=-1)  # [B, k]

                diff = pos_scores - neg_scores - self.margin
                loss_per_neg = -F.logsigmoid(diff)  # [B, k]

                # Apply weights
                # pos_weights_batch is now the continuous hybrid ratio (0.0-1.0)
                weighted_loss = loss_per_neg * pos_weights_batch.unsqueeze(1) * neg_weights_batch
                loss = weighted_loss.mean()
                # --- End Loss Calculation ---

                if not torch.isfinite(loss):
                    print(f"\n!!! NaN/Inf detected in BPR loss at epoch {epoch}, batch {batch_idx}!!!")
                    raise ValueError("NaN/Inf in BPR loss. Stopping training.")

                loss = loss / self.accum_steps
                loss.backward()

                # --- 5. Optimizer Step ---
                if (batch_idx + 1) % self.accum_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

                    if not np.isfinite(grad_norm.item()):
                        print(f"\n!!! NaN/Inf GRADIENT detected at epoch {epoch}, batch {batch_idx}!!!")
                        self.model.zero_grad(set_to_none=True)
                    else:
                        self.step_count += 1
                        current_step_lr = self._get_lr(epoch)
                        self._update_parameters(current_step_lr)
                        epoch_grad_norm += grad_norm.item()

                    self.model.zero_grad(set_to_none=True)

                epoch_loss += loss.item() * self.accum_steps
                num_batches += 1

                progress.set_postfix({
                    'bpr_loss': f'{epoch_loss / num_batches:.6f}',
                })

            self.model.zero_grad(set_to_none=True)
            progress.close()
            avg_loss = epoch_loss / num_batches
            avg_grad = epoch_grad_norm / (num_batches / self.accum_steps)

            print(f"Epoch {epoch} | BPR Loss: {avg_loss:.6f} | Avg grad: {avg_grad:.4f}")

            # --- 6. Evaluation ---
            self.model.eval()
            # GNNEvaluator already uses model.forward_cpu(), so it's compatible
            val_evaluator = GNNEvaluator(self.model, self.train_graph, "val", self.config)
            val_metrics = val_evaluator.evaluate()
            cur_ndcg = val_metrics['ndcg@k']

            # --- UNIFIED (Request 4): Also log the raw NDCG ---
            cur_ndcg_raw = val_metrics['ndcg_raw@k']
            print(f"Epoch {epoch} | NDCG@K: {cur_ndcg:.6f} (Novelty) | {cur_ndcg_raw:.6f} (Raw)")
            # --- END UNIFIED ---

            if cur_ndcg > self.best_ndcg:
                improvement = cur_ndcg - self.best_ndcg
                self.best_ndcg = cur_ndcg
                self.best_metrics = val_metrics
                patience = 0
                if not trial:
                    torch.save(self.model.state_dict(), self.save_path)
                    print(f"> Best model saved ({improvement:.6f})")
            else:
                patience += 1
                print(f"No improvement ({patience}/{max_patience})")
                if patience >= max_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        print(f"\n>>> finished training")

        print(f"Best NDCG@K: {self.best_ndcg:.6f}")
        if not trial:
            with open(self.config.paths.val_eval, "w") as f:
                json.dump(self.best_metrics, f, indent=4)
            print(f"Model saved to {self.save_path}")