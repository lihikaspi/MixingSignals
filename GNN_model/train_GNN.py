import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import HeteroData
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
    creates the BPR Dataset.
    """
    def __init__(self, train_graph: HeteroData, config: Config, raw_user2neg: dict):
        """
        Args:
            train_graph: HeteroData object containing the train graph.
            config: Config object containing the hyperparameters.
            raw_user2neg: dict containing the raw user-item negatives.
        """
        self.config = config
        self.num_items = int(train_graph['item'].num_nodes)
        self.raw_user2neg = raw_user2neg

        # Hyperparameters
        self.neutral_neg_weight = config.gnn.neutral_neg_weight
        self.neg_samples_per_pos = config.gnn.neg_samples_per_pos
        self.edge_type_mapping = config.preprocessing.edge_type_mapping
        self.min_listen_weight = config.gnn.min_listen_weight

        # Process graph data
        self.user2pos, self.user2pos_ratios, self.all_users_pos_sets = self._process_graph_edges(train_graph)

        # Process negatives
        self.user2neg = self._clean_negative_samples()

        self.users = sorted(list(self.user2pos.keys()))
        print(f"Advanced BPRDataset: Loaded {len(self.users)} users.")


    def _process_graph_edges(self, graph):
        """
        Parses graph edges to build positive interaction maps.

        Args:
            graph: HeteroData object containing the train graph.

        Returns:
            positive interaction maps.
        """
        edge_index = graph['user', 'interacts', 'item'].edge_index.cpu()
        edge_attr = graph['user', 'interacts', 'item'].edge_attr.cpu()

        # Indices for attributes (0=type, 2=ratio based on build_graph.py)
        # If config has it, use config, else default to implicit knowledge of build_graph
        idx_type = getattr(self.config.gnn, 'edge_attr_indices', {}).get('type', 0)
        idx_ratio = getattr(self.config.gnn, 'edge_attr_indices', {}).get('ratio', 2)

        types = edge_attr[:, idx_type]
        ratios = edge_attr[:, idx_ratio]

        user2pos = defaultdict(list)
        user2pos_ratios = defaultdict(list)
        pos_sets = defaultdict(set)

        pos_types = [self.edge_type_mapping[k] for k in ["listen", "like", "undislike"]]


        for u, i, t, r in zip(edge_index[0].tolist(), edge_index[1].tolist(), types.tolist(), ratios.tolist()):
            if int(round(t)) in pos_types:

                if int(round(t)) == self.edge_type_mapping["listen"]:
                    final_weight = self.min_listen_weight + (1.0 - self.min_listen_weight) * max(0.0, min(r, 1.0))
                else:
                    final_weight = r

                user2pos[u].append(i)
                user2pos_ratios[u].append(final_weight)
                pos_sets[u].add(i)

        return user2pos, user2pos_ratios, pos_sets


    def _clean_negative_samples(self):
        """
        Filters raw negatives to ensure they don't overlap with positives.

        Returns:
            cleaned negative interaction map.
        """
        clean_map = {}
        for u, neg_tuples in self.raw_user2neg.items():
            pos_set = self.all_users_pos_sets.get(u, set())
            valid_negs = []
            for (idx, embed) in neg_tuples:
                # idx -1 is cold start (always valid), otherwise check disjoint
                if idx == -1 or idx not in pos_set:
                    valid_negs.append((idx, embed))

            if valid_negs:
                clean_map[u] = valid_negs
        return clean_map


    def __len__(self):
        return len(self.users)


    def __getitem__(self, idx):
        u = self.users[idx]

        # 1. Sample Positive
        pos_idx = np.random.randint(len(self.user2pos[u]))
        i_pos = self.user2pos[u][pos_idx]
        pos_weight = self.user2pos_ratios[u][pos_idx]

        # 2. Sample Negatives
        hard_negs = self.user2neg.get(u, []).copy()
        pos_set = self.all_users_pos_sets[u]

        neg_indices, neg_embeds, neg_weights = [], [], []

        for _ in range(self.neg_samples_per_pos):
            if hard_negs:
                # Hard Negative (Explicit Dislike)
                sample = hard_negs.pop(np.random.randint(len(hard_negs)))
                neg_indices.append(sample[0])
                neg_embeds.append(sample[1])
                neg_weights.append(1.0)
            else:
                # Easy Negative (Random Sampling)
                while True:
                    rnd = np.random.randint(self.num_items)
                    if rnd not in pos_set:
                        neg_indices.append(rnd)
                        neg_embeds.append(None)
                        neg_weights.append(self.neutral_neg_weight)
                        break

        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(i_pos, dtype=torch.long),
            neg_indices, neg_embeds,
            torch.tensor(pos_weight, dtype=torch.float),
            torch.tensor(neg_weights, dtype=torch.float)
        )


def collate_bpr_advanced(batch):
    """
    Collates BPR batch, handling variable audio embeddings.

    Args:
        batch: list of tuples containing user, positive item, negative item, positive audio embedding, negative audio embedding, positive weight, negative weight.

    Returns:
        BPR dataset instances
    """
    batch_size = len(batch)
    k = len(batch[0][2])

    # Detect embedding dim
    embed_dim = 128
    for b in batch:
        for emb in b[3]:
            if emb is not None:
                embed_dim = len(emb)
                break

    u_idx = torch.stack([b[0] for b in batch])
    i_pos = torch.stack([b[1] for b in batch])
    pos_w = torch.stack([b[4] for b in batch])
    neg_w = torch.stack([b[5] for b in batch])

    i_neg = torch.zeros((batch_size, k), dtype=torch.long)
    neg_emb = torch.zeros((batch_size, k, embed_dim), dtype=torch.float)
    is_cold = torch.zeros((batch_size, k), dtype=torch.bool)

    for i, (_, _, indices, embeds, _, _) in enumerate(batch):
        i_neg[i] = torch.tensor(indices, dtype=torch.long)
        for j, emb in enumerate(embeds):
            if emb is not None:
                neg_emb[i, j] = torch.tensor(emb, dtype=torch.float)
                is_cold[i, j] = True

    return u_idx, i_pos, i_neg, neg_emb, is_cold, pos_w, neg_w


class GNNTrainer:
    """
    Trainer class for LightGCN model.
    """
    def __init__(self, model: LightGCN, train_graph: HeteroData, config: Config):
        """
        Args:
            model: LightGCN model instance.
            train_graph: HeteroData object containing the train graph.
            config: Config object containing the hyperparameters.
        """
        self.config = config
        self.device = torch.device('cpu')  # Force CPU for training loop
        self.model = model.to(self.device)
        self.train_graph = train_graph.to(self.device)

        self._init_hyperparameters(config)

        # Prepare Data
        neg_map = self._load_negative_interactions()
        self.dataset = BPRDataset(train_graph, config, neg_map)

        self.loader = DataLoader(
            self.dataset,
            batch_size=config.gnn.batch_size,
            shuffle=True,
            num_workers=config.gnn.num_workers,
            collate_fn=collate_bpr_advanced,
            pin_memory=False,
            drop_last=False
        )

        self.step_count = 0
        self.warmup_steps = len(self.loader)
        self.lr_cycle = config.gnn.lr_cycle

        # Metrics tracking
        self.best_ndcg = 0.0
        self.best_metrics = None


    def _init_hyperparameters(self, config):
        """Sets up training hyperparameters."""
        self.num_epochs = config.gnn.num_epochs
        self.save_path = config.paths.trained_gnn
        self.lr_base = config.gnn.lr
        self.accum_steps = config.gnn.accum_steps
        self.max_grad_norm = config.gnn.max_grad_norm
        self.weight_decay = config.gnn.weight_decay
        self.margin = config.gnn.margin
        self.dropout = config.gnn.dropout
        self.max_patience = config.gnn.max_patience


    def _load_negative_interactions(self):
        """Loads In-Graph and Cold-Start negatives into a unified map."""
        # Map item_id -> graph_idx
        item_ids = self.train_graph['item'].item_id.cpu().numpy()
        id_map = {uid: i for i, uid in enumerate(item_ids)}

        neg_map = defaultdict(list)

        # 1. In-Graph
        df_graph = pd.read_parquet(self.config.paths.negative_train_in_graph_file, columns=['user_id', 'item_id'])
        for row in df_graph.itertuples(index=False):
            idx = id_map.get(row.item_id, -1)
            if idx != -1:
                neg_map[row.user_id].append((idx, None))

        # 2. Cold-Start
        df_cold = pd.read_parquet(self.config.paths.negative_train_cold_start_file,
                                  columns=['user_id', 'normalized_embed'])
        for row in df_cold.itertuples(index=False):
            neg_map[row.user_id].append((-1, row.normalized_embed))

        return neg_map


    def _get_lr(self, epoch):
        """Cosine annealing with warmup."""
        if self.step_count < self.warmup_steps:
            return self.lr_base * (
                    self.step_count / self.warmup_steps) if self.step_count > 0 else self.lr_base / self.warmup_steps

        cycle_epoch = (epoch - (self.warmup_steps / len(self.loader))) % self.lr_cycle
        return 1e-6 + (self.lr_base - 1e-6) * (1 + math.cos(math.pi * cycle_epoch / self.lr_cycle)) / 2


    def _update_parameters(self, lr):
        """Manual SGD update."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is None: continue

                # Param-specific LR
                scale = 2.0 if 'artist_emb' in name or 'album_emb' in name else 0.5 if 'edge_mlp' in name else 1.0

                grad = param.grad.data
                if self.weight_decay > 0 and 'emb' in name:
                    grad = grad.add(param.data, alpha=self.weight_decay)

                param.data.add_(grad, alpha=-lr * scale)


    def _calc_bpr_loss(self, u_emb, pos_i_emb, neg_i_emb, pos_w, neg_w):
        """Computes weighted BPR loss."""
        pos_scores = (u_emb * pos_i_emb).sum(dim=-1, keepdim=True)
        neg_scores = (u_emb.unsqueeze(1) * neg_i_emb).sum(dim=-1)

        loss = -F.logsigmoid(pos_scores - neg_scores - self.margin)
        return (loss * pos_w.unsqueeze(1) * neg_w).mean()


    def train(self, trial=False):
        print(f">>> Starting CPU-based Training (Advanced BPR)")
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        patience = 0
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()

            # 1. Full Graph Forward (CPU) - Compute embeddings once per epoch
            print(f">>> Epoch {epoch}: Computing full graph embeddings...")
            u_all, i_all, _ = self.model.forward_cpu()

            # 2. Batch Training
            total_loss = 0
            progress = tqdm(self.loader, desc=f"Epoch {epoch}", leave=True)

            for i, batch in enumerate(progress):
                u_idx, pos_idx, neg_idx, neg_embs, is_cold, pos_w, neg_w = [x.to(self.device) for x in batch]

                # Lookup Embeddings
                u_emb = F.dropout(u_all[u_idx], p=self.dropout, training=True)
                pos_emb = F.dropout(i_all[pos_idx], p=self.dropout, training=True)

                # Assemble Negatives
                neg_emb_batch = torch.zeros(u_idx.size(0), self.dataset.neg_samples_per_pos, self.model.embed_dim,
                                            device=self.device)

                # Fill Cold-Start
                neg_emb_batch[is_cold] = neg_embs[is_cold]

                # Fill In-Graph
                mask_graph = ~is_cold
                flat_idx = neg_idx[mask_graph]
                if flat_idx.numel() > 0:
                    neg_emb_batch[mask_graph] = i_all[flat_idx]

                neg_emb_batch = F.dropout(neg_emb_batch, p=self.dropout, training=True)

                # Loss & Backprop
                loss = self._calc_bpr_loss(u_emb, pos_emb, neg_emb_batch, pos_w, neg_w)

                if not torch.isfinite(loss): raise ValueError("NaN loss")

                (loss / self.accum_steps).backward(retain_graph=True)

                # Calculate current LR for this step (using epoch info or warmup)
                current_lr = self._get_lr(epoch)

                if (i + 1) % self.accum_steps == 0:
                    self.step_count += 1
                    # Re-calculate LR if step count changed (mostly relevant during warmup)
                    current_lr = self._get_lr(epoch)
                    self._update_parameters(current_lr)
                    self.model.zero_grad(set_to_none=True)

                total_loss += loss.item()

                # Added 'lr' to the progress bar
                progress.set_postfix({'loss': f'{total_loss / (i + 1):.4f}', 'lr': f'{current_lr:.6f}'})

            # 3. Evaluation
            self.model.eval()
            val_metrics = GNNEvaluator(self.model, self.train_graph, "val", self.config).evaluate()
            ndcg = val_metrics['ndcg@k']
            print(f"Epoch {epoch} | NDCG: {ndcg:.4f}")

            # --- FIX: Ensure model is moved back to CPU for the next training epoch ---
            self.model.to(self.device)
            # ------------------------------------------------------------------------

            if ndcg > self.best_ndcg:
                self.best_ndcg = ndcg
                self.best_metrics = val_metrics  # <--- Store best metrics
                patience = 0
                if not trial:
                    torch.save(self.model.state_dict(), self.save_path)
                    print(f"> Best model saved ({ndcg:.4f})")
            else:
                patience += 1
                print(f"> no improvement ({patience}/{self.max_patience})")
                if patience >= self.max_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # 4. Save Best Metrics to JSON (End of Training)
        print(f"\n>>> Finished training. Best NDCG: {self.best_ndcg:.4f}")
        if not trial and self.best_metrics is not None:
            with open(self.config.paths.val_eval, "w") as f:
                json.dump(self.best_metrics, f, indent=4)
            print(f"Best metrics saved to {self.config.paths.val_eval}")

