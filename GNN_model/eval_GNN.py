import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict
from torch_geometric.data import HeteroData
from config import Config


class GNNEvaluator:
    def __init__(self, model: torch.nn.Module, graph: HeteroData, eval_set: str, config: Config):
        """
        Args:
            model: trained GNN model
            graph: PyG HeteroData graph (needed for full user/item embeddings)
            device: cpu or cuda
        """
        self.device = config.gnn.device
        self.model = model.to(self.device)  # model lives on GPU for training
        self.scores_path = getattr(config.paths, f"{eval_set}_scores_file")
        self.top_k = config.gnn.k_hit
        self.eval_batch_size = config.gnn.eval_batch_size

        self.num_users = graph['user'].num_nodes
        self.num_items = graph['item'].num_nodes

        # Cache for embeddings
        self._cached_embeddings = None
        self._orig_id_to_graph_idx = None

        # Pre-load and process ground truth relevance scores
        self.ground_truth = {}
        self.eval_user_indices = np.array([])
        self._load_and_process_ground_truth()

    def _load_and_process_ground_truth(self):
        """
        UNIFIED (Request 4): Load the pre-computed scores
        (both adjusted_score and base_relevance)
        and pre-process into a dict.
        """
        df = pd.read_parquet(self.scores_path)
        # We expect columns 'user_id', 'item_id', 'adjusted_score', 'base_relevance', 'seen_in_train'
        df = df.rename(columns={"user_id": "user_idx", "item_id": "item_idx"})

        # Create a fast lookup dictionary for ground truth
        self.ground_truth = {}

        # Group by user_idx
        for uid, group in df.groupby('user_idx'):
            self.ground_truth[uid] = {
                "item_idx": group["item_idx"].values,  # Original item IDs
                # --- UNIFIED (Request 4): Load both scores ---
                "adjusted_score": group["adjusted_score"].values, # Novelty-adjusted
                "base_relevance": group["base_relevance"].values, # Raw relevance
                # --- END UNIFIED ---
                "seen_in_train": group["seen_in_train"].values.astype(bool)
            }

        self.eval_user_indices = np.array(list(self.ground_truth.keys()), dtype=np.int64)

    def _get_embeddings(self):
        """
        Run full-graph propagation once on CPU (to avoid OOM)
        and cache embeddings.
        """
        if self._cached_embeddings is None:
            # --- REVERTED TO CPU FORWARD PASS ---
            # This is compatible with the CPU-based training loop
            self.model.to('cpu')
            self.model.eval()
            with torch.no_grad():
                # Use the CPU-based forward pass
                user_emb_cpu, item_emb_cpu, _ = self.model.forward_cpu()
                # Cache them on the CPU
                self._cached_embeddings = (user_emb_cpu.cpu(), item_emb_cpu.cpu())

            # Move model back to GPU for training (if it was there)
            self.model.to(self.device)
            # --- END REVERT ---

            if self._orig_id_to_graph_idx is None:
                # Get the original IDs tensor (size=num_items) stored in the model
                item_orig_ids_tensor = self.model.item_original_ids.cpu()

                # Create a mapping from original_id -> graph_idx
                self._orig_id_to_graph_idx = {
                    orig_id.item(): graph_idx
                    for graph_idx, orig_id in enumerate(item_orig_ids_tensor)
                }

        return self._cached_embeddings

    def _pre_map_ground_truth(self, mapper):
        """
        Uses the embedding mapper to convert all original item IDs
        in the ground truth dict to the model's graph indices.
        This is done once to avoid repeated .map() calls.
        """
        for uid in self.eval_user_indices:
            gt = self.ground_truth[uid]
            orig_ids = gt["item_idx"]

            valid_indices = []
            graph_indices = []

            for i, orig_id in enumerate(orig_ids):
                graph_idx = mapper.get(orig_id)
                if graph_idx is not None:
                    valid_indices.append(i)
                    graph_indices.append(graph_idx)

            # Store the mapped and filtered ground truth
            gt["graph_idx"] = np.array(graph_indices, dtype=np.int64)
            # --- UNIFIED (Request 4): Store both scores ---
            gt["valid_adj_scores"] = gt["adjusted_score"][valid_indices]
            gt["valid_base_rel"] = gt["base_relevance"][valid_indices]
            # --- END UNIFIED ---
            gt["valid_seen"] = gt["seen_in_train"][valid_indices]

    def evaluate(self) -> Dict[str, float]:
        """
        Batched, GPU-accelerated evaluation.
        """
        print(">>> Starting evaluation (batched, GPU)...")
        k = self.top_k

        # 1. Get Embeddings (on CPU) and Mapper
        user_emb_cpu, item_emb_cpu = self._get_embeddings()
        mapper = self._orig_id_to_graph_idx

        # 2. Pre-map all ground truth item IDs
        self._pre_map_ground_truth(mapper)

        # --- MOVE FULL ITEM TENSOR TO GPU (ONCE) ---
        item_emb_gpu = item_emb_cpu.to(self.device)

        # 3. Initialize metrics
        metrics = {
            "ndcg@k": [],
            # --- UNIFIED (Request 4): Add raw ndcg metric ---
            "ndcg_raw@k": [],
            # --- END UNIFIED ---
            "hit_like@k": [],
            "hit_like_listen@k": [],
            "auc": [],
            "dislike_fpr@k": [],
            "novelty@k": []
        }
        
        # --- OOM FIX: Use a smaller, OOM-safe batch size for GPU scoring ---
        # This is much faster than 1, but much safer than 512.
        EVAL_OOM_SAFE_BATCH_SIZE = 64 
        # --- END FIX ---

        # 4. Process users in batches
        # This outer loop still uses the config batch size, e.g., 512.
        # It defines the "chunk" of users we process from CPU memory.
        for i in range(0, len(self.eval_user_indices), self.eval_batch_size):
            batch_user_indices = self.eval_user_indices[i: i + self.eval_batch_size]

            # --- START OOM FIX: Re-batch the GPU work ---
            # We have 512 user indices. Process them in mini-batches of 64 on the GPU.
            all_batch_topk_indices_cpu = []
            all_batch_scores_cpu = []

            for j in range(0, len(batch_user_indices), EVAL_OOM_SAFE_BATCH_SIZE):
                mini_batch_indices = batch_user_indices[j : j + EVAL_OOM_SAFE_BATCH_SIZE]
                
                # Move a mini-batch of user embeddings to GPU
                mini_batch_user_emb_gpu = user_emb_cpu[mini_batch_indices].to(self.device) # [64, D]

                # --- Perform scoring for the mini-batch on GPU ---
                # This creates a [64, 934k] tensor. Much safer.
                mini_batch_scores_gpu = torch.mm(mini_batch_user_emb_gpu, item_emb_gpu.T)

                # --- Get Top-K on GPU ---
                _, mini_batch_topk_indices_gpu = torch.topk(mini_batch_scores_gpu, k, dim=-1)

                # --- Move results to CPU and append ---
                all_batch_topk_indices_cpu.append(mini_batch_topk_indices_gpu.cpu().numpy())
                all_batch_scores_cpu.append(mini_batch_scores_gpu.cpu().numpy())

            # Now combine the results from all mini-batches
            batch_topk_indices_cpu = np.concatenate(all_batch_topk_indices_cpu, axis=0)
            batch_scores_cpu = np.concatenate(all_batch_scores_cpu, axis=0)
            # --- END OOM FIX ---


            # 5. Calculate metrics for each user in the batch (on CPU)
            # This loop now just iterates over the CPU results
            for j, user_idx in enumerate(batch_user_indices):
                topk_idx = batch_topk_indices_cpu[j]
                topk_set = set(topk_idx)

                # Get pre-processed ground truth
                gt = self.ground_truth[user_idx]
                gt_items = gt["graph_idx"]  # Already mapped
                # --- UNIFIED (Request 4): Get both scores ---
                gt_adj = gt["valid_adj_scores"]  # Already filtered (novelty-adjusted)
                gt_base_rel = gt["valid_base_rel"]  # Already filtered (raw)
                # --- END UNIFIED ---

                # --- NDCG@k (graded, based on adjusted_score) ---
                relevance = np.zeros(self.num_items, dtype=float)
                relevance[gt_items] = gt_adj

                top_rel = relevance[topk_idx]
                dcg = np.sum(top_rel / np.log2(np.arange(2, k + 2)))
                ideal = np.sort(np.maximum(gt_adj, 0))[::-1][:k]
                idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
                ndcg = dcg / (idcg if idcg > 0 else 1.0)
                metrics["ndcg@k"].append(ndcg)

                # --- UNIFIED (Request 4): Calculate Raw NDCG@k (graded, based on base_relevance) ---
                relevance_raw = np.zeros(self.num_items, dtype=float)
                relevance_raw[gt_items] = gt_base_rel  # Use base_relevance

                top_rel_raw = relevance_raw[topk_idx]
                dcg_raw = np.sum(top_rel_raw / np.log2(np.arange(2, k + 2)))
                ideal_raw = np.sort(np.maximum(gt_base_rel, 0))[::-1][:k]
                idcg_raw = np.sum(ideal_raw / np.log2(np.arange(2, len(ideal_raw) + 2)))
                ndcg_raw = dcg_raw / (idcg_raw if idcg_raw > 0 else 1.0)
                metrics["ndcg_raw@k"].append(ndcg_raw)
                # --- END UNIFIED ---

                # --- Hit@k (like-equivalent) ---
                like_items = gt_items[gt_adj > 1.0]  # >1 ˜ explicit like
                metrics["hit_like@k"].append(float(len(set(like_items) & topk_set) > 0))

                # --- Hit@k (like+listen) ---
                pos_items = gt_items[gt_adj > 0.5]
                metrics["hit_like_listen@k"].append(float(len(set(pos_items) & topk_set) > 0))

                # --- AUC (pos vs neg) ---
                # Use adjusted score for AUC calculation
                pos_items_auc = gt_items[gt_adj > 0]
                neg_items_auc = gt_items[gt_adj < 0]

                if len(pos_items_auc) > 0 and len(neg_items_auc) > 0:
                    y_true = np.concatenate([
                        np.ones(len(pos_items_auc)),
                        np.zeros(len(neg_items_auc))
                    ])

                    # Get the full score vector for this user
                    pred_scores_user_cpu = batch_scores_cpu[j]

                    y_score = np.concatenate([
                        pred_scores_user_cpu[pos_items_auc],
                        pred_scores__cpu[neg_items_auc]
                    ])

                    try:
                        metrics["auc"].append(roc_auc_score(y_true, y_score))
                    except ValueError:
                        pass  # Handle cases where all scores are identical

                # --- Dislike FPR@k ---
                dislike_items = gt_items[gt_adj < 0]
                if len(dislike_items):
                    metrics["dislike_fpr@k"].append(float(len(set(dislike_items) & topk_set) > 0))

                # --- Novelty@k (fraction unseen) ---
                all_interacted_items_set = set(gt_items)
                unseen_in_topk = sum(1 for i in topk_idx if i not in all_interacted_items_set)
                metrics["novelty@k"].append(unseen_in_topk / k)

        # ---- average over users ----
        print(">>> Evaluation complete.")
        return {m: float(np.mean(v)) if len(v) else 0.0 for m, v in metrics.items()}