import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict, Tuple
from torch_geometric.data import HeteroData
from GNN_model.GNN_class import LightGCN
from config import Config


class GNNEvaluator:
    """
    Evaluator class for GNN-based models.
    """
    def __init__(self, model: LightGCN, graph: HeteroData, eval_set: str, config: Config):
        """
        Args:
            model: trained GNN model
            graph: PyG HeteroData graph (needed for full user/item embeddings)
            eval_set: 'val' or 'test' identifier for file paths
            config: configuration object
        """
        self.device = config.gnn.device
        self.model = model.to(self.device)
        self.scores_path = getattr(config.paths, f"{eval_set}_scores_file")
        self.top_k = config.gnn.k_hit
        self.eval_batch_size = config.gnn.eval_batch_size

        self.gpu_safe_batch_size = 64

        self.num_users = graph['user'].num_nodes
        self.num_items = graph['item'].num_nodes

        # Caches
        self._cached_user_emb = None
        self._cached_item_emb = None
        self._orig_id_to_graph_idx = None

        # Data Loading
        self.ground_truth = {}
        self.eval_user_indices = np.array([])
        self._load_ground_truth_data()


    def _load_ground_truth_data(self):
        """Loads the pre-computed scores and processes them into a lookup dict."""
        df = pd.read_parquet(self.scores_path)
        df = df.rename(columns={"user_id": "user_idx", "item_id": "item_idx"})

        # Debug for dislikes
        total_dislikes = (df['adjusted_score'] < 0).sum()
        print(f"DEBUG (Internal GNN): Found {total_dislikes} Dislike interactions (score < 0) in {self.scores_path}")

        self.ground_truth = {}

        # Group by user for fast lookup
        for uid, group in df.groupby('user_idx'):
            self.ground_truth[uid] = {
                "item_idx": group["item_idx"].values,  # Original IDs
                "adjusted_score": group["adjusted_score"].values.astype(float),
                "base_relevance": group["base_relevance"].values.astype(float),
                "listen_plus_relevance": group["listen_plus_relevance"].values.astype(float),
                "like_relevance": group["like_relevance"].values.astype(float),
                "seen_in_train": group["seen_in_train"].values.astype(bool)
            }

        self.eval_user_indices = np.array(list(self.ground_truth.keys()), dtype=np.int64)


    def _ensure_embeddings_and_mapper(self):
        """Ensures embeddings are generated (on CPU) and ID mapper is built."""
        if self._cached_user_emb is None or self._cached_item_emb is None:
            # 1. Run Inference on CPU to avoid OOM during full-graph prop
            self.model.to('cpu')
            self.model.eval()
            with torch.no_grad():
                user_emb, item_emb, _ = self.model.forward_cpu()
                self._cached_user_emb = user_emb.cpu()
                self._cached_item_emb = item_emb.cpu()

            # Restore model to device
            self.model.to(self.device)

        if self._orig_id_to_graph_idx is None:
            # 2. Build ID mapping
            item_orig_ids = self.model.item_original_ids.cpu()
            self._orig_id_to_graph_idx = {
                orig_id.item(): graph_idx
                for graph_idx, orig_id in enumerate(item_orig_ids)
            }


    def _pre_map_ground_truth_items(self):
        """
        Converts original item IDs in ground_truth to graph indices.
        Filters out items that don't exist in the current graph.
        """
        mapper = self._orig_id_to_graph_idx

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

            # Update dict with mapped and filtered data
            gt["graph_idx"] = np.array(graph_indices, dtype=np.int64)
            gt["valid_adj_scores"] = gt["adjusted_score"][valid_indices]
            gt["valid_base_rel"] = gt["base_relevance"][valid_indices]
            gt["valid_listen_plus_rel"] = gt["listen_plus_relevance"][valid_indices]
            gt["valid_like_rel"] = gt["like_relevance"][valid_indices]
            gt["valid_seen"] = gt["seen_in_train"][valid_indices]


    def _predict_batch_top_k(self, batch_user_indices: np.ndarray, item_emb_gpu: torch.Tensor) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Computes dot-product scores and retrieves Top-K indices for a batch of users.
        Uses mini-batching internally to prevent GPU OOM errors.

        Args:
            batch_user_indices: array of user indices to process
            item_emb_gpu: pre-computed item embeddings on GPU (already moved to device)

        Returns:
            top-k recommendations and predicted scores for each user in the batch
        """
        all_topk = []
        all_scores = []

        # Mini-batch processing
        for j in range(0, len(batch_user_indices), self.gpu_safe_batch_size):
            mini_idx = batch_user_indices[j: j + self.gpu_safe_batch_size]

            # Move only specific users to GPU
            mini_user_emb = self._cached_user_emb[mini_idx].to(self.device)

            # Matrix Multiplication [Batch, Items]
            scores = torch.mm(mini_user_emb, item_emb_gpu.T)

            # Top-K
            _, topk_indices = torch.topk(scores, self.top_k, dim=-1)

            all_topk.append(topk_indices.cpu().numpy())
            all_scores.append(scores.cpu().numpy())

        return np.concatenate(all_topk, axis=0), np.concatenate(all_scores, axis=0)


    def _calc_ndcg(self, topk_idx: np.ndarray, gt_indices: np.ndarray, gt_scores: np.ndarray) -> float:
        """
        Calculates NDCG@K for a single user given specific ground truth scores.

        Args:
            topk_idx: array of top-k item indices
            gt_indices: array of original item IDs in ground truth
            gt_scores: array of ground truth relevance scores for each item (adjusted, base, etc.)

        Returns:
            NDCG@k score for the given user and top-k items
        """
        k = self.top_k

        # Sparse lookup optimized
        top_rel = np.zeros(len(topk_idx))
        gt_map = {idx: score for idx, score in zip(gt_indices, gt_scores)}

        for i, item_idx in enumerate(topk_idx):
            top_rel[i] = gt_map.get(item_idx, 0.0)

        # DCG
        dcg = np.sum(top_rel / np.log2(np.arange(2, k + 2)))

        # IDCG
        ideal = np.sort(np.maximum(gt_scores, 0))[::-1][:k]
        idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))

        return dcg / (idcg if idcg > 0 else 1.0)


    def _compute_metrics_for_user(self, user_idx: int, topk_idx: np.ndarray, pred_scores: np.ndarray) -> Dict[
        str, float]:
        """
        Calculates all metrics for a single user.

        Args:
            user_idx: user ID
            topk_idx: array of top-k item indices
            pred_scores: array of predicted scores for each item in top-k

        Returns:
            dict containing all metrics for the given user and top-k items
        """
        gt = self.ground_truth[user_idx]
        gt_items = gt["graph_idx"]
        gt_adj = gt["valid_adj_scores"]
        gt_base = gt["valid_base_rel"]
        gt_listen_plus = gt["valid_listen_plus_rel"]
        gt_like = gt["valid_like_rel"]
        gt_seen = gt["valid_seen"]

        topk_set = set(topk_idx)
        metrics = {}

        # 1. NDCG (Adjusted & Raw)
        metrics["ndcg@k"] = self._calc_ndcg(topk_idx, gt_items, gt_adj)
        metrics["ndcg_raw@k"] = self._calc_ndcg(topk_idx, gt_items, gt_base)
        metrics["ndcg_listen_plus@k"] = self._calc_ndcg(topk_idx, gt_items, gt_listen_plus)
        metrics["ndcg_like@k"] = self._calc_ndcg(topk_idx, gt_items, gt_like)

        # 2. Hits
        # Hit Like (Explicit > 1.0)
        like_items = gt_items[gt_adj >= 1.0]
        metrics["hit_like@k"] = float(len(set(like_items) & topk_set) > 0)

        # Hit Like+Listen (Implicit > 0.5)
        pos_items = gt_items[gt_adj > 0.5]
        metrics["hit_like_listen@k"] = float(len(set(pos_items) & topk_set) > 0)

        # 3. Dislike Rate (Precision of Negative Items)
        dislike_items = gt_items[gt_adj < 0]
        metrics["dislike_rate@k"] = 0.0
        if len(dislike_items) > 0:
            bad_recs_count = len(set(dislike_items) & topk_set)
            metrics["dislike_rate@k"] = bad_recs_count / len(topk_idx)

        # 4. Novelty (Based on Training Data)
        # Identify items in ground truth that were SEEN in train
        train_items = set(gt_items[gt_seen])

        if len(topk_idx) > 0:
            # Count recommended items that are NOT in the training set
            new_items = [i for i in topk_idx if i not in train_items]
            metrics["novelty@k"] = len(new_items) / len(topk_idx)
        else:
            metrics["novelty@k"] = 0.0

        # 5. AUC (Positive vs Negative discrimination)
        pos_mask = gt_adj > 0
        neg_mask = gt_adj <= 0

        metrics["auc"] = np.nan  # Default if cannot calc
        if np.any(pos_mask) and np.any(neg_mask):
            pos_items_auc = gt_items[pos_mask]
            neg_items_auc = gt_items[neg_mask]

            # Construct labels
            y_true = np.concatenate([np.ones(len(pos_items_auc)), np.zeros(len(neg_items_auc))])

            # Extract predicted scores for these specific items
            y_score = np.concatenate([pred_scores[pos_items_auc], pred_scores[neg_items_auc]])

            try:
                metrics["auc"] = roc_auc_score(y_true, y_score)
            except ValueError:
                pass  # Happens if all scores are identical

        return metrics


    def evaluate(self) -> Dict[str, float]:
        """
        Main entry point: Batched, GPU-accelerated evaluation.

        Returns:
            dict containing all metrics calculated for all users in the dataset.
        """
        print(">>> Starting evaluation (batched, GPU)...")

        # 1. Prepare Data
        self._ensure_embeddings_and_mapper()
        self._pre_map_ground_truth_items()

        # Move full item embeddings to GPU once
        item_emb_gpu = self._cached_item_emb.to(self.device)

        # 2. Accumulate Results
        agg_metrics = {
            "ndcg@k": [], "ndcg_raw@k": [],
            "ndcg_listen_plus@k": [], "ndcg_like@k": [], "hit_like@k": [],
            "hit_like_listen@k": [], "auc": [], "dislike_rate@k": [], "novelty@k": []
        }

        # 3. Process Users in Batches
        total_users = len(self.eval_user_indices)

        for i in range(0, total_users, self.eval_batch_size):
            batch_indices = self.eval_user_indices[i: i + self.eval_batch_size]

            # A. Predict (Handles OOM internally)
            batch_topk, batch_scores = self._predict_batch_top_k(batch_indices, item_emb_gpu)

            # B. Calculate Metrics (CPU)
            for j, user_idx in enumerate(batch_indices):
                user_metrics = self._compute_metrics_for_user(
                    user_idx,
                    batch_topk[j],
                    batch_scores[j]
                )

                # Append to aggregators
                for key, val in user_metrics.items():
                    if not np.isnan(val):  # Skip NaNs (e.g. from failed AUC)
                        agg_metrics[key].append(val)

        # 4. Average
        print(">>> Evaluation complete.")
        final_results = {m: float(np.mean(v)) if len(v) else 0.0 for m, v in agg_metrics.items()}
        return final_results