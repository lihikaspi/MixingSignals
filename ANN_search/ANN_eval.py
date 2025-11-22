import numpy as np
import pandas as pd
from typing import Dict, List
import json
import os
import time
from config import Config


class RecEvaluator:
    """
    Evaluator class for ANN recommendations.
    Optimized for speed using Numpy slicing instead of Pandas groupby.
    """
    def __init__(self, recs, config: Config):
        self.gnn_recs = recs.get("gnn", {})
        self.content_recs = recs.get("content", {})
        self.popular_recs = recs.get("popular", {})
        self.random_recs = recs.get("random", {})
        self.cf_recs = recs.get("cf", {})

        self.top_k = config.ann.top_k
        self.test_k = config.ann.test_k
        self.relevance_scores = config.paths.test_scores_file
        self.base_eval_dir = config.paths.eval_dir
        self.mapping_path = config.paths.user_mapping

        print("Loading ground truth data (Optimized)...")
        self.ground_truth, self.all_test_users = self._load_ground_truth_fast()

        print("Loading User ID Mapping...")
        self.user_map = self._load_user_mapping()


    def _load_ground_truth_fast(self):
        """
        Loads ground truth using Numpy arrays and slicing.
        Much faster than pandas.groupby for large datasets.
        """
        t0 = time.time()
        df = pd.read_parquet(self.relevance_scores)
        df = df.rename(columns={"user_id": "user_idx", "item_id": "item_idx"})

        # Check for dislikes
        total_dislikes = (df['adjusted_score'] < 0).sum()
        print(
            f"DEBUG: Found {total_dislikes} Dislike interactions (score < 0) in Test Set out of {len(df)} total rows.")

        # Ensure sorted by user_idx for slicing
        df = df.sort_values("user_idx")

        # Extract columns as numpy arrays
        user_indices = df['user_idx'].values.astype(np.int64)
        item_indices = df['item_idx'].values.astype(np.int64)

        # Extract scores
        adj_arr = df['adjusted_score'].values.astype(float)
        base_arr = df['base_relevance'].values.astype(float)
        listen_arr = df['listen_plus_relevance'].values.astype(float)
        like_arr = df['like_relevance'].values.astype(float)

        # Check for seen_in_train for Novelty
        if 'seen_in_train' in df.columns:
            seen_arr = df['seen_in_train'].values.astype(bool)
        else:
            seen_arr = np.zeros_like(adj_arr, dtype=bool)

        # Find split points (indices where user_id changes)
        unique_users, start_indices = np.unique(user_indices, return_index=True)
        end_indices = np.append(start_indices[1:], len(user_indices))

        gt_lookup = {}

        # Build dictionary using slices (Fast)
        for i, uid in enumerate(unique_users):
            s, e = start_indices[i], end_indices[i]
            gt_lookup[uid] = {
                "items": item_indices[s:e],
                "adj": adj_arr[s:e],
                "base": base_arr[s:e],
                "listen_plus": listen_arr[s:e],
                "like": like_arr[s:e],
                "seen": seen_arr[s:e]
            }

        print(f"Loaded Ground Truth for {len(gt_lookup)} users in {time.time() - t0:.2f}s")
        return gt_lookup, unique_users


    def _load_user_mapping(self) -> Dict[int, int]:
        if not os.path.exists(self.mapping_path):
            print(f"WARNING: Mapping file not found at {self.mapping_path}. Output JSONs will use Encoded IDs.")
            return {}

        # Fast load
        df = pd.read_parquet(self.mapping_path)
        return dict(zip(df['user_id'], df['uid']))


    def _save_eval_results(self, results: dict, path: str):
        print(f"Saving results to {path}...")
        final_results = results.copy()

        # Swap keys: Encoded ID -> Original ID
        if self.user_map and "per_user" in results:
            new_per_user = {}
            # Pre-fetch get method for speed
            get_orig = self.user_map.get

            for encoded_id, metrics in results["per_user"].items():
                enc_id_int = int(encoded_id)
                orig_id = get_orig(enc_id_int, enc_id_int)
                new_per_user[str(orig_id)] = metrics

            final_results["per_user"] = new_per_user

        def convert(o):
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return o

        with open(path, "w") as f:
            json.dump(final_results, f, default=convert, indent=4)


    # --- Baselines ---
    def _popular_baseline(self):
        if self.popular_recs:
            self._save_eval_results(self._eval(self.popular_recs, "Popular"),
                                    self.eval_dir + "/popular_eval_results.json")


    def _random_baseline(self):
        if self.random_recs:
            self._save_eval_results(self._eval(self.random_recs, "Random"),
                                    self.eval_dir + "/random_eval_results.json")


    def _cf_baseline(self):
        if self.cf_recs:
            self._save_eval_results(self._eval(self.cf_recs, "CF"),
                                    self.eval_dir + "/cf_eval_results.json")


    def _content_baseline(self):
        if self.content_recs:
            self._save_eval_results(self._eval(self.content_recs, "Content"),
                                    self.eval_dir + "/content_eval_results.json")


    def _eval_gnn_recs(self):
        if self.gnn_recs:
            self._save_eval_results(self._eval(self.gnn_recs, "GNN"), self.eval_dir + "/gnn_eval_results.json")


    def _eval_baselines(self):
        self._popular_baseline()
        self._random_baseline()
        self._cf_baseline()
        self._content_baseline()


    # --- Metrics ---
    def _calc_ndcg(self, pred_rels, true_rels, k_val):
        # DCG
        gains = np.maximum(pred_rels, 0)
        dcg = np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))
        if dcg == 0: return 0.0

        # IDCG (Ideal sorting of all positive test items)
        ideal = np.sort(np.maximum(true_rels, 0))[::-1][:k_val]
        idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))

        return dcg / idcg if idcg > 0 else 0.0


    def _eval(self, recs: Dict[int, List[int]], name: str) -> Dict[str, Dict]:
        print(f"Evaluating {name}...")
        k = self.top_k
        per_user_metrics = {}

        # Pre-allocate lists
        keys = ["ndcg@k", "ndcg_raw@k", "ndcg_listen_plus@k", "ndcg_like@k",
                "hit_like@k", "hit_like_listen@k", "dislike_rate@k", "novelty@k"]
        all_metrics = {key: [] for key in keys}

        count = 0
        total = len(self.all_test_users)

        # Loop Optimization: Iterate over ALL test users
        for uid in self.all_test_users:
            gt_data = self.ground_truth[uid]
            rec_items = recs.get(uid, [])  # Default to empty if no recs

            topk_idx = np.array(rec_items[:k], dtype=np.int64)
            topk_set = set(topk_idx)

            gt_items = gt_data["items"]
            gt_adj = gt_data["adj"]
            gt_base = gt_data["base"]
            gt_listen = gt_data["listen_plus"]
            gt_like = gt_data["like"]
            gt_seen = gt_data["seen"]

            # Map GT items to their scores for fast lookup
            # Use dictionary for sparse lookup instead of dense array
            item_to_gt_idx = {item: i for i, item in enumerate(gt_items)}

            rel_adj = np.zeros(len(topk_idx), dtype=float)
            rel_base = np.zeros(len(topk_idx), dtype=float)
            rel_listen = np.zeros(len(topk_idx), dtype=float)
            rel_like = np.zeros(len(topk_idx), dtype=float)

            for i, item in enumerate(topk_idx):
                if item in item_to_gt_idx:
                    idx = item_to_gt_idx[item]
                    rel_adj[i] = gt_adj[idx]
                    rel_base[i] = gt_base[idx]
                    rel_listen[i] = gt_listen[idx]
                    rel_like[i] = gt_like[idx]

            m = {}
            m["ndcg@k"] = self._calc_ndcg(rel_adj, gt_adj, k)
            m["ndcg_raw@k"] = self._calc_ndcg(rel_base, gt_base, k)
            m["ndcg_listen_plus@k"] = self._calc_ndcg(rel_listen, gt_listen, k)
            m["ndcg_like@k"] = self._calc_ndcg(rel_like, gt_like, k)

            like_items_set = set(gt_items[gt_adj >= 1.0])
            m["hit_like@k"] = 1.0 if not like_items_set.isdisjoint(topk_set) else 0.0

            pos_items_set = set(gt_items[gt_adj >= 0.5])
            m["hit_like_listen@k"] = 1.0 if not pos_items_set.isdisjoint(topk_set) else 0.0

            # Dislike Rate (Count of dislikes in top k / k)
            dislike_mask = gt_adj < 0
            if np.any(dislike_mask):
                dislike_items_set = set(gt_items[dislike_mask])
                bad_recs_count = len(topk_set.intersection(dislike_items_set))
                m["dislike_rate@k"] = bad_recs_count / len(topk_idx) if len(topk_idx) > 0 else 0.0
            else:
                m["dislike_rate@k"] = 0.0

            # Novelty (Items NOT in training set)
            train_items_set = set(gt_items[gt_seen])
            if len(topk_idx) > 0:
                new_items = [x for x in topk_idx if x not in train_items_set]
                m["novelty@k"] = len(new_items) / len(topk_idx)
            else:
                m["novelty@k"] = 0.0

            per_user_metrics[uid] = m
            for k_metric, val in m.items():
                all_metrics[k_metric].append(val)

            count += 1
            if count % 1000 == 0:
                print(f"  > Processed {count}/{total} users...", end='\r')

        print(f"  > Finished {name}. Processed {count} users.")
        avg_metrics = {k: float(np.mean(v)) if v else 0.0 for k, v in all_metrics.items()}

        return {"per_user": per_user_metrics, "avg": avg_metrics}

    def eval(self):
        for k in self.test_k:
            self.top_k = k
            # Create dir inside eval_dir for this k
            k_eval_dir = os.path.join(self.base_eval_dir, f"top_{k}")
            os.makedirs(k_eval_dir, exist_ok=True)
            self.eval_dir = k_eval_dir
            print(f"\n--- Evaluating for Top-{k} Recommendations ---")
            self._eval_gnn_recs()
            self._eval_baselines()
