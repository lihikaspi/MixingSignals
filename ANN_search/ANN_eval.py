import numpy as np
import pandas as pd
from typing import Dict, List
import json
import os
from config import Config


class RecEvaluator:
    """
    Evaluator class for ANN and baseline recommendations.
    """
    def __init__(self, recs, config: Config):
        """
        Args:
            recs: dict containing GNN-ANN and baseline recommendations. Keys are user IDs, values are lists of recommended song IDs.
            config: configuration object
        """
        self.gnn_recs = recs.get("gnn", {})
        self.content_recs = recs.get("content", {})
        self.popular_recs = recs.get("popular", {})
        self.random_recs = recs.get("random", {})
        self.cf_recs = recs.get("cf", {})

        self.top_k = config.ann.top_k
        self.relevance_scores = config.paths.test_scores_file
        self.eval_dir = config.paths.eval_dir
        self.mapping_path = config.paths.user_mapping

        self.ground_truth, self.all_test_users = self._load_ground_truth_fast()
        self.user_map = self._load_user_mapping()


    def _load_ground_truth_fast(self):
        """
        Loads ground truth relevance scores and pre-processes them into a lookup dict.

        Returns: lookup dict containing ground truth data for each user.
        """
        df = pd.read_parquet(self.relevance_scores)
        df = df.rename(columns={"user_id": "user_idx", "item_id": "item_idx"})

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

        # Build dictionary
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

        print(f"Loaded Ground Truth for {len(gt_lookup)} users")
        return gt_lookup, unique_users


    def _load_user_mapping(self) -> Dict[int, int]:
        """
        loads users mapping from the train graph encoded IDs to original IDs

        Returns:
            dict mapping encoded IDs to original IDs. Returns empty dict if mapping file is not found.
        """
        if not os.path.exists(self.mapping_path):
            print(f"WARNING: Mapping file not found at {self.mapping_path}. Output JSONs will use Encoded IDs.")
            return {}

        # Fast load
        df = pd.read_parquet(self.mapping_path)
        return dict(zip(df['user_id'], df['uid']))


    def _save_eval_results(self, results: dict, path: str):
        """
        saves evaluation results to a JSON file.

        Args:
            results: dict containing per-user and average metrics
            path: path to save the JSON file to
        """
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
        """
        evaluates popular baseline recommendations. saves results to a JSON file.
        """
        if self.popular_recs:
            self._save_eval_results(self._eval(self.popular_recs, "Popular"),
                                    self.eval_dir + "/popular_eval_results.json")


    def _random_baseline(self):
        """
        evaluates random baseline recommendations. saves results to a JSON file.
        """
        if self.random_recs:
            self._save_eval_results(self._eval(self.random_recs, "Random"),
                                    self.eval_dir + "/random_eval_results.json")


    def _cf_baseline(self):
        """
        evaluates collaborative filtering baseline recommendations. saves results to a JSON file.
        """
        if self.cf_recs:
            self._save_eval_results(self._eval(self.cf_recs, "CF"),
                                    self.eval_dir + "/cf_eval_results.json")


    def _content_baseline(self):
        """
        evaluates content-based baseline recommendations. saves results to a JSON file.
        """
        if self.content_recs:
            self._save_eval_results(self._eval(self.content_recs, "Content"),
                                    self.eval_dir + "/content_eval_results.json")


    def _eval_gnn_recs(self):
        """
        evaluates GNN-ANN recommendations. saves results to a JSON file.
        """
        if self.gnn_recs:
            self._save_eval_results(self._eval(self.gnn_recs, "GNN"), self.eval_dir + "/gnn_eval_results.json")


    def _eval_baselines(self):
        """
        evaluates all baselines. Results are saved to a JSON file.
        """
        self._popular_baseline()
        self._random_baseline()
        self._cf_baseline()
        self._content_baseline()


    # --- Metrics ---
    def _calc_ndcg(self, pred_rels, true_rels, k_val):
        """
        Calculates NDCG@k for a given set of predicted and ground truth relevance scores.

        Args:
            pred_rels: array of predicted relevance scores
            true_rels: array of ground truth relevance scores
            k_val: k value for NDCG calculation

        Returns:
            NDCG@k score
        """
        # DCG
        gains = np.maximum(pred_rels, 0)
        dcg = np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))
        if dcg == 0: return 0.0

        # IDCG (Ideal sorting of all positive test items)
        ideal = np.sort(np.maximum(true_rels, 0))[::-1][:k_val]
        idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))

        return dcg / idcg if idcg > 0 else 0.0


    def _eval(self, recs: Dict[int, List[int]], name: str) -> Dict[str, Dict]:
        """
        evaluates a set of recommendations using all metrics

        Args:
            recs: dict containing user IDs as keys and list of recommended song IDs as values
            name: name of the set of recommendations (e.g. "Popular", "GNN", etc.)

        Returns:
            dict containing per-user and average metrics for the given set of recommendations.
        """
        print(f"Evaluating {name}...")
        k = self.top_k
        per_user_metrics = {}

        # Pre-allocate lists
        keys = ["ndcg@k", "ndcg_raw@k", "ndcg_listen_plus@k", "ndcg_like@k",
                "hit_like@k", "hit_like_listen@k", "dislike_rate@k", "novelty@k"]
        all_metrics = {key: [] for key in keys}

        count = 0
        total = len(self.all_test_users)

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
        """
        evaluate GNN-ANN recommendations and all baselines. Results are saved to JSON files.
        """
        self._eval_gnn_recs()
        self._eval_baselines()