import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from typing import Dict, List
import json
from config import Config


class RecEvaluator:
    """
    Evaluator class for ANN recommendations.
    Optimized with single-load Ground Truth and Debugging.
    """
    def __init__(self, recs, config: Config):
        self.gnn_recs = recs["gnn"]
        self.content_recs = recs["content"]
        self.popular_recs = recs["popular"]
        self.random_recs = recs["random"]
        self.cf_recs = recs["cf"]

        self.top_k = config.ann.top_k
        self.relevance_scores = config.paths.test_scores_file
        self.eval_dir = config.paths.eval_dir

        print("Loading ground truth data for evaluation...")
        self.ground_truth = self._load_ground_truth()


    def _load_ground_truth(self):
        """Loads parquet once and converts it to a fast lookup dict."""
        df = pd.read_parquet(self.relevance_scores)
        df = df.rename(columns={"user_id": "user_idx", "item_id": "item_idx"})

        # Force types to ensure matching
        df['user_idx'] = df['user_idx'].astype(np.int64)
        df['item_idx'] = df['item_idx'].astype(np.int64)

        gt_lookup = {}
        for uid, group in df.groupby("user_idx"):
            gt_lookup[uid] = {
                "items": group["item_idx"].values,
                "adj": group["adjusted_score"].values.astype(float),
                "base": group["base_relevance"].values.astype(float),
                "listen_plus": group["listen_plus_relevance"].values.astype(float),
                "like": group["like_relevance"].values.astype(float)
            }
        print(f"Loaded ground truth for {len(gt_lookup)} users.")
        return gt_lookup


    def _popular_baseline(self):
        print("Evaluating Popular baseline...")
        popular_eval = self._eval(self.popular_recs, "Popular")
        save_path = self.eval_dir + "/popular_eval_results.json"
        self._save_eval_results(popular_eval, save_path)


    def _random_baseline(self):
        print("Evaluating Random baseline...")
        random_eval = self._eval(self.random_recs, "Random")
        save_path = self.eval_dir + "/random_eval_results.json"
        self._save_eval_results(random_eval, save_path)


    def _cf_baseline(self):
        print("Evaluating CF baseline...")
        cf_eval = self._eval(self.cf_recs, "CF")
        save_path = self.eval_dir + "/cf_eval_results.json"
        self._save_eval_results(cf_eval, save_path)


    def _content_baseline(self):
        print("Evaluating Content baseline...")
        content_eval = self._eval(self.content_recs, "Content")
        save_path = self.eval_dir + "/content_eval_results.json"
        self._save_eval_results(content_eval, save_path)


    def _eval_gnn_recs(self):
        print("Evaluating GNN recommendations...")
        gnn_eval = self._eval(self.gnn_recs, "GNN")
        save_path = self.eval_dir + "/gnn_eval_results.json"
        self._save_eval_results(gnn_eval, save_path)


    def _eval_baselines(self):
        self._popular_baseline()
        self._random_baseline()
        self._cf_baseline()
        self._content_baseline()


    def _save_eval_results(self, results: dict, path: str):
        def convert(o):
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return o

        with open(path, "w") as f:
            json.dump(results, f, default=convert, indent=4)


    def _calc_ndcg(self, topk_idx, relevance, gt_adj, k):
        # Linear Gain NDCG
        top_rel = relevance[topk_idx]
        top_rel = np.maximum(top_rel, 0)

        dcg = np.sum(top_rel / np.log2(np.arange(2, len(top_rel) + 2)))

        ideal = np.sort(np.maximum(gt_adj, 0))[::-1][:k]
        idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))

        return dcg / (idcg if idcg > 0 else 1.0)


    def _eval(self, recs: Dict[int, List[int]], name: str) -> Dict[str, Dict]:
        k = self.top_k

        per_user_metrics = {}
        all_metrics = {
            "ndcg@k": [], "ndcg_raw@k": [],
            "ndcg_listen_plus@k": [], "ndcg_like@k": [], # ADDED
            "hit_like@k": [],
            "hit_like_listen@k": [], "auc": [], "dislike_fpr@k": [], "novelty@k": []
        }

        debug_printed = False

        # Iterate only over users present in the recommendations
        for uid, rec_items in recs.items():
            uid = int(uid)  # Ensure int
            if uid not in self.ground_truth:
                continue

            gt_data = self.ground_truth[uid]

            topk_idx = rec_items[:k]
            topk_set = set(topk_idx)

            gt_items = gt_data["items"]
            gt_adj = gt_data["adj"]
            gt_base = gt_data["base"]
            gt_listen_plus = gt_data["listen_plus"]
            gt_like = gt_data["like"]

            # --- DEBUGGING FIRST MATCH ---
            if not debug_printed:
                print(f"--- DEBUG {name} ---")
                print(f"User ID: {uid}")
                print(f"Top-5 Recs: {topk_idx[:5]}")
                print(f"GT Items (first 5): {gt_items[:5]}")
                overlap = len(set(topk_idx) & set(gt_items))
                print(f"Overlap Count: {overlap}")
                debug_printed = True
            # -----------------------------

            max_item_idx = max(np.max(gt_items), max(topk_idx)) + 1

            relevance_adj = np.zeros(max_item_idx, dtype=float)
            relevance_adj[gt_items] = gt_adj

            relevance_base = np.zeros(max_item_idx, dtype=float)
            relevance_base[gt_items] = gt_base

            relevance_listen_plus = np.zeros(max_item_idx, dtype=float)  # ADDED
            relevance_listen_plus[gt_items] = gt_listen_plus  # ADDED

            relevance_like = np.zeros(max_item_idx, dtype=float)  # ADDED
            relevance_like[gt_items] = gt_like  # ADDED

            user_metrics = {}

            user_metrics["ndcg@k"] = self._calc_ndcg(topk_idx, relevance_adj, gt_adj, k)
            user_metrics["ndcg_raw@k"] = self._calc_ndcg(topk_idx, relevance_base, gt_base, k)
            user_metrics["ndcg_listen_plus@k"] = self._calc_ndcg(topk_idx, relevance_listen_plus, gt_listen_plus, k)
            user_metrics["ndcg_like@k"] = self._calc_ndcg(topk_idx, relevance_like, gt_like, k)

            like_items = gt_items[gt_adj > 1.0]
            user_metrics["hit_like@k"] = float(len(set(like_items) & topk_set) > 0)

            pos_items = gt_items[gt_adj > 0.5]
            user_metrics["hit_like_listen@k"] = float(len(set(pos_items) & topk_set) > 0)

            user_metrics["auc"] = 0.0
            all_metrics["auc"].append(0.0)

            dislike_items = gt_items[gt_adj < 0]
            dislike_fpr = float(len(set(dislike_items) & topk_set) > 0) if len(dislike_items) else 0.0
            user_metrics["dislike_fpr@k"] = dislike_fpr
            all_metrics["dislike_fpr@k"].append(dislike_fpr)

            unseen_in_topk = sum(1 for i in topk_idx if i not in gt_items)
            novelty = unseen_in_topk / k
            user_metrics["novelty@k"] = novelty
            all_metrics["novelty@k"].append(novelty)

            per_user_metrics[uid] = user_metrics
            for m, val in user_metrics.items():
                all_metrics[m].append(val)

        avg_metrics = {m: float(np.mean(v)) if len(v) else 0.0 for m, v in all_metrics.items()}

        return {
            "per_user": per_user_metrics,
            "avg": avg_metrics
        }

    def eval(self):
        """Main evaluation wrapper."""
        self._eval_gnn_recs()
        self._eval_baselines()