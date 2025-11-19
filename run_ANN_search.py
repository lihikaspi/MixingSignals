import os
import pandas as pd
import numpy as np
from config import config
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from ANN_search.ANN_index import ANNIndex
from ANN_search.ANN_eval import RecEvaluator


def check_prev_files():
    """
    check for the files created in the previous stage.
    if at least one file is missing raises FileNotFoundError
    """
    needed = [config.paths.user_embeddings_gnn, config.paths.song_embeddings_gnn,
              config.paths.cold_start_songs_file, config.paths.filtered_audio_embed_file,
              config.paths.filtered_user_embed_file, config.paths.filtered_song_ids,
              config.paths.filtered_user_ids, config.paths.popular_song_ids,
              config.paths.positive_interactions_file]
    fail = False
    for file in needed:
      if not os.path.exists(file):
        print("Couldn't find file: {}".format(file))
        fail = True
    if fail:
      raise FileNotFoundError("Needed files are missing, run previous stage to create the needed files!")
    else:
      print("All needed files are present! starting indexing ... ")


def recommend_popular():
    print("\nmaking popularity-based recommendations ")
    song_ids = np.load(config.paths.popular_song_ids)
    user_ids = np.load(config.paths.filtered_user_ids)
    top_k_ids = song_ids[config.ann.top_k:]

    results = {uid: recs.tolist() for uid, recs in zip(user_ids, top_k_ids)}

    return results


def recommend_random():
    print("\nmaking random recommendations ")
    song_ids = np.load(config.paths.filtered_song_ids)
    user_ids = np.load(config.paths.filtered_user_ids)
    num_users = len(user_ids)
    num_songs = len(song_ids)
    rec_song_ids = np.random.randint(num_songs, size=(num_users, config.ann.top_k))

    results = {uid: recs.tolist() for uid, recs in zip(user_ids, rec_song_ids)}

    return results


def recommend_cf(top_k=10, top_sim_items=50):
    print("\nmaking content-based recommendations ")
    """
    Memory-friendly item-based CF using sparse operations.
    Returns a dict: {user_id: [song_id1, song_id2, ...], ...}
    """
    # ---- Step 1: Load interactions ----
    interactions = pd.read_parquet(config.paths.positive_interactions_file)
    # columns: ['user_id', 'item_id']

    # ---- Step 2: Load user and song IDs ----
    user_ids = np.load(config.paths.filtered_user_ids)
    song_ids = np.load(config.paths.filtered_song_ids)

    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    idx_to_user = {i: u for i, u in enumerate(user_ids)}
    song_to_idx = {s: i for i, s in enumerate(song_ids)}
    idx_to_song = {i: s for i, s in enumerate(song_ids)}

    num_users = len(user_ids)
    num_songs = len(song_ids)

    # ---- Step 3: Build sparse user-item matrix ----
    rows, cols, data = [], [], []
    for row in interactions.itertuples(index=False):
        u, s = row.user_id, row.item_id
        if u in user_to_idx and s in song_to_idx:
            rows.append(user_to_idx[u])
            cols.append(song_to_idx[s])
            data.append(1.0)  # implicit feedback

    R = csr_matrix((data, (rows, cols)), shape=(num_users, num_songs), dtype=np.float32)

    # ---- Step 4: Normalize item vectors ----
    # L2 normalize columns (item vectors)
    R_item = R.T.tocsr()
    R_item_norm = normalize(R_item, axis=1)

    # ---- Step 5: Compute top neighbors for each item sparsely ----
    top_neighbors = {}
    for i in range(num_songs):
        vec_i = R_item_norm[i]
        # Only compute dot product with items that share users
        coo = vec_i.dot(R_item_norm.T)
        # convert to dict of top items
        if coo.nnz > 0:
            sims = coo.toarray().ravel()
            sims[i] = 0.0  # ignore self
            top_idx = np.argsort(-sims)[:top_sim_items]
            top_neighbors[i] = {j: sims[j] for j in top_idx if sims[j] > 0}

    # ---- Step 6: Generate recommendations ----
    results_dict = {}
    for u_idx in range(num_users):
        user_vector = R[u_idx].indices  # items user interacted with
        scores = {}
        for item in user_vector:
            neighbors = top_neighbors.get(item, {})
            for n_item, sim in neighbors.items():
                if n_item not in user_vector:
                    scores[n_item] = scores.get(n_item, 0) + sim
        top_items = sorted(scores, key=lambda x: -scores[x])[:top_k]
        results_dict[idx_to_user[u_idx]] = [idx_to_song[i] for i in top_items]

    return results_dict


def main():
    gnn_index = ANNIndex("gnn", config)
    gnn_recs = gnn_index.retrieve_recs()

    content_index = ANNIndex("content", config)
    content_recs = content_index.retrieve_recs()

    popular_recs = recommend_popular()
    random_recs = recommend_random()
    cf_recs = recommend_cf()

    recs = {
        "gnn": gnn_recs,
        "content": content_recs,
        "popular": popular_recs,
        "random": random_recs,
        "cf": cf_recs
    }

    evaluator = RecEvaluator(recs, config)
    evaluator.eval()


if __name__ == "__main__":
    check_prev_files()
    main()
