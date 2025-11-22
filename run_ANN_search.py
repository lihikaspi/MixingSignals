import os
import pandas as pd
import numpy as np
from config import config
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
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


def recommend_popular(max_k):
    print(f"making popularity-based recommendations for top-{max_k}")
    song_ids = np.load(config.paths.popular_song_ids)
    user_ids = np.load(config.paths.filtered_user_ids)
    top_k_recs = song_ids[:max_k].tolist()

    results = {uid: top_k_recs for uid in user_ids}

    return results


def recommend_random(max_k):
    print(f"making random recommendations for top-{max_k}")
    song_ids = np.load(config.paths.filtered_song_ids)
    user_ids = np.load(config.paths.filtered_user_ids)
    num_users = len(user_ids)
    num_songs = len(song_ids)
    rec_song_ids = np.random.randint(num_songs, size=(num_users, max_k))

    results = {uid: recs.tolist() for uid, recs in zip(user_ids, rec_song_ids)}

    return results


def prepare_cf_index(n_components=64) -> ANNIndex:
    """
    Calculates variables for Collaborative Filtering using Matrix Factorization (SVD).
    Injects them into an ANNIndex instance and returns it.
    """
    print("Preparing SVD-based CF variables...")

    # 1. Load Data
    interactions = pd.read_parquet(config.paths.positive_interactions_file)
    user_ids = np.load(config.paths.filtered_user_ids)
    song_ids = np.load(config.paths.filtered_song_ids)

    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    song_to_idx = {s: i for i, s in enumerate(song_ids)}

    # 2. Build Sparse Matrix
    # Filter only valid users/items
    valid_mask = interactions['user_id'].isin(user_to_idx) & interactions['item_id'].isin(song_to_idx)
    valid_interactions = interactions[valid_mask]

    rows = valid_interactions['user_id'].map(user_to_idx).values
    cols = valid_interactions['item_id'].map(song_to_idx).values
    data = np.ones(len(rows), dtype=np.float32)

    R = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(song_ids)), dtype=np.float32)

    # 3. SVD (Matrix Factorization)
    print(f"Computing TruncatedSVD with {n_components} components...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)

    # User embeddings: Projection of users onto latent features
    user_emb_svd = svd.fit_transform(R)

    # Item embeddings: The latent features themselves (transpose V)
    item_emb_svd = svd.components_.T

    # 4. Init ANNIndex and Inject Data
    print("Initializing ANNIndex for CF...")
    cf_ann = ANNIndex("cf", config)

    # Manually set the variables usually loaded from disk
    cf_ann.user_embs = user_emb_svd.astype(np.float32)
    cf_ann.song_embs = item_emb_svd.astype(np.float32)
    cf_ann.user_ids = user_ids
    cf_ann.song_ids = song_ids

    return cf_ann


def main():
    max_k = max(config.ann.test_k)
    gnn_index = ANNIndex("gnn", config)
    gnn_recs = gnn_index.retrieve_recs(max_k)

    content_index = ANNIndex("content", config)
    content_recs = content_index.retrieve_recs(max_k)

    cf_index_obj = prepare_cf_index()
    cf_index_obj.build_index()
    cf_index_obj.save()
    cf_recs, _ = cf_index_obj.recommend(k=max_k)

    popular_recs = recommend_popular(max_k)
    random_recs = recommend_random(max_k)

    recs = {
        "gnn": gnn_recs,
        "content": content_recs,
        "popular": popular_recs,
        "random": random_recs,
        "cf": cf_recs
    }

    print("\n----------- EVALUATION  -----------")
    evaluator = RecEvaluator(recs, config)
    evaluator.eval()


if __name__ == "__main__":
    check_prev_files()
    main()