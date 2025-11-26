import faiss
import numpy as np
import pandas as pd
from config import Config


class ANNIndex:
    """
    build ANN index for user and song embeddings.
    """
    def __init__(self, index_type: str, config: Config):
        """
        Args:
            index_type: "gnn" or "content" identifier for the index type.
            config: configuration object.
        """
        self.index_type = index_type
        self.ann_index_path = getattr(config.paths, f"{self.index_type}_index")
        self.ann_song_ids_path = getattr(config.paths, f"{self.index_type}_song_ids")

        if index_type == "gnn":
            self.user_embeddings_gnn = config.paths.user_embeddings_gnn
            self.song_embeddings_gnn = config.paths.song_embeddings_gnn
            self.cold_start_songs_file = config.paths.cold_start_songs_file

        elif index_type == "content":
            self.user_content_embeddings = config.paths.filtered_user_embed_file
            self.audio_embeddings = config.paths.filtered_audio_embed_file

        self.nlist = config.ann.nlist
        self.nprobe = config.ann.nprobe
        self.top_k = config.ann.top_k

        self.index = None
        self.song_ids = None
        self.user_embs = None
        self.user_ids = None
        self.song_embs = None


    def _load_content_embeddings(self):
        """Load content-based embeddings."""
        user_data = pd.read_parquet(self.user_content_embeddings)
        self.user_ids = user_data['user_id'].to_numpy()
        self.user_embs = np.vstack(user_data['avg_embed'].to_numpy()).astype('float32')

        audio_data = pd.read_parquet(self.audio_embeddings)
        self.song_ids = audio_data['item_id'].to_numpy()
        self.song_embs = np.vstack(audio_data['normalized_embed'].to_numpy()).astype('float32')

        print(f"{self.index_type} index: Loaded {len(self.user_embs)} users and {len(self.song_embs)} songs.")


    def _load_gnn_embeddings(self):
        """Load user and song embeddings, combining GNN and cold-start audio-based ones."""
        # ---- Load user embeddings ----
        user_data = np.load(self.user_embeddings_gnn)
        self.user_embs = user_data['embeddings']
        self.user_ids = user_data['original_ids']

        # ---- Load song embeddings ----
        song_data = np.load(self.song_embeddings_gnn)
        song_embs_gnn = song_data['embeddings']
        song_ids_gnn = song_data['original_ids']

        # ---- Load cold-start / audio embeddings ----
        audio_df = pd.read_parquet(self.cold_start_songs_file)
        song_ids_audio = audio_df['item_id'].to_numpy()
        song_embs_audio = np.vstack(audio_df['normalized_embed'].to_numpy()).astype('float32')

        # ---- Combine ----
        self.song_embs = np.vstack([song_embs_gnn, song_embs_audio])
        self.song_ids = np.concatenate([song_ids_gnn, song_ids_audio])

        print(f"{self.index_type} index: Loaded {len(self.user_embs)} users and {len(self.song_embs)} songs.")


    def load_embeddings(self):
        """Load embeddings based on the index type."""
        if self.index_type == "content":
            self._load_content_embeddings()
        elif self.index_type == "gnn":
            self._load_gnn_embeddings()
        else:
            raise ValueError(f"Invalid index type: {self.index_type}")


    def build_index(self):
        """Build and train a FAISS IVF Flat index."""
        if self.song_embs is None:
            raise ValueError("Song embeddings not loaded. Call load_embeddings() first.")

        faiss.normalize_L2(self.song_embs)
        d = self.song_embs.shape[1]

        quantizer = faiss.IndexFlatIP(d)
        self.index = faiss.IndexIVFFlat(quantizer, d, self.nlist, faiss.METRIC_INNER_PRODUCT)

        print(f"Started training FAISS IVF {self.index_type} index")
        self.index.train(self.song_embs)
        print("Adding embeddings to index...")
        self.index.add(self.song_embs)

        self.index.nprobe = self.nprobe
        print(f"    > Index built: ntotal={self.index.ntotal}, nlist={self.nlist}, nprobe={self.nprobe}")


    def save(self):
        """Save the FAISS index and song ID mapping."""
        if self.index is None:
            raise ValueError("No index to save. Build or load an index first.")

        faiss.write_index(self.index, self.ann_index_path)
        np.save(self.ann_song_ids_path, self.song_ids)
        print(f"Saved index to {self.ann_index_path} and song IDs to {self.ann_song_ids_path}")


    def load(self):
        """Load FAISS index and song IDs."""
        self.index = faiss.read_index(self.ann_index_path)
        self.song_ids = np.load(self.ann_song_ids_path)
        print(f"Loaded index from {self.ann_index_path} with {self.index.ntotal} vectors.")


    def recommend(self, k: int = None):
        """
        Get top-k song recommendations for each user embedding.

        Args:
            k: number of recommendations to return. Defaults to top_k from config.

        Returns:
            dict mapping user IDs to lists of recommended song IDs.
        """
        if self.index is None or self.song_ids is None:
            raise ValueError("Index not loaded or built.")
        if self.user_embs is None:
            raise ValueError("User embeddings not loaded.")

        k = k or self.top_k
        faiss.normalize_L2(self.user_embs)

        D, I = self.index.search(self.user_embs, k)
        rec_song_ids = self.song_ids[I]
        rec_scores = D

        results = {uid: recs.tolist() for uid, recs in zip(self.user_ids, rec_song_ids)}

        print(f"Generated top-{k} recommendations for {len(self.user_embs)} users.\n")
        return results


    def retrieve_recs(self, k: int = None):
        """
        End-to-end pipeline: load embeddings → build index → recommend.

        Args:
            k: number of recommendations to return. Defaults to top_k from config.

        Returns:
            dict mapping user IDs to lists of recommended song IDs.
        """
        k = k or self.top_k
        self.load_embeddings()
        self.build_index()
        self.save()
        results = self.recommend(k)
        return results
