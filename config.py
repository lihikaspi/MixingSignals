from dataclasses import dataclass, field
import torch
import torch.distributed as dist
from typing import List, Tuple, Dict
import os
import numpy as np

# -------------------
# DATASET CONFIG
# -------------------
@dataclass
class DatasetConfig:
    dataset_size: str = "50m"          # Options: "50m", "500m", "5b"
    dataset_type: str = "flat"         # Options: "flat", "sequential"
    download_full: bool = False        # True = download everything, False = only essentials


# -------------------
# PIPELINE CONFIG
# -------------------
@dataclass
class PipelineConfig:
    stage_files: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("download", "download_data.py"),
        ("gnn_prep", "run_GNN_prep.py"),
        ("gnn_train", "run_GNN_train.py"),
        ("ann_search", "run_ANN_search.py"),
    ])


# -------------------
# PATHS CONFIG
# -------------------
@dataclass
class PathsConfig:
    dataset_size: str = "50m"
    data_dir: str = field(init=False)
    csv_dir: str = field(init=False)
    processed_dir: str = "processed_data"
    gnn_models_dir: str = "models/GNN"
    ann_models_dir: str = "models/ANN"
    eval_dir: str = "eval_results"

    data_cols_file: str = field(init=False)
    data_stats_file: str = field(init=False)

    raw_listens_file: str = field(init=False)
    raw_likes_file: str = field(init=False)
    raw_dislikes_file: str = field(init=False)
    raw_unlikes_file: str = field(init=False)
    raw_undislikes_file: str = field(init=False)
    raw_multi_event_file: str = field(init=False)

    audio_embeddings_file: str = field(init=False)

    album_mapping_file: str = field(init=False)
    artist_mapping_file: str = field(init=False)

    interactions_file: str = field(init=False)
    train_set_file: str = field(init=False)
    val_set_file: str = field(init=False)
    test_set_file: str = field(init=False)

    cold_start_songs_file: str = field(init=False)
    filtered_audio_embed_file: str = field(init=False)
    filtered_user_embed_file: str = field(init=False)
    filtered_song_ids: str = field(init=False)
    filtered_user_ids: str = field(init=False)
    popular_song_ids: str = field(init=False)
    positive_interactions_file: str = field(init=False)
    negative_train_in_graph_file: str = field(init=False)
    negative_train_cold_start_file: str = field(init=False)

    train_edges_file: str = field(init=False)
    train_graph_file: str = field(init=False)
    test_graph_file: str = field(init=False)

    val_scores_file: str = field(init=False)
    test_scores_file: str = field(init=False)

    raw_data_files: List[str] = field(init=False)
    split_paths: Dict[str, str] = field(init=False)

    trained_gnn: str = field(init=False)
    best_param: str = field(init=False)
    user_embeddings_gnn: str = field(init=False)
    song_embeddings_gnn: str = field(init=False)

    gnn_index: str = field(init=False)
    gnn_song_ids: str = field(init=False)
    content_index: str = field(init=False)
    content_song_ids: str = field(init=False)
    cf_index: str = field(init=False)
    cf_song_ids: str = field(init=False)

    test_eval: str = field(init=False)
    val_eval: str = field(init=False)

    def __post_init__(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.gnn_models_dir, exist_ok=True)
        os.makedirs(self.ann_models_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)

        self.data_dir = f"project_data/YambdaData{self.dataset_size}"
        self.csv_dir = f"project_data/YambdaDataCSV"

        self.data_cols_file = f"{self.csv_dir}/yambda_columns.csv"
        self.data_stats_file = f"{self.csv_dir}/YambdaStats_{self.dataset_size}.csv"

        self.raw_listens_file = f"{self.data_dir}/listens.parquet"
        self.raw_likes_file = f"{self.data_dir}/likes.parquet"
        self.raw_dislikes_file = f"{self.data_dir}/dislikes.parquet"
        self.raw_unlikes_file = f"{self.data_dir}/unlikes.parquet"
        self.raw_undislikes_file = f"{self.data_dir}/undislikes.parquet"
        self.raw_multi_event_file = f"{self.data_dir}/multi_event.parquet"

        self.audio_embeddings_file = f"{self.data_dir}/embeddings.parquet"

        self.album_mapping_file = f"{self.data_dir}/album_mapping.parquet"
        self.artist_mapping_file = f"{self.data_dir}/artist_mapping.parquet"

        self.interactions_file = f"{self.processed_dir}/interactions.parquet"
        self.train_set_file = f"{self.processed_dir}/train.parquet"
        self.val_set_file = f"{self.processed_dir}/val.parquet"
        self.test_set_file = f"{self.processed_dir}/test.parquet"

        self.cold_start_songs_file = f"{self.processed_dir}/cold_start_songs.parquet"
        self.filtered_audio_embed_file = f"{self.processed_dir}/filtered_audio_embed.parquet"
        self.filtered_user_embed_file = f"{self.processed_dir}/filtered_user_embed.parquet"
        self.filtered_song_ids = f"{self.processed_dir}/filtered_song_ids.npy"
        self.filtered_user_ids = f"{self.processed_dir}/filtered_user_ids.npy"
        self.popular_song_ids = f"{self.processed_dir}/popular_song_ids.npy"
        self.positive_interactions_file = f"{self.processed_dir}/positive_interactions.parquet"
        self.negative_train_in_graph_file = f"{self.processed_dir}/neg_train_in_graph.parquet"
        self.negative_train_cold_start_file = f"{self.processed_dir}/neg_train_cold_start.parquet"

        self.train_edges_file = f"{self.processed_dir}/train_edges.parquet"
        self.train_graph_file = f"{self.processed_dir}/train_graph.pt"
        self.test_graph_file = f"{self.processed_dir}/test_graph.pt"

        self.val_scores_file = f"{self.processed_dir}/val_scores.parquet"
        self.test_scores_file = f"{self.processed_dir}/test_scores.parquet"

        self.raw_data_files = [
            self.raw_listens_file,
            self.raw_likes_file,
            self.raw_dislikes_file,
            self.raw_unlikes_file,
            self.raw_undislikes_file
        ]

        self.split_paths = {
            "train": self.train_set_file,
            "val": self.val_set_file,
            "test": self.test_set_file,
        }

        self.trained_gnn = f"{self.gnn_models_dir}/best_model.pth"
        self.best_param = f"{self.gnn_models_dir}/best_params.txt"
        self.user_embeddings_gnn = f"{self.gnn_models_dir}/user_embeddings.npz"
        self.song_embeddings_gnn = f"{self.gnn_models_dir}/song_embeddings.npz"

        self.gnn_index = f"{self.ann_models_dir}/gnn_index.faiss"
        self.gnn_song_ids = f"{self.ann_models_dir}/gnn_song_ids.npy"
        self.content_index = f"{self.ann_models_dir}/content_index.faiss"
        self.content_song_ids = f"{self.ann_models_dir}/content_song_ids.npy"
        self.cf_index = f"{self.ann_models_dir}/cf_index.faiss"
        self.cf_song_ids = f"{self.ann_models_dir}/cf_song_ids.npy"

        self.test_eval = f"{self.eval_dir}/gnn_test_eval.txt"
        self.val_eval = f"{self.eval_dir}/gnn_val_eval.txt"


# -------------------
# PREPROCESSING CONFIG
# -------------------
@dataclass
class PreprocessingConfig:
    low_interaction_threshold: int = 680
    high_interaction_threshold: int = 6800
    edge_type_mapping: Dict[str, int] = field(default_factory=lambda: {
        "listen": 1,
        "like": 2,
        "dislike": 3,
        "unlike": 4,
        "undislike": 5
    })
    weights: Dict[str, float] = field(default_factory=lambda: {
        "listen": 0.7,
        "like": 1.0,
        "dislike": 0,
        "unlike": 0,
        "undislike": 0.3
    })
    split_ratios: Dict[str, float] = field(default_factory=lambda: {
        "train": 0.8,
        "val": 0.1,
        "test": 0.1
    })
    novelty = {
        "unseen_boost": 0.20,  # extra multiplier for songs never seen
        "train_penalty": 0.05,  # how hard we penalise over-played songs
        "recency_beta": 0.0001,  # exponential decay of recency penalty
        "max_familiarity": 20,  # denominator for familiarity normalisation
    }


# -------------------
# GNN CONFIG
# -------------------
@dataclass
class GNNConfig:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    edge_attr_indices: Dict[str, int] = field(default_factory=lambda: {
        "type": 0,
        "count": 1,
        "ratio": 2,
        "weight": 3  # The heuristic weight
    })

    embed_dim: int = 128
    num_layers: int = 5
    lambda_align: float = 0.0
    audio_lr_scale: float = 0.1

    edge_mlp_hidden_dim: int = 8
    edge_mlp_input_dim: int = 4

    neutral_neg_weight: float = 0.31
    min_listen_weight: float = 0.3

    lr: float = 0.03
    lr_decay: float = 0.98
    lr_cycle: int = 5
    momentum: float = 0.9
    max_grad_norm: float = 1.0
    init_std: float = 0.005

    num_epochs: int = 50
    batch_size: int = 256
    weight_decay: float = 4e-6
    num_workers: int = 8
    eval_every: int = 5
    neg_samples_per_pos: int = 3
    accum_steps: int = 1
    audio_scale: float = 0.5
    metadata_scale: float = 30.0
    dropout: float = 0.005
    margin: float = 0.3
    max_patience: int = 5

    eval_batch_size: int = 512

    k_hit: int = 10
    top_popular_k: int = 100


# -------------------
# ANN CONFIG
# -------------------
@dataclass
class ANNConfig:
    top_k: int = 10
    top_sim_items: int = 50
    nprobe: int = 32
    nlist: int = 4096

    seed: int = 42
    np.random.seed(seed)


# -------------------
# COMBINED CONFIG
# -------------------
@dataclass
class Config:
    dataset: DatasetConfig = DatasetConfig()
    pipeline: PipelineConfig = PipelineConfig()
    paths: PathsConfig = PathsConfig(dataset_size=DatasetConfig().dataset_size)
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    gnn: GNNConfig = GNNConfig()
    ann: ANNConfig = ANNConfig()


# single global config object
config = Config()
