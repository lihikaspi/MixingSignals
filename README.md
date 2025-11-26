# Music Recommendation System

A **graph-based music recommendation system** leveraging GNNs and ANN search for scalable retrieval.

Data Analysis and Presentation - Final project     
Created By: Lihi Kaspi, Harel Oved & Niv Maman

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [The Yandex Yambda Dataset](#the-yandex-yambda-dataset)
4. [Setup and Requirements](#setup-and-requirements)
5. [File Structure](#file-structure)
6. [Configuration](#configuration)
7. [Project Pipeline](#project-pipeline)
   - [Stage 1: Download Dataset](#stage-1-download-dataset)
   - [Stage 2: Prepare Data for GNN](#stage-2-prepare-data-for-gnn)
   - [Stage 3: Train and Evaluate GNN](#stage-3-train-and-evaluate-gnn)
   - [Stage 4: ANN Search and Evaluation](#stage-4-ann-search-and-evaluation)

---

## Overview

This project implements a large-scale music recommendation system using the Yandex Yambda dataset.
It combines Graph Neural Networks (GNNs) with Approximate Nearest Neighbor (ANN) search to deliver efficient and accurate 
recommendations across millions of user–song interactions.

The core workflow:
1. **Download and preprocess** the Yandex Yambda dataset.  
2. **Construct a bipartite user–song interaction graph.**  
3. **Train a LightGCN GNN** to learn user and song embeddings.  
4. **Build and query an ANN index** using Faiss for fast recommendation retrieval.  
5. **Evaluate recommendations** using standard metrics and baselines.


---

## Quick Start

```bash
# Clone repository and navigate to project directory
git clone https://github.com/lihikaspi/MixingSignals
cd MixingSignals

# Install dependencies (requires Python 3.10+)
pip install -r requirements.txt

# Verify GPU setup
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run full pipeline (requires GPU)
python run_all.py

# Or run individual stages
python run_all.py --stage 1  # Download dataset
python run_all.py --stage 2  # Prepare data for GNN
python run_all.py --stage 3  # Train and evaluate GNN
python run_all.py --stage 4  # ANN search and evaluation
```

---

## The Yandex Yambda Dataset

The **Yandex Yambda dataset** contains billions of user-track interactions, both **implicit** 
(listens) and **explicit** (likes/dislikes), with **precomputed audio embeddings** for millions of songs.

More information: [Yandex Yambda Dataset on Hugging Face](https://huggingface.co/datasets/yandex/yambda)

### Components

- **Multi-event file:** Unified table with multiple interaction types (`listen`, `like`, `unlike`, `dislike`, `undislike`)  
- **Single-event files:** Interactions of a single type  
- **Audio embeddings:** Precomputed embeddings per song  
- **Mapping files:** Song-to-album and song-to-artist relationships

### Dataset Scales

| Scale | Users     | Songs     | Interactions   |
|-------|-----------|-----------|----------------|
| 50M   | 10,000    | 934,057   | ~50M           |
| 500M  | 100,000   | 3,004,578 | ~500M          |
| 5B    | 1,000,000 | 9,390,623 | ~5B            |

---

## Setup and Requirements

### Prerequisites

- **Python:** 3.10 or higher  
- **GPU:** CUDA-compatible GPU required for GNN training and ANN indexing (CPU-only runs are not supported)  
- **Disk Space:** Enough to store the chosen dataset scale (50M, 500M, 5B interactions)

### Installing Dependencies

```bash
pip install -r requirements.txt
```

#### Installing PyTorch and PyTorch Geometric

> PyTorch and PyG versions must match, including CUDA versions

The `requirements.txt` includes PyTorch 2.0.1 with CUDA 11.8 support by default.     
For other CUDA versions, modify the PyTorch versions in `requirements.txt`.

For additional PyG installation guidance, follow the official guide:
[pytorch-geometric installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

### Verify Installation

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

import torch_geometric
print(f"PyG version: {torch_geometric.__version__}")
```

---

## File Structure

> **Note:** only code files uploaded to this repository.

```
├── config.py                   # Configuration and hyperparameters
├── run_all.py                  # Main pipeline runner
├── download_data.py            # Dataset download
├── run_GNN_prep.py             # GNN data preparation
├── run_GNN_train.py            # GNN training and evaluation
├── run_ANN_search.py           # ANN search and evaluation
├── requirements.txt            # Python dependencies

├── GNN_prep/
│   ├── event_processor.py      # Process and filter interactions
│   ├── edge_assembler.py       # Aggregate edges and features
│   └── build_graph.py          # Build graph for GNN input

├── GNN_model/
│   ├── GNN_class.py            # LightGCN model definition
│   ├── train_GNN.py            # Training 
│   ├── eval_GNN.py             # GNN evaluation 
│   ├── explore_graph.py        # Training graph statistics 
│   ├── song_diagnostics.py     # Song embeddigs scales
│   └── hyperparam_search.py    # Optuna study

├── ANN_search/
│   ├── ANN_index.py            # FAISS-based indexing
│   └── ANN_eval.py             # Metrics for retrieval performance

├── models/
│   ├── GNN/
│   │   ├── best_model.pth      # Trained GNN checkpoint
│   │   ├── user_embeddings.npz # Final user embeddings
│   │   └── song_embeddings.npz # Final song embeddings
│   └── ANN/
│       ├── index.faiss         # ANN index file
│       └── song_ids.npy        # Song ID mapping

└── project_data/
    ├── download_yambda.py      # Hugging Face dataset wrapper
    ├── yambda_inspect.py       # Dataset column inspection script
    ├── yambda_stats.py         # Dataset statistics generation script
    ├── YambdaDataCSV/          # Extra dataset info directory
    └── YambdaData50m/          # Raw dataset directory
```


---

## Configuration

All key parameters are in `config.py`:

```python
# Dataset
config.dataset.dataset_size         # "50m", "500m", "5b"
config.dataset.dataset_type         # "flat" or "sequential"
config.dataset.download_full        # True/False

# Preprocessing
config.preprocessing.low_interaction_threshold     
config.preprocessing.split_ratios   # dict with train, val and test

# GNN
config.gnn.device                   # "cuda" if available
config.gnn.embed_dim              
config.gnn.layers_num

# ANN
config.ann.top_k                    # number of recommendations to retrieve
```

---

## Project Pipeline

The pipeline consists of **four main stages**:

### Stage 1: Download Dataset

- Downloads Yandex Yambda via Hugging Face and saves parquet files locally.  
- Default download: multi-event file, audio embeddings, and mapping files.  
- Optional: download single-event files by updating `config.dataset.download_full`.

```bash
# via the top-level runner (recommended)
python run_all.py --stage download

# directly
python download_data.py
```

Additional scripts: dataset columns and statistics

```bash
# Saves a CSV with column names and the first row of each file:
python project_data/yambda_inspect.py

# Saves a CSV with basic statistics for each interaction file
python project_data/yambda_stats.py
```

### Stage 2: Prepare Data for GNN

- **Data preprocessing** (`EventProcessor` in `event_processor.py`):
  - Filter users with fewer than threshold interactions  
  - Filter songs without audio embeddings  
  - Encode user and song IDs  
  - Split into train, validation, and test sets

- **Edge assembly** (`EdgeAssembler` in `edge_assembler.py`):
  - Aggregate interactions into `(user, song, event)` records  
  - Add initial event-type weights  
  - Encode artist and album IDs  
  - Save ready-to-build edges

- **Graph construction** (`GraphBuilder` in `build_graph.py`):
  - Build bipartite graph with users and songs as nodes  
  - Edge features: event type, interaction count, played ratio  
  - Node features: user embedding, song embedding, artist & album embeddings  
  - Saves graph as `graph.pt`

```bash
# via top-level runner
python run_all.py --stage 2

# directly
python run_GNN_prep.py
```

### Stage 3: Train and Evaluate GNN

- **Model:** LightGCN with weighted edges (users learn embeddings, songs use frozen audio + learnable artist/album embeddings)  
- **Edge weights:** Predicted via a small MLP (EdgeWeightMLP)  
- **Loss:** Weighted BPR loss, differentiating hard/neutral negatives and event types  
- **Optimizer:** SGD with gradient accumulation and separate learning rates for parameter groups  
- **Evaluation:** `GNNEvaluator` computes NDCG@K, Hit@K, AUC, and Dislike-FPR@K

```bash
# via the top-level runner (recommended)
python run_all.py --stage gnn_train

# directly
python run_GNN_train.py
```

### Stage 4: ANN Search and Evaluation

- **ANNIndex** (`ANN_index.py`):  
  - Builds Faiss index with trained embeddings + cold-start audio embeddings  
  - Searches for top-K songs for each user

- **Evaluation** (`ANN_eval.py`):  
  - Compare GNN recommendations against baselines:  
    - Popular songs  
    - Random recommendations  
    - Collaborative Filtering
    - Content-Based recommendations

```bash
# via top-level runner
python run_all.py --stage 4

# directly
python run_ANN_search.py
```

