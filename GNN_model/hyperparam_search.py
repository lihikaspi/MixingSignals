import torch
import optuna
import json
from torch_geometric.data import HeteroData
from GNN_model.train_GNN import GNNTrainer
from GNN_model.GNN_class import LightGCN
from GNN_model.eval_GNN import GNNEvaluator
from config import config


def objective(trial):
    # --- UPDATED SEARCH RANGES BASED ON config.py ---
    try:
        # Best lr: 0.01. Range (1e-4, 5e-2) is good.
        lr = trial.suggest_float("lr", 0.05, 0.2, log=True)

        # Best neg_samples_per_pos: 2. Let's search [2, 3, 4, 5, 6]
        neg_samples_per_pos = trial.suggest_int("neg_samples_per_pos", 2, 6)

        # Best listen_weight: 0.8. Let's search around it.
        listen_weight = trial.suggest_float("listen_weight", 0.6, 1.0)

        # Best neutral_neg_weight: 0.5. Let's search around it.
        neutral_neg_weight = trial.suggest_float("neutral_neg_weight", 0.3, 0.7)

        # Best num_layers: 3. Range [2, 5] is good.
        num_layers = trial.suggest_int("num_layers", 2, 5)

        # Best weight_decay: 1e-5. Range (1e-6, 1e-3) is good.
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        # Best dropout: 0.25. Range (0.0, 0.5) is good.
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

        # Best metadata_scale: 30.0. Let's search around it.
        # metadata_scale = trial.suggest_float("metadata_scale", 10.0, 50.0, log=True)

        # Best audio_scale: 0.5. Range (0.1, 2.0) is good.
        # audio_scale = trial.suggest_float("audio_scale", 0.1, 2.0)

        # Best margin: 0.3. Range (0.1, 0.5) is good.
        margin = trial.suggest_float("bpr_margin", 0.1, 0.5)

        # --- APPLY ALL SUGGESTED PARAMS ---
        config.gnn.lr = lr
        config.gnn.neg_samples_per_pos = neg_samples_per_pos
        config.gnn.listen_weight = listen_weight
        config.gnn.neutral_neg_weight = neutral_neg_weight
        config.gnn.num_layers = num_layers
        config.gnn.weight_decay = weight_decay
        config.gnn.dropout = dropout
        # config.gnn.metadata_scale = metadata_scale
        # config.gnn.audio_scale = audio_scale
        config.gnn.margin = margin  # Updated from bpr_margin

        config.gnn.num_epochs = 7
        config.gnn.max_patience = 2

        torch.cuda.empty_cache()

        train_graph = torch.load(config.paths.train_graph_file)

        model = LightGCN(train_graph, config)
        trainer = GNNTrainer(model, train_graph, config)

        # Optional: short training for hyperparameter search
        # This will use trainer.best_ndcg from the training loop
        trainer.train(trial=True)

        metric = trainer.best_ndcg

	# Check if the metrics dict exists and save it
        if hasattr(trainer, 'best_metrics'):
            # Ensure metrics are JSON-serializable (convert tensors/numpy to float)
            serializable_metrics = {k: (v.item() if hasattr(v, 'item') else v)
                                    for k, v in trainer.best_metrics.items()}
            trial.set_user_attr("best_metrics", serializable_metrics)

        return metric

    except torch.cuda.OutOfMemoryError as e:
        print(f"\n--- !!! Trial {trial.number} FAILED due to CUDA OOM !!! ---")
        print(f"    Parameters: {trial.params}")
        print(f"    Error: {e}")
        # Clean up memory
        torch.cuda.empty_cache()
        # Tell Optuna to prune this trial
        raise optuna.exceptions.TrialPruned()


def main():
    storage_url = f"sqlite:///{config.paths.eval_dir}/hp_search.db"
    # --- UPDATED STUDY NAME ---
    study_name = "gnn_hp_search_cpu_no_scales"

    study = optuna.create_study(
        storage=storage_url,
        study_name=study_name,
        direction="maximize",
        load_if_exists=True
    )

    # --- UPDATED TRIAL COUNT ---
    study.optimize(objective, n_trials=100)  # Increased trials for more params

    best_params = study.best_params

    # with open(config.paths.best_param, "w") as f:
    #     json.dump(best_params, f, indent=4)

    print("Best hyperparameters:", best_params)


if __name__ == "__main__":
    main()