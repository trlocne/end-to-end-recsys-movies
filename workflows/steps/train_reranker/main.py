"""
step4: train_reranker  (gộp tune_reranker + train_reranker)
  - Nếu --skip-tune: bỏ qua Ray Tune, dùng default hyperparams
  - Phase 1 (optional): Ray Tune để tìm lr, batch_size, dropout
  - Phase 2: Ray Train full training với best config
  - Lưu reranker.pt lên S3
"""
import argparse
import json
import logging
import sys
import os
import shutil
import gc

current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists("/app/src"):
    project_root = "/app"
elif "steps" in current_dir:
    project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
else:
    project_root = os.getcwd()

if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import numpy as np
import torch
import ray
import ray.data
from ray.train import ScalingConfig, RunConfig as TrainRunConfig
from ray.train.torch import TorchTrainer
from ray import tune
from ray.tune import Tuner, TuneConfig, RunConfig as TuneRunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from torch.utils.data import DataLoader, random_split

from src.models.lightgcn import LightGCN
from src.models.reranker import DeepFMReRanker, DeepFMFeatureExtractor
from src.data.dataset import ReRankerDataset, BPRTrainDataset
from src.training.trainer_ray import reranker_train_loop_per_worker

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_reranker")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path-glue",      required=True, help="S3 run path containing train_df, val_df, test_df")
    parser.add_argument("--data-path-data-prep", required=True, help="S3 run path containing users_map, items_map")
    parser.add_argument("--feature-file", required=True, help="S3 path to features.pt")
    parser.add_argument("--feature-rerank-file", required=True, help="S3 path to feature_rerank.pt")
    parser.add_argument("--gnn-model",    required=True, help="S3 path to gnn.pt")
    parser.add_argument("--model-output", required=True, help="S3 path and filename for reranker.pt")
    parser.add_argument("--mlflow-uri",   default="https://ml-23b3d76cdedc41608f6f313dc69d6067.ecs.ap-southeast-1.on.aws/")
    parser.add_argument("--mlflow-exp",   default="recsys-moivelens")
    parser.add_argument("--name-registry", default=None)
    parser.add_argument("--num-epochs",   type=int, default=20)
    parser.add_argument("--embed-dim",    type=int, default=128)
    parser.add_argument("--num-layers",   type=int, default=3)
    parser.add_argument("--skip-tune",    action="store_true", help="Skip Ray Tune hyperparameter search")
    parser.add_argument("--tune-samples", type=int, default=3)
    parser.add_argument("--tune-epochs",  type=int, default=10)
    parser.add_argument("--max-concurrent-trials", type=int, default=1)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=1, help="Number of Ray workers for distributed training")
    parser.add_argument("--ray-data-parallelism", type=int, default=1)
    parser.add_argument("--use-reuse-actors",    action="store_true", default=True)
    parser.add_argument("--ray-results-path", default=None, help="Shared storage path for Ray Train/Tune results (e.g. s3://...)")

    # Sync identifiers shared between GNN and Reranker training within the
    # same pipeline run. Jenkins post-train pipeline reads these tags to ensure
    # retrieval and reranker champions come from the same training payload.
    parser.add_argument("--pipeline-run-id", default=os.environ.get("PIPELINE_RUN_ID"),
                        help="Shared pipeline / workflow run id (e.g. Argo workflow uid).")
    parser.add_argument("--dataset-version", default=os.environ.get("DATASET_VERSION"),
                        help="Dataset snapshot version used for this training run.")

    args = parser.parse_args()

    import uuid
    run_uid = str(uuid.uuid4())[:8]
    logger.info(f"Execution UID: {run_uid}")

    use_gpu = torch.cuda.is_available()
    run_storage_path = args.ray_results_path
    if not run_storage_path and args.model_output.startswith("s3://"):
        run_storage_path = f"{os.path.dirname(args.model_output)}/ray_results"
    train_run_config = TrainRunConfig(storage_path=run_storage_path) if run_storage_path else TrainRunConfig()

    if os.path.exists("/app/src"):
        _project_root = "/app"
    elif "steps" in os.path.dirname(os.path.abspath(__file__)):
        _project_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../")
        )
    else:
        _project_root = os.getcwd()

    if ray.is_initialized():
        ray.shutdown()
    ray.init(runtime_env={"env_vars": {"PYTHONPATH": _project_root}})

    use_gpu = torch.cuda.is_available()

    from src.utils.s3 import download_from_s3, upload_to_s3

    local_feats = "features.pt"
    if args.feature_file.startswith("s3://"):
        download_from_s3(args.feature_file, local_feats)
    else:
        local_feats = args.feature_file

    local_rerank_feats = "feature_rerank.pt"
    if args.feature_rerank_file.startswith("s3://"):
        download_from_s3(args.feature_rerank_file, local_rerank_feats)
    else:
        local_rerank_feats = args.feature_rerank_file

    local_gnn = "gnn.pt"
    if args.gnn_model.startswith("s3://"):
        download_from_s3(args.gnn_model, local_gnn)
    else:
        local_gnn = args.gnn_model

    feats = torch.load(local_feats, weights_only=False)
    rerank_feats = torch.load(local_rerank_feats, weights_only=False)
    logger.info("features.pt, feature_rerank.pt and gnn.pt downloaded.")

    train_pd = pd.read_parquet(f"{args.data_path_data_prep}/train_df.parquet")
    val_pd   = pd.read_parquet(f"{args.data_path_data_prep}/val_df.parquet")
    test_pd = pd.read_parquet(f"{args.data_path_data_prep}/test_df.parquet")
    num_users = len(pd.read_parquet(f"{args.data_path_data_prep}/users_map.parquet"))
    num_items = len(pd.read_parquet(f"{args.data_path_data_prep}/items_map.parquet"))
    logger.info("Data loaded | %d users | %d items", num_users, num_items)

    gnn_state = torch.load(local_gnn, map_location="cpu", weights_only=False)
    if not isinstance(gnn_state, dict):
        raise ValueError("Unexpected GNN checkpoint format: expected a state_dict dictionary.")

    # Infer architecture from checkpoint to avoid shape mismatches
    ckpt_embed_dim = args.embed_dim
    embedding_key = None
    for candidate in ("model.embedding.weight", "item_embedding.weight", "embedding.weight"):
        if candidate in gnn_state and hasattr(gnn_state[candidate], "ndim") and gnn_state[candidate].ndim == 2:
            embedding_key = candidate
            break
    if embedding_key:
        ckpt_embed_dim = int(gnn_state[embedding_key].shape[1])

    ckpt_num_layers = args.num_layers
    alpha_key = "model.alpha" if "model.alpha" in gnn_state else "alpha"
    if alpha_key in gnn_state and hasattr(gnn_state[alpha_key], "shape"):
        alpha_len = int(gnn_state[alpha_key].shape[0])
        # In LightGCN, alpha usually has length num_layers + 1
        if alpha_len >= 1:
            ckpt_num_layers = max(1, alpha_len - 1)

    if ckpt_embed_dim != args.embed_dim or ckpt_num_layers != args.num_layers:
        logger.warning(
            "Overriding GNN architecture from checkpoint: embed_dim %s->%s, num_layers %s->%s",
            args.embed_dim,
            ckpt_embed_dim,
            args.num_layers,
            ckpt_num_layers,
        )

    gnn_model = LightGCN(
        num_users,
        num_items,
        embedding_dim=ckpt_embed_dim,
        num_layers=ckpt_num_layers,
    )
    try:
        gnn_model.load_state_dict(gnn_state)
    except RuntimeError as e:
        ckpt_shape = tuple(gnn_state[embedding_key].shape) if embedding_key else None
        raise RuntimeError(
            "Failed to load gnn checkpoint into LightGCN. "
            f"users={num_users}, items={num_items}, embed_dim={ckpt_embed_dim}, num_layers={ckpt_num_layers}, "
            f"embedding_key={embedding_key}, embedding_shape={ckpt_shape}. Original error: {e}"
        ) from e
    gnn_model.eval()

    # Pre-extract embeddings from GNN for the extractor
    with torch.no_grad():
        edge_index = feats["edge_index"].to(gnn_model.device if hasattr(gnn_model, "device") else "cpu")
        user_emb, item_emb = gnn_model(edge_index)

    watch_history = rerank_feats["history"]
    feature_extractor = DeepFMFeatureExtractor(
        user_emb=user_emb,
        item_emb=item_emb,
        user_features=rerank_feats.get("user_features", {}),
        item_features=rerank_feats.get("item_features", {}),
        user_watch_history=watch_history,
        embed_dim=ckpt_embed_dim,
    )
    logger.info("Feature extractor ready | feature_dim=%d", feature_extractor.feature_dim)

    all_interactions = pd.concat([train_pd, val_pd, test_pd])
    global_pos_map = BPRTrainDataset._build_pos_map(all_interactions)
    
    def prepare_ray_dataset(df, name):
        logger.info(f"Preparing Ray Dataset for {name}...")
        ds = ReRankerDataset(df, feature_extractor, num_items, user_pos_items=global_pos_map)
        
        # Use a DataLoader to batch feature extraction
        loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=0) # keep 0 to avoid serialization issues with extractor
        
        all_feats = []
        all_labels = []
        
        # Move extractor to GPU if available for faster batch processing
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feature_extractor.user_emb = feature_extractor.user_emb.to(device)
        feature_extractor.item_emb = feature_extractor.item_emb.to(device)
        feature_extractor.device = device

        for batch in loader:
            # ReRankerDataset returns features as tensors
            all_feats.append(batch["features"].numpy().astype(np.float32))
            all_labels.append(batch["label"].numpy().astype(np.float32))
            
        feats_np = np.concatenate(all_feats).astype(np.float32)
        labels_np = np.concatenate(all_labels).astype(np.float32)
        
        # Create Ray dataset with proper batching - features as array, not list
        df = pd.DataFrame({
            "features": feats_np.tolist(),  # Keep as list to avoid array serialization issues
            "label": labels_np
        })
        
        return ray.data.from_pandas(df)

    ray_train_ds = prepare_ray_dataset(train_pd, "Train")
    ray_val_ds   = prepare_ray_dataset(val_pd, "Val")
    
    # Aggressive memory cleanup
    del train_pd, val_pd, test_pd, all_interactions
    gc.collect()

    best_lr = args.lr
    best_dropout = args.dropout
    best_batch_size = args.batch_size
    best_weight_decay = args.weight_decay
    best_mlp_dims = [256, 128]

    if not args.skip_tune:
        logger.info("=== Phase 1: Ray Tune ===")
        search_space = {
            "lr": tune.loguniform(1e-3, 5e-3),
            "dropout": tune.uniform(0.1, 0.5),
            "weight_decay": tune.loguniform(1e-4, 1e-3),
            "mlp_dims": tune.choice([[256, 128], [128, 64], [64, 32]])
        }

        def tune_trainable(config):
            full_config = {
                "model_type": "DeepFM Re-ranker",
                "phase": "Rerank",
                "input_dim": feature_extractor.feature_dim,
                "feature_dim": feature_extractor.feature_dim,
                "embed_dim": args.embed_dim,
                "mlp_dims": config["mlp_dims"],
                "dropout": config["dropout"],
                "weight_decay": config["weight_decay"],
                "use_batch_norm": True,
                "lr": config["lr"],
                "batch_size": args.batch_size,
                "num_epochs": args.tune_epochs,
                "eval_every": 1,
                "mlflow_uri": args.mlflow_uri,
                "mlflow_exp": args.mlflow_exp,
                "is_tuning": True,
                "run_uid": run_uid,
                "model_name": "ReRanker",
                "name_registry": args.name_registry,
                "model_output": args.model_output,
                "feature_file": args.feature_file,
                "feature_rerank_file": args.feature_rerank_file,
                "data_path_glue": args.data_path_glue,
                "data_path_data_prep": args.data_path_data_prep,
                "pipeline_run_id": args.pipeline_run_id,
                "dataset_version": args.dataset_version
            }

            trainer_tune = TorchTrainer(
                train_loop_per_worker=reranker_train_loop_per_worker,
                train_loop_config=full_config,
                scaling_config=ScalingConfig(num_workers=1, use_gpu=use_gpu),
                datasets={"train": ray_train_ds, "val": ray_val_ds},
                run_config=train_run_config,
            )
            result = trainer_tune.fit()
            if result.metrics:
                tune.report(result.metrics)

        scheduler = ASHAScheduler(max_t=20, grace_period=4, reduction_factor=2)
        search_alg = OptunaSearch()

        tuner = Tuner(
            trainable=tune.with_resources(tune_trainable, {"cpu": 1}),
            param_space=search_space,
            tune_config=TuneConfig(
                metric="loss", mode="min",
                scheduler=scheduler,
                search_alg=search_alg,
                num_samples=args.tune_samples,
                max_concurrent_trials=args.max_concurrent_trials,
                reuse_actors=args.use_reuse_actors
            ),
            run_config=TuneRunConfig(name="ReRanker_Tuning", verbose=1, storage_path=run_storage_path)
        )
        results = tuner.fit()
        best_result = results.get_best_result(metric="loss", mode="min")
        best_cfg = best_result.config
        best_lr = best_cfg.get("lr", best_lr)
        best_batch_size = args.batch_size
        best_weight_decay = best_cfg.get("weight_decay", best_weight_decay)
        best_mlp_dims = best_cfg.get("mlp_dims", best_mlp_dims)
        logger.info("Best Tune: lr=%.5f dropout=%.2f batch=%d weight_decay=%.6f mlp_dims=%s", 
                    best_lr, best_dropout, best_batch_size, best_weight_decay, str(best_mlp_dims))

    logger.info("=== Phase 2: Full Training ===")
    model = DeepFMReRanker(
        feature_extractor.feature_dim, 
        embed_dim=args.embed_dim,
        dropout=best_dropout, 
        mlp_dims=best_mlp_dims
    )
    
    full_trainer_config = {
        "model": model,
        "input_dim": feature_extractor.feature_dim,
        "embed_dim": args.embed_dim,
        "mlp_dims": best_mlp_dims,
        "dropout": best_dropout,
        "use_batch_norm": True,
        "lr": best_lr,
        "batch_size": best_batch_size,
        "num_epochs": args.num_epochs,
        "eval_every": 1,
        "weight_decay": best_weight_decay,
        "mlflow_uri": args.mlflow_uri,
        "mlflow_exp": args.mlflow_exp,
        "is_tuning": False,
        "run_uid": run_uid,
        "model_name": "ReRanker",
        "phase": "Rerank",
        "name_registry": args.name_registry,
        "model_output": args.model_output,
        "feature_file": args.feature_file,
        "feature_rerank_file": args.feature_rerank_file,
        "data_path_glue": args.data_path_glue,
        "data_path_data_prep": args.data_path_data_prep,
        "pipeline_run_id": args.pipeline_run_id,
        "dataset_version": args.dataset_version
    }

    trainer = TorchTrainer(
        train_loop_per_worker=reranker_train_loop_per_worker,
        train_loop_config=full_trainer_config,
        scaling_config=ScalingConfig(num_workers=args.num_workers, use_gpu=use_gpu),
        datasets={"train": ray_train_ds, "val": ray_val_ds},
        run_config=train_run_config,
    )
    result = trainer.fit()

    best_checkpoint = result.checkpoint
    if best_checkpoint:
        with best_checkpoint.as_directory() as ckpt_dir:
            ckpt_path = os.path.join(ckpt_dir, "checkpoint.pt")
            if os.path.exists(ckpt_path):
                ckpt_dict = torch.load(ckpt_path, map_location="cpu")
                model.load_state_dict(ckpt_dict["model_state_dict"])

    torch.save(model.state_dict(), "reranker.pt")
    
    # Comprehensive metadata
    from datetime import datetime
    best_params = vars(args).copy()
    best_params.update({
        "model_type": "DeepFM Re-ranker",
        "description": "Metadata for DeepFM Re-ranker training job",
        "trained_at": datetime.now().isoformat(),
        "best_lr": best_lr,
        "best_dropout": best_dropout,
        "best_batch_size": best_batch_size,
        "best_weight_decay": best_weight_decay,
        "best_mlp_dims": best_mlp_dims,
        "feature_dim": feature_extractor.feature_dim,
        "model_output": args.model_output,
        "data_path_glue": args.data_path_glue,
        "data_path_data_prep": args.data_path_data_prep,
        "feature_file": args.feature_file,
        "feature_rerank_file": args.feature_rerank_file,
        
    })
    
    # Add final training metrics and best trial info
    if hasattr(result, "metrics") and result.metrics:
        # Filter out large or non-serializable objects from metrics if any
        serializable_metrics = {k: v for k, v in result.metrics.items() 
                               if isinstance(v, (int, float, str, list, dict, bool, type(None)))}
        best_params["final_metrics"] = serializable_metrics

    with open("reranker_params.json", "w") as f:
        json.dump(best_params, f, indent=4, default=str)

    if args.model_output.startswith("s3://"):
        upload_to_s3("reranker.pt", args.model_output)
        s3_base = os.path.dirname(args.model_output)
        upload_to_s3("reranker_params.json", f"{s3_base}/reranker_params.json")
    else:
        shutil.move("reranker.pt", args.model_output)
        shutil.move("reranker_params.json", os.path.join(os.path.dirname(args.model_output), "reranker_params.json"))

    logger.info("train_reranker complete. Model at %s", args.model_output)

if __name__ == "__main__":
    main()
