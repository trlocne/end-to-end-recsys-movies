import os
import tempfile
import copy
import torch
import torch.nn as nn
import ray
import ray.train
import mlflow
import mlflow.pytorch
import logging
import json
from typing import Dict, Any, Optional, Tuple, List
from ray.train import report, Checkpoint, get_context, get_dataset_shard, get_checkpoint
from ray.train.torch import prepare_model
from src.utils.metrics import evaluate_batch
import numpy as np
import torch.distributed as dist
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

logger = logging.getLogger(__name__)

_VAL_LOSS_METRIC_KEYS = ("val_loss", "final.val_loss", "tuning.val_loss", "loss")


def _val_loss_from_run_metrics(metrics: Optional[Dict[str, Any]]) -> Optional[float]:
    if not metrics:
        return None
    for key in _VAL_LOSS_METRIC_KEYS:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _extract_candidate_val_loss(metrics: Optional[Dict[str, Any]]) -> Optional[float]:
    return _val_loss_from_run_metrics(metrics)


def _champion_compare_mode() -> str:
    raw = os.environ.get("CHAMPION_COMPARE_MODE", "incumbents").strip().lower()
    return raw if raw in ("incumbents", "global") else "incumbents"


def _model_registry_alias_names() -> Tuple[str, str]:
    cand = (os.environ.get("MODEL_ALIAS") or "champion").strip() or "champion"
    prod = (os.environ.get("PRODUCTION_ALIAS") or "production").strip() or "production"
    return cand, prod


def _resolve_registry_name(config: Optional[Dict[str, Any]], artifact_path: str) -> str:
    cfg = config or {}
    explicit_name = str(cfg.get("name_registry", "")).strip()
    if explicit_name:
        return explicit_name

    phase = str(cfg.get("phase", "")).strip().lower()
    if phase == "rerank":
        return "Recsys-ReRank"
    if phase == "gnn":
        return "Recsys-GNN"

    if "rerank" in artifact_path.lower():
        return "Recsys-ReRank"
    if "rerank" in str(cfg.get("model_name", "")).lower():
        return "Recsys-ReRank"
    return "Recsys-GNN"


def _apply_champion_success(
    client: mlflow.tracking.MlflowClient,
    registered_name: str,
    target_version,
    candidate_alias: str,
    val_loss: float,
    versions_to_clear_tag: List[Any],
) -> None:
    client.set_model_version_tag(
        name=registered_name,
        version=target_version.version,
        key="champion",
        value="true",
    )
    try:
        client.set_registered_model_alias(
            name=registered_name,
            alias=candidate_alias,
            version=target_version.version,
        )
    except Exception:
        pass
    logger.info(
        "Champion set: version=%s val_loss=%.6f alias=%s",
        target_version.version,
        val_loss,
        candidate_alias,
    )
    for version in versions_to_clear_tag:
        try:
            client.delete_model_version_tag(
                name=registered_name,
                version=version.version,
                key="champion",
            )
        except Exception:
            continue


def _mark_champion_global(
    client: mlflow.tracking.MlflowClient,
    registered_name: str,
    run_id: str,
    val_loss: float,
    target_version,
    candidate_alias: str,
) -> None:
    is_champion = True
    worse_versions = []
    for version in client.search_model_versions(f"name='{registered_name}'"):
        if version.run_id == run_id:
            continue
        try:
            run_data = client.get_run(version.run_id).data
            other_val_loss = _val_loss_from_run_metrics(run_data.metrics)
            if other_val_loss is not None and other_val_loss < val_loss:
                is_champion = False
                break
            if other_val_loss is not None:
                worse_versions.append(version)
        except Exception:
            continue

    if is_champion:
        _apply_champion_success(
            client,
            registered_name,
            target_version,
            candidate_alias,
            val_loss,
            worse_versions,
        )


def _mark_champion_incumbents(
    client: mlflow.tracking.MlflowClient,
    registered_name: str,
    run_id: str,
    val_loss: float,
    target_version,
    candidate_alias: str,
    production_alias: str,
) -> None:
    prev_candidate_mv = None
    try:
        prev_candidate_mv = client.get_model_version_by_alias(
            registered_name, candidate_alias
        )
    except Exception:
        pass

    incumbent_run_ids: List[str] = []
    for alias in (candidate_alias, production_alias):
        try:
            mv = client.get_model_version_by_alias(registered_name, alias)
            if mv.run_id and mv.run_id != run_id:
                incumbent_run_ids.append(mv.run_id)
        except Exception:
            continue

    seen = set()
    deduped = []
    for rid in incumbent_run_ids:
        if rid not in seen:
            seen.add(rid)
            deduped.append(rid)

    is_champion = True
    for rid in deduped:
        try:
            run_data = client.get_run(rid).data
            other_val_loss = _val_loss_from_run_metrics(run_data.metrics)
            if other_val_loss is not None and other_val_loss < val_loss:
                is_champion = False
                break
        except Exception:
            continue

    versions_to_clear = []
    if (
        prev_candidate_mv is not None
        and int(prev_candidate_mv.version) != int(target_version.version)
    ):
        versions_to_clear.append(prev_candidate_mv)

    if is_champion:
        _apply_champion_success(
            client,
            registered_name,
            target_version,
            candidate_alias,
            val_loss,
            versions_to_clear,
        )


def _mark_champion_if_best(registered_name: str, run_id: str, val_loss: Optional[float]):
    """Update candidate model alias if this run beats incumbents (default) or all versions (global).

    CHAMPION_COMPARE_MODE: ``incumbents`` (default) compares val_loss only to versions
    pointed to by MODEL_ALIAS and PRODUCTION_ALIAS; ``global`` preserves legacy behavior
    (search all registered versions). Candidate alias written is MODEL_ALIAS (default
    ``champion``).
    """
    if val_loss is None:
        return

    client = mlflow.tracking.MlflowClient()
    latest_versions = client.get_latest_versions(registered_name, stages=["None"])
    target_version = next((v for v in latest_versions if v.run_id == run_id), None)
    if target_version is None:
        return

    candidate_alias, production_alias = _model_registry_alias_names()
    mode = _champion_compare_mode()
    if mode == "global":
        _mark_champion_global(
            client,
            registered_name,
            run_id,
            val_loss,
            target_version,
            candidate_alias,
        )
    else:
        _mark_champion_incumbents(
            client,
            registered_name,
            run_id,
            val_loss,
            target_version,
            candidate_alias,
            production_alias,
        )


def _safe_log_model_to_mlflow(
    model,
    artifact_path: str,
    config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    signature: Optional[ModelSignature] = None,
):
    """Log MLflow model first, fallback to stable artifact bundle."""
    base_model = model.module if hasattr(model, "module") else model
    if mlflow.active_run() is None:
        logger.warning(
            "Skip model logging for '%s': no active MLflow run.",
            artifact_path,
        )
        return

    registered_model_name = _resolve_registry_name(config, artifact_path)

    try:
        model_for_logging = copy.deepcopy(base_model).cpu().eval()
        metadata = {
            "model_type": (config or {}).get("model_type", base_model.__class__.__name__),
            "framework": "pytorch",
            "task": (config or {}).get("task", artifact_path),
            "model_class": base_model.__class__.__name__,
            "model_module": base_model.__class__.__module__,
        }
        hyperparameters = {
            k: v
            for k, v in (config or {}).items()
            if isinstance(v, (int, float, str, bool)) and k not in ("model", "feats_ref")
        }
        if hyperparameters:
            metadata["hyperparameters"] = hyperparameters

        log_kwargs = {
            "pytorch_model": model_for_logging,
            "artifact_path": artifact_path,
            "pip_requirements": ["torch", "mlflow", "cloudpickle"],
            "metadata": metadata,
        }
        if signature is not None:
            log_kwargs["signature"] = signature
        log_kwargs["registered_model_name"] = registered_model_name

        mlflow.pytorch.log_model(**log_kwargs)

        active_run = mlflow.active_run()
        if active_run is not None:
            _mark_champion_if_best(
                registered_model_name,
                active_run.info.run_id,
                _extract_candidate_val_loss(metrics),
            )

        mlflow.set_tag(f"{artifact_path}_logged_model", "success")
        logger.info("Logged MLflow model for '%s'.", artifact_path)
        return
    except Exception as log_model_err:
        mlflow.set_tag(f"{artifact_path}_log_model_error", str(log_model_err)[:500])
        logger.warning(
            "mlflow.pytorch.log_model failed for '%s', fallback to state_dict: %s",
            artifact_path,
            log_model_err,
        )

    try:
        with tempfile.TemporaryDirectory() as tmp:
            state_dict_path = os.path.join(tmp, "state_dict.pt")
            metadata_path = os.path.join(tmp, "config.json")

            # Persist CPU tensors so artifact is portable across devices.
            cpu_state_dict = {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in base_model.state_dict().items()
            }
            torch.save(cpu_state_dict, state_dict_path)
            metadata = {
                "model_class": base_model.__class__.__name__,
                "model_module": base_model.__class__.__module__,
                "state_dict_file": "state_dict.pt",
                "torch_version": torch.__version__,
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)

            mlflow.log_artifact(state_dict_path, artifact_path=artifact_path)
            mlflow.log_artifact(metadata_path, artifact_path=artifact_path)
            mlflow.set_tag(f"{artifact_path}_logged_model", "fallback_state_dict")
            logger.info("Logged stable artifact bundle for '%s'.", artifact_path)

            active_fb = mlflow.active_run()
            if active_fb is not None:
                model_uri = f"runs:/{active_fb.info.run_id}/{artifact_path}"
                try:
                    mlflow.register_model(model_uri=model_uri, name=registered_model_name)
                except Exception as reg_err:
                    logger.warning(
                        "mlflow.register_model after fallback failed for '%s': %s",
                        artifact_path,
                        reg_err,
                    )
                else:
                    _mark_champion_if_best(
                        registered_model_name,
                        active_fb.info.run_id,
                        _extract_candidate_val_loss(metrics),
                    )
    except Exception as e:
        mlflow.set_tag(f"{artifact_path}_logged_model", "failed")
        mlflow.set_tag(f"{artifact_path}_fallback_error", str(e)[:500])
        logger.warning("Model artifact logging failed for '%s': %s", artifact_path, e)


def _log_config_artifact(config: Dict[str, Any]):
    """Log a JSON copy of simple config values."""
    if mlflow.active_run() is None:
        logger.warning("Skip config artifact logging: no active MLflow run.")
        return
    serializable_config = {
        k: v for k, v in config.items() if isinstance(v, (int, float, str, bool, list, dict))
    }
    with tempfile.TemporaryDirectory() as tmp:
        config_path = os.path.join(tmp, "config.json")
        with open(config_path, "w") as f:
            json.dump(serializable_config, f, indent=4)
        mlflow.log_artifact(config_path)


def generate_recommendations(model, data_loader, edge_index, edge_weight, top_k=20):
    """Vectorized recommendation generation for evaluation."""
    model.eval()
    recommended = {}
    relevant_truth = {}

    with torch.no_grad():
        base_model = model.module if hasattr(model, "module") else model
        u_emb, i_emb = base_model(edge_index, edge_weight)
        
        for batch in data_loader:
            data = batch.get("item", batch) if isinstance(batch, dict) else batch
            users = data.get("user")
            relevants = data.get("relevant")
                
            if users is None: continue
            
            if not torch.is_tensor(users):
                u_ids = torch.as_tensor(users, device=u_emb.device).long()
            else:
                u_ids = users.to(u_emb.device).long()
            
            scores = u_emb[u_ids] @ i_emb.T
            _, top_indices = torch.topk(scores, k=min(top_k, i_emb.size(0)), dim=1)
            
            top_indices_np = top_indices.cpu().numpy()
            u_ids_np = u_ids.cpu().numpy()
            
            for i, uid in enumerate(u_ids_np):
                uid = int(uid)
                if uid not in recommended:
                    recommended[uid] = top_indices_np[i].tolist()
                    if relevants is not None:
                        rel = relevants[i]
                        if hasattr(rel, "tolist"):
                            relevant_truth[uid] = rel.tolist()
                        elif isinstance(rel, (list, tuple, np.ndarray)):
                            relevant_truth[uid] = list(rel)
                        else:
                            relevant_truth[uid] = [int(rel)]
                            
    return recommended, relevant_truth

def _setup_mlflow(config: Dict[str, Any], rank: int):
    if rank != 0:
        return
    
    mlflow_uri = config.get("mlflow_uri", "http://localhost:5000")
    mlflow_exp = config.get("mlflow_exp", "recsys-moivelens")
    is_tuning = config.get("is_tuning", False)
    phase = "Tuning" if is_tuning else "Training"
    
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(mlflow_exp)
        
        run_name = f"{config.get('model_name')}_{phase}" if config.get("model_name") else f"GNN_{phase}"
            
        active_run = mlflow.active_run()
        mlflow.start_run(run_name=run_name, nested=(active_run is not None))
        
        mlflow.set_tag("run_uid", config.get("run_uid"))
        mlflow.set_tag("phase", phase.lower())

        for sync_key in ("pipeline_run_id", "dataset_version"):
            sync_val = config.get(sync_key)
            if sync_val:
                mlflow.set_tag(sync_key, str(sync_val))

        # Log explicit artifact path tags so serving workflow can resolve inputs
        # without reconstructing paths from bucket conventions or workflow.uid.
        # Keys: feature_file (full S3 path to features.pt),
        #       data_prep_path (full S3 dir containing train_df, users_map, etc.)
        for tag_key, config_key in [
            ("feature_file",   "feature_file"),
            ("data_prep_path", "data_path_data_prep"),
        ]:
            val = config.get(config_key)
            if val:
                mlflow.set_tag(tag_key, str(val))

        # Feature schema version for downstream compatibility gating
        schema_ver = config.get("feature_schema_version")
        if schema_ver:
            mlflow.set_tag("feature_schema_version", str(schema_ver))

        try:
            trial_id = get_context().get_trial_id()
            if trial_id:
                mlflow.set_tag("trial_id", trial_id)
        except:
            pass
        
   
        _HYPERPARAM_KEYS = {
            "lr", "embed_dim", "input_dim", "num_layers", "reg_weight",
            "batch_size", "num_epochs", "eval_every", "k",
            "dropout", "weight_decay", "tune_samples", "tune_epochs",
        }
        params = {
            k: v for k, v in config.items()
            if k in _HYPERPARAM_KEYS and isinstance(v, (int, float, str, bool))
        }
        if params:
            mlflow.log_params(params)
        logger.info(f"MLflow initialized: {phase} run '{run_name}'")
    except Exception as e:
        logger.warning(f"Failed to initialize MLflow: {e}")

def _finalize_training(model, metrics, rank, config: Dict[str, Any]):
    if rank != 0:
        return
        
    try:
        base_model = model.module if hasattr(model, "module") else model
        
        is_tuning = config.get("is_tuning", False)
        if not is_tuning:
            _safe_log_model_to_mlflow(base_model, "gnn_model", config=config, metrics=metrics)
        
        _log_config_artifact(config)
            
        mlflow.end_run()
        logger.info("MLflow run finalized.")
    except Exception as e:
        logger.warning(f"Failed to finalize MLflow: {e}")

def compute_bpr_loss(user_emb, item_emb, users, pos_items, neg_items, eps=1e-10):
    pos_scores = (user_emb[users] * item_emb[pos_items]).sum(1)
    neg_scores = (user_emb[users] * item_emb[neg_items]).sum(1)
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + eps).mean()

def compute_reg_loss(base_model, users, pos_items, neg_items, reg_weight):
    user_norm = base_model.user_embedding(users).norm(2, 1).pow(2)
    pos_norm = base_model.item_embedding(pos_items).norm(2, 1).pow(2)
    neg_norm = base_model.item_embedding(neg_items).norm(2, 1).pow(2)
    return reg_weight * (user_norm + pos_norm + neg_norm).mean()

def training_step(model, batch, edge_index, edge_weight, optimizer, reg_weight, device):
    optimizer.zero_grad()
    users = batch["user"].to(device)
    pos_items = batch["pos_item"].to(device)
    neg_items = batch["neg_item"].to(device)

    user_emb, item_emb = model(edge_index, edge_weight)
    loss = compute_bpr_loss(user_emb, item_emb, users, pos_items, neg_items)
    
    base_model = model.module if hasattr(model, 'module') else model
    reg = compute_reg_loss(base_model, users, pos_items, neg_items, reg_weight)
    
    total_loss = loss + reg
    total_loss.backward()
    optimizer.step()
    return total_loss.item(), loss.item(), reg.item()

def gnn_train_loop_per_worker(config: Dict[str, Any]):
    config.setdefault("phase", "GNN")
    lr = config.get("lr", 1e-3)
    reg_weight = config.get("reg_weight", 1e-4)
    batch_size = config.get("batch_size", 1024)
    num_epochs = config.get("num_epochs", 100)
    eval_every = config.get("eval_every", 5)
    weight_decay = config.get("weight_decay", 1e-5)
    k = config.get("k", 20)
    
    context = get_context()
    rank = context.get_world_rank()
    world_size = context.get_world_size()
    device = ray.train.torch.get_device()
    
    _setup_mlflow(config, rank)
    
    model = config.get('model')
    if model is None:
        from src.models.lightgcn import LightGCN
        model = LightGCN(
            num_users=config['num_users'],
            num_items=config['num_items'],
            embedding_dim=config['embed_dim'],
            num_layers=config['num_layers']
        )
    model = prepare_model(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

    feats_data = ray.get(config['feats_ref'])
    edge_index = feats_data['edge_index'].to(device)
    edge_weight = feats_data.get('edge_weight')
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    start_epoch = 0
    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as ckpt_dir:
            ckpt_path = os.path.join(ckpt_dir, "checkpoint.pt")
            if os.path.exists(ckpt_path):
                if config.get("is_tuning", False):
                    logger.info("Checkpoint found but skipped for tuning run.")
                else:
                    ckpt_dict = torch.load(ckpt_path, map_location="cpu")
                    base_model = getattr(model, "module", model)
                    try:
                        base_model.load_state_dict(ckpt_dict["model_state_dict"])
                        optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"])
                        start_epoch = ckpt_dict.get("epoch", -1) + 1
                        logger.info(f"Resumed training from checkpoint at epoch {start_epoch}.")
                    except RuntimeError as e:
                        logger.warning(
                            "Checkpoint load skipped due to model shape mismatch. "
                            "Starting fresh training. Details: %s",
                            e,
                        )

    train_ds = get_dataset_shard("train")
    if train_ds:
        train_loader = train_ds.iter_torch_batches(batch_size=batch_size)
    else:
        train_loader = []

    val_ds_shard = get_dataset_shard("val")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        total_bpr = 0.0
        total_reg = 0.0
        num_batches = 0

        for batch in train_loader:
            loss_val, bpr_val, reg_val = training_step(model, batch, edge_index, edge_weight, optimizer, reg_weight, device)
            total_loss += loss_val
            total_bpr += bpr_val
            total_reg += reg_val
            num_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_bpr = total_bpr / num_batches if num_batches > 0 else 0
        avg_reg = total_reg / num_batches if num_batches > 0 else 0
        
        metrics = {
            "loss": avg_loss,
            "val_loss": avg_loss,
            "bpr_loss": avg_bpr,
            "reg_loss": avg_reg
        }

        if epoch % eval_every == 0 and val_ds_shard is not None:
            model.eval()
            val_loader = val_ds_shard.iter_torch_batches(batch_size=batch_size)
            recommended, val_data_shard = generate_recommendations(model, val_loader, edge_index, edge_weight, top_k=k)
            
            val_p_at_k = [k] if isinstance(k, int) else k
            if recommended:
                val_metrics_raw = evaluate_batch(recommended, val_data_shard, ks=val_p_at_k)
                for m_name, m_val in val_metrics_raw.items():
                    key = f"val_{m_name.replace('@', '_at_')}"
                    metrics[key] = m_val
                
            if world_size > 1:
                for key in list(metrics.keys()):
                    if key.startswith("val_"):
                        val_tensor = torch.tensor(metrics[key], device=device)
                        dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
                        metrics[key] = val_tensor.item() / world_size

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_model = model.module if hasattr(model, "module") else model
            save_dict = {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics
            }
            torch.save(save_dict, os.path.join(tmpdir, "checkpoint.pt"))
            report(metrics=metrics, checkpoint=Checkpoint.from_directory(tmpdir))

            if rank == 0:
                mlflow.log_metrics(metrics, step=epoch)

    _finalize_training(model, metrics, rank, config)

def reranker_train_loop_per_worker(config: Dict[str, Any]):
    config.setdefault("phase", "Rerank")
    lr = config.get("lr", 1e-4)
    batch_size = config.get("batch_size", 1024)
    num_epochs = config.get("num_epochs", 100)
    eval_every = config.get("eval_every", 5)
    weight_decay = config.get("weight_decay", 1e-5)
    k = config.get("k", 20)

    context = get_context()
    rank = context.get_world_rank()
    world_size = context.get_world_size()
    device = ray.train.torch.get_device()

    _setup_mlflow(config, rank)

    model = config.get("model")
    if model is None:
        from src.models.reranker import DeepFMReRanker
        model = DeepFMReRanker(
            input_dim=config['input_dim'],
            embed_dim=config['embed_dim'],
            mlp_dims=config['mlp_dims'],
            dropout=config['dropout'],
            use_batch_norm=config['use_batch_norm']
        )
    
    model = prepare_model(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    criterion = nn.BCEWithLogitsLoss()

    checkpoint = get_checkpoint()
    start_epoch = 0
    if checkpoint:
        with checkpoint.as_directory() as ckpt_dir:
            ckpt_path = os.path.join(ckpt_dir, "checkpoint.pt")
            if os.path.exists(ckpt_path):
                if config.get("is_tuning", False):
                    logger.info("Checkpoint found but skipped for tuning run (reranker).")
                else:
                    ckpt_dict = torch.load(ckpt_path, map_location="cpu")
                    base_model = getattr(model, "module", model)
                    try:
                        base_model.load_state_dict(ckpt_dict["model_state_dict"])
                        optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"])
                        start_epoch = ckpt_dict.get("epoch", -1) + 1
                        logger.info(f"Resumed reranker training from checkpoint at epoch {start_epoch}.")
                    except RuntimeError as e:
                        logger.warning(
                            "Reranker checkpoint load skipped due to model shape mismatch. "
                            "Starting fresh training. Details: %s",
                            e,
                        )

    train_ds = get_dataset_shard("train")
    val_ds = get_dataset_shard("val")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        train_loader = train_ds.iter_torch_batches(batch_size=batch_size, drop_last=True)
        for batch in train_loader:
            optimizer.zero_grad()
            
            x = batch["features"].to(device)
            y = batch["label"].to(device).view(-1, 1).float()
            
            logits = model(x)
            loss = criterion(logits, y)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        scheduler.step()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        metrics = {"loss": avg_loss}
        
        if epoch % eval_every == 0 and val_ds is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                val_loader = val_ds.iter_torch_batches(batch_size=batch_size, drop_last=True)
                for batch in val_loader:
                    x = batch["features"].to(device)
                    y = batch["label"].to(device).view(-1, 1).float()
                    val_loss += criterion(model(x), y).item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
            metrics["val_loss"] = avg_val_loss
            
            if world_size > 1:
                val_loss_tensor = torch.tensor(avg_val_loss, device=device)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                metrics["val_loss"] = val_loss_tensor.item() / world_size
        
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_model = model.module if hasattr(model, "module") else model
            save_dict = {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics
            }
            torch.save(save_dict, os.path.join(tmpdir, "checkpoint.pt"))
            report(metrics=metrics, checkpoint=Checkpoint.from_directory(tmpdir))

            if rank == 0:
                mlflow.log_metrics(metrics, step=epoch)

    if rank == 0:
        is_tuning = config.get("is_tuning", False)
        if not is_tuning:
            feature_dim = config.get("input_dim")
            signature = None
            if isinstance(feature_dim, int) and feature_dim > 0:
                signature = ModelSignature(
                    inputs=Schema(
                        [TensorSpec(name="features", type=np.dtype(np.float32), shape=(-1, feature_dim))]
                    ),
                    outputs=Schema([TensorSpec(type=np.dtype(np.float32), shape=(-1, 1))]),
                )
            _safe_log_model_to_mlflow(
                getattr(model, "module", model),
                "reranker_model",
                config=config,
                metrics=metrics,
                signature=signature,
            )
        _log_config_artifact(config)
        mlflow.end_run()

