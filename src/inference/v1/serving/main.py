import os
import sys
import logging
import uvicorn
import pandas as pd
import numpy as np
import yaml
import torch
import feast

current_dir = os.path.dirname(os.path.abspath(__file__))
if "pipeline" in current_dir:
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
else:
    project_root = os.path.abspath(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.inference import RecommendationService, create_app
from src.inference.v1.core.triton_client import TritonRerankerClient
from src.models.reranker import DeepFMFeatureExtractor
from src.utils.s3 import download_from_s3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("serving-prod")

def load_serving_config():

    serving_config_s3 = os.getenv("SERVING_CONFIG_S3")
    if serving_config_s3:
        try:
            local_config = "/tmp/serving_config.yaml"
            download_from_s3(serving_config_s3, local_config)
            with open(local_config, "r") as f:
                cfg = yaml.safe_load(f) or {}
            logger.info(f"serving_config loaded from S3: {serving_config_s3}")
            return cfg
        except Exception as e:
            logger.warning(f"Failed to download serving_config from S3: {e}; falling back to image config")

    config_path = os.path.join(project_root, "configs/serving_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        logger.info(f"serving_config loaded from image fallback: {config_path}")
        return cfg
    return {}

def load_artifacts():
    serving_config = load_serving_config()
    emb_cfg = serving_config.get("embeddings", {})
    embed_dim = int(
        serving_config.get("models", {}).get(
            "gnn_embed_dim", int(os.getenv("GNN_EMBED_DIM", "128"))
        )
    )
    embedding_source = os.getenv("EMBEDDING_SOURCE", "parquet").strip().lower()

    metadata_s3 = os.getenv("METADATA_S3")
    local_user = "/tmp/user_embeddings.parquet"
    local_item = "/tmp/item_embeddings.parquet"
    local_meta = "/tmp/movies_metadata.parquet"

    def get_file(path_s3, local_path):
        if not path_s3:
            return
        if path_s3.startswith("s3://"):
            from src.utils.s3 import download_from_s3, download_folder_from_s3

            if path_s3.endswith("/") or (
                not path_s3.endswith(".parquet") and not path_s3.endswith(".pt")
            ):
                download_folder_from_s3(path_s3, local_path)
            else:
                download_from_s3(path_s3, local_path)
        else:
            if os.path.exists(path_s3) and path_s3 != local_path:
                import shutil

                shutil.copy(path_s3, local_path)
            elif not os.path.exists(local_path):
                logger.warning(f"File not found: {path_s3}")

    get_file(metadata_s3, local_meta)
    metadata_df = None
    if os.path.exists(local_meta):
        metadata_df = pd.read_parquet(local_meta)
        logger.info(f"Metadata loaded. Rows: {len(metadata_df)}")

    feature_extractor = None

    if embedding_source == "feast_only":
        logger.info(
            "EMBEDDING_SOURCE=feast_only — skipping bulk user/item parquet; "
            "embeddings are read via Feast get_online_features (Redis)."
        )
        user_emb = np.empty((0, embed_dim), dtype=np.float32)
        item_emb = np.empty((0, embed_dim), dtype=np.float32)
    else:
        user_emb_s3 = emb_cfg.get("user_emb") or os.getenv("USER_EMB_S3")
        item_emb_s3 = emb_cfg.get("item_emb") or os.getenv("ITEM_EMB_S3")
        base_path = (emb_cfg.get("base_path") or "").strip().rstrip("/")
        if base_path:
            if not (user_emb_s3 and str(user_emb_s3).strip()):
                user_emb_s3 = f"{base_path}/user_embeddings.parquet"
            if not (item_emb_s3 and str(item_emb_s3).strip()):
                item_emb_s3 = f"{base_path}/movie_embeddings.parquet"

        get_file(user_emb_s3, local_user)
        get_file(item_emb_s3, local_item)

        for path, label, src in (
            (local_user, "user", user_emb_s3),
            (local_item, "item", item_emb_s3),
        ):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Missing {label} embeddings at {path} (source={src!r}). "
                    "Ensure AWS credentials are configured and SERVING_CONFIG_S3 / "
                    "USER_EMB_S3 / ITEM_EMB_S3 point to valid S3 paths."
                )

        user_df = pd.read_parquet(local_user).sort_values("user_id")
        item_df = pd.read_parquet(local_item).sort_values("movie_id")
        user_emb = np.stack(user_df["embedding"].values)
        item_emb = np.stack(item_df["embedding"].values)

        if metadata_df is not None and "movie_id" in metadata_df.columns and "item_idx" not in metadata_df.columns:
            id_to_idx = {mid: idx for idx, mid in enumerate(item_df["movie_id"])}
            metadata_df["item_idx"] = metadata_df["movie_id"].map(id_to_idx)

        feature_file_s3 = (
            serving_config.get("training_lineage", {}).get("feature_file")
            or os.getenv("FEATURE_FILE_S3")
        )
        if feature_file_s3:
            local_feat = "/tmp/features.pt"
            get_file(feature_file_s3, local_feat)
            if os.path.exists(local_feat):
                try:
                    feats = torch.load(local_feat, map_location="cpu", weights_only=False)
                    reranker_input_dim = int(
                        serving_config.get("models", {}).get("reranker_input_dim", 0)
                        or os.getenv("RERANKER_INPUT_DIM", 0)
                    )
                    expected_input_dim = 8 * embed_dim + 32
                    if reranker_input_dim and reranker_input_dim != expected_input_dim:
                        logger.warning(
                            f"embed_dim={embed_dim} → feature_dim={expected_input_dim} "
                            f"but reranker_input_dim={reranker_input_dim}. "
                            "GNN/reranker version mismatch — skipping feature extractor, "
                            "reranking via Feast fallback."
                        )
                    else:
                        feature_extractor = DeepFMFeatureExtractor(
                            user_emb=torch.from_numpy(user_emb),
                            item_emb=torch.from_numpy(item_emb),
                            user_features=feats.get("user_features", {}),
                            item_features=feats.get("item_features", {}),
                            user_watch_history=feats.get("history", {}),
                            embed_dim=embed_dim,
                        )
                        logger.info(
                            f"DeepFMFeatureExtractor initialized. input_dim={feature_extractor.feature_dim}"
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to init DeepFMFeatureExtractor: {e} — falling back to Feast path"
                    )
        else:
            logger.warning(
                "feature_file not set in serving_config.training_lineage or FEATURE_FILE_S3 — skipping feature extractor"
            )

    # All model inference (reranker forward pass) goes through Triton HTTP.
    triton_uri = os.getenv("TRITON_URI", "triton-server.serving.svc.cluster.local:8001")
    triton_client = None
    for uri in [triton_uri, "triton-server.serving.svc.cluster.local:8001"]:
        try:
            triton_client = TritonRerankerClient(uri)
            logger.info(f"Connected to Triton at {uri}")
            break
        except Exception as e:
            logger.warning(f"Cannot connect to Triton at {uri}: {e}")

    if not triton_client:
        logger.warning("No Triton server reachable — reranking will be unavailable until Triton is up.")

    feature_source = os.getenv("FEATURE_SOURCE", "triton")
    fs = None
    feast_enable = os.getenv("FEAST_ENABLE", "true").lower() in ("1", "true", "yes")
    if feast_enable:
        try:
            repo_path = os.path.join(project_root, "src", "feature_repo")
            if os.path.isdir(repo_path):
                fs = feast.FeatureStore(repo_path=repo_path)
                logger.info(
                    "Feast FeatureStore initialized from %s (online features: CDC views, recent interactions, …)",
                    repo_path,
                )
            else:
                logger.warning("Feast repo not found at %s — online CDC features unavailable", repo_path)
        except Exception as e:
            logger.error("Failed to initialize Feast FeatureStore: %s", e)
    else:
        logger.info("FEAST_ENABLE=false — skipping Feast (no online CDC / Redis features)")

    if embedding_source == "feast_only" and not fs:
        logger.error(
            "EMBEDDING_SOURCE=feast_only requires FEAST_ENABLE=true and a valid feature_repo."
        )

    if feature_source == "triton" and fs is not None:
        logger.info(
            "FEATURE_SOURCE=triton — reranker forward passes use Triton; Feast still used for get_online_features."
        )

    return (
        user_emb,
        item_emb,
        feature_extractor,
        fs,
        metadata_df,
        triton_client,
        embed_dim,
        embedding_source,
    )

def main():
    (
        user_emb,
        item_emb,
        feature_extractor,
        fs,
        metadata_df,
        triton_client,
        embed_dim,
        embedding_source,
    ) = load_artifacts()

    serving_config = load_serving_config()

    from elasticsearch import Elasticsearch
    from pymilvus import connections, Collection

    milvus_config = serving_config.get('milvus', {})
    milvus_uri = milvus_config.get('uri') or os.getenv("MILVUS_URI")
    milvus_token = milvus_config.get('token') or os.getenv("MILVUS_TOKEN")
    milvus_col_name = milvus_config.get('movie_collection') or os.getenv("MILVUS_COLLECTION", "movie_embeddings")
    
    es_uri = os.getenv("ES_URI", "http://elasticsearch.infra.svc.cluster.local:9200")
    es_api_key = os.getenv("ES_API_KEY")
    es_index = os.getenv("ES_INDEX", "recsys-movies")
    
    es_client = None
    if es_api_key:
        try:
            es_client = Elasticsearch(es_uri, api_key=es_api_key)
            logger.info(f"Elasticsearch client initialized. Index: {es_index}")
        except Exception as e:
            logger.error(f"Failed to init ES: {e}")

    milvus_collection = None
    if milvus_uri and milvus_token:
        try:
            connections.connect("default", uri=milvus_uri, token=milvus_token)
            milvus_collection = Collection(milvus_col_name)
            milvus_collection.load()
            logger.info(f"Milvus collection {milvus_col_name} loaded.")
        except Exception as e:
            logger.error(f"Failed to init Milvus: {e}")

    use_mmr = os.getenv("USE_MMR", "true").lower() == "true"

    service = RecommendationService(
        user_emb=user_emb,
        item_emb=item_emb,
        reranker=None,
        feature_extractor=feature_extractor,
        milvus_collection=milvus_collection,
        es_client=es_client,
        es_index_name=es_index,
        top_k_candidates=int(os.getenv("TOP_K_CANDIDATES", 200)),
        top_k_final=int(os.getenv("TOP_K_FINAL", 20)),
        feast_fs=fs,
        metadata_df=metadata_df,
        use_mmr=use_mmr,
        triton_client=triton_client,
        embedding_source=embedding_source,
        embed_dim=embed_dim,
    )
    
    app = create_app(service)
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting Serving Node on port {port} (Triton: {'connected' if triton_client else 'unavailable'})")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
