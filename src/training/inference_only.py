import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
import logging
import mlflow

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.lightgcn import LightGCN
from src.utils.s3 import download_from_s3, upload_to_s3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gnn_inference")

GNN_EXPORT_EMBED_DIM = 128


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="S3 path to pre-trained gnn.pt")
    parser.add_argument("--feature-file", required=True, help="S3 path to latest features.pt")
    parser.add_argument("--data-path-data-prep", required=True, help="S3 path to latest users_map/items_map")
    parser.add_argument("--output-path", required=True, help="S3 prefix to save new embeddings")
    
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=GNN_EXPORT_EMBED_DIM,
        help=f"Production GNN uses fixed dim {GNN_EXPORT_EMBED_DIM}.",
    )
    parser.add_argument("--num-layers", type=int, default=3)

    args = parser.parse_args()
    if args.embed_dim != GNN_EXPORT_EMBED_DIM:
        logger.warning(
            "Ignoring --embed-dim=%s; inference export uses gnn_embed_dim=%s",
            args.embed_dim,
            GNN_EXPORT_EMBED_DIM,
        )

    local_model = "gnn.pt"
    local_feats = "features.pt"

    # Check if model-path is an MLflow URI
    if args.model_path.startswith("models:/"):
        logger.info(f"Downloading model from MLflow: {args.model_path}")
        model = mlflow.pytorch.load_model(args.model_path)
        # Do not unwrap model.model: that is torch_geometric LightGCN, whose forward()
        # takes (edge_index, edge_label_index), not (edge_index, edge_weight). Our
        # src.models.lightgcn.LightGCN.forward uses get_embedding(..., edge_weight=...).
        model_loaded = True
    else:
        download_from_s3(args.model_path, local_model)
        model_loaded = False

    download_from_s3(args.feature_file, local_feats)

    num_users = len(pd.read_parquet(f"{args.data_path_data_prep}/users_map.parquet"))
    num_items = len(pd.read_parquet(f"{args.data_path_data_prep}/items_map.parquet"))
    feats = torch.load(local_feats, weights_only=False)

    if model_loaded:
        # Model already loaded from MLflow, just set to eval mode
        model.eval()
        # Verify embed_dim matches (src.models.lightgcn.LightGCN wraps PyG model in .model)
        actual_embed_dim = None
        if hasattr(model, 'embedding_dim'):
            actual_embed_dim = model.embedding_dim
        elif hasattr(model, 'embed_dim'):
            actual_embed_dim = model.embed_dim
        if actual_embed_dim is None and hasattr(model, 'model'):
            inner = model.model
            if hasattr(inner, 'embedding') and hasattr(inner.embedding, 'weight'):
                actual_embed_dim = inner.embedding.weight.shape[1]
        if actual_embed_dim is None and hasattr(model, 'embedding'):
            actual_embed_dim = model.embedding.weight.shape[1]
        if actual_embed_dim is not None and actual_embed_dim != GNN_EXPORT_EMBED_DIM:
            raise ValueError(
                f"Model embedding dim {actual_embed_dim} != required {GNN_EXPORT_EMBED_DIM}"
            )
    else:
        model = LightGCN(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=GNN_EXPORT_EMBED_DIM,
            num_layers=args.num_layers,
        )
        model.load_state_dict(torch.load(local_model, map_location="cpu"))
        model.eval()

    edge_index = feats["edge_index"]
    if edge_index.dtype != torch.long:
        edge_index = edge_index.long()
    edge_weight = feats.get("edge_weight")

    logger.info("Generating embeddings...")
    with torch.no_grad():
        u_emb, i_emb = model(edge_index, edge_weight)
        u_emb = u_emb.cpu().numpy()
        i_emb = i_emb.cpu().numpy()

    u_emb = u_emb / (np.linalg.norm(u_emb, axis=1, keepdims=True) + 1e-8)
    i_emb = i_emb / (np.linalg.norm(i_emb, axis=1, keepdims=True) + 1e-8)
    if u_emb.shape[1] != GNN_EXPORT_EMBED_DIM or i_emb.shape[1] != GNN_EXPORT_EMBED_DIM:
        raise ValueError(
            f"Expected dim {GNN_EXPORT_EMBED_DIM}, got u={u_emb.shape[1]} i={i_emb.shape[1]}"
        )
    logger.info("gnn_embed_dim=%s (Triton feature_dim=%s)", GNN_EXPORT_EMBED_DIM, 8 * GNN_EXPORT_EMBED_DIM + 32)

    user_file = "user_embeddings.parquet"
    movie_file = "movie_embeddings.parquet"
    dp = args.data_path_data_prep.rstrip("/")
    um = pd.read_parquet(f"{dp}/users_map.parquet").sort_values("user_idx").reset_index(drop=True)
    im = pd.read_parquet(f"{dp}/items_map.parquet").sort_values("item_idx").reset_index(drop=True)
    uid_key = "userId" if "userId" in um.columns else "user_id"
    mid_key = "movieId" if "movieId" in im.columns else "movie_id"
    u_f = u_emb.astype(np.float32)
    i_f = i_emb.astype(np.float32)
    if len(um) != len(u_f) or len(im) != len(i_f):
        raise ValueError("users_map/items_map row count mismatch vs embeddings")
    pd.DataFrame(
        {
            "user_id": um[uid_key].astype("int64").values,
            "embedding": [u_f[j].tolist() for j in range(len(u_f))],
        }
    ).to_parquet(user_file, index=False)
    pd.DataFrame(
        {
            "movie_id": im[mid_key].astype("int64").values,
            "embedding": [i_f[j].tolist() for j in range(len(i_f))],
        }
    ).to_parquet(movie_file, index=False)

    upload_to_s3(user_file, f"{args.output_path}/user_embeddings.parquet")
    upload_to_s3(movie_file, f"{args.output_path}/movie_embeddings.parquet")
    
    logger.info(f"Embeddings successfully uploaded to {args.output_path}")

if __name__ == "__main__":
    main()
