import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.convert_to_tensorrt import (
    _build_structured_s3_prefix,
    _upload_artifacts_to_s3,
    onnx_to_tensorrt,
    pytorch_to_onnx,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("convert_champion_to_tensorrt")


def _find_pt_file(path: str) -> str:
    p = Path(path)
    if p.is_file() and p.suffix == ".pt":
        return str(p)
    if p.is_dir():
        candidates = sorted(p.rglob("*.pt"))
        if candidates:
            return str(candidates[0])
    raise FileNotFoundError(f"No .pt model file found under: {path}")


def _resolve_model_dims(
    client: MlflowClient, run_id: str, input_dim: Optional[int], embed_dim: Optional[int]
):
    run = client.get_run(run_id)
    params = run.data.params or {}

    resolved_input_dim = input_dim
    resolved_embed_dim = embed_dim

    if resolved_input_dim is None:
        value = params.get("input_dim")
        if value is not None:
            resolved_input_dim = int(value)

    if resolved_embed_dim is None:
        value = params.get("embed_dim")
        if value is not None:
            resolved_embed_dim = int(value)

    if resolved_input_dim is None:
        resolved_input_dim = 219
        logger.warning("input_dim not found in MLflow params, fallback to default=%s", resolved_input_dim)

    if resolved_embed_dim is None:
        resolved_embed_dim = 64
        logger.warning("embed_dim not found in MLflow params, fallback to default=%s", resolved_embed_dim)

    return resolved_input_dim, resolved_embed_dim

def main():
    parser = argparse.ArgumentParser(description="Convert MLflow champion model to ONNX/TensorRT.")
    parser.add_argument("--mlflow-uri", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-alias", default="champion")
    parser.add_argument("--artifact-path", default="", help="Optional artifact path in MLflow run")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-dim", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--release-pointer", default="production")
    parser.add_argument("--s3-base-prefix", default="")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    client = MlflowClient(tracking_uri=args.mlflow_uri)
    champion = client.get_model_version_by_alias(args.model_name, args.model_alias)
    run_id = champion.run_id
    model_version = str(champion.version)

    logger.info(
        "Resolved model alias: name=%s alias=%s version=%s run_id=%s",
        args.model_name,
        args.model_alias,
        model_version,
        run_id,
    )
    input_dim, embed_dim = _resolve_model_dims(client, run_id, args.input_dim, args.embed_dim)
    logger.info("Resolved model dims: input_dim=%s embed_dim=%s", input_dim, embed_dim)

    os.makedirs(args.output_dir, exist_ok=True)
    local_artifact = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=args.artifact_path if args.artifact_path else None,
    )
    model_path = _find_pt_file(local_artifact)
    logger.info("Using model file: %s", model_path)

    onnx_path = os.path.join(args.output_dir, "reranker.onnx")
    engine_path = os.path.join(args.output_dir, "reranker.engine")
    pytorch_to_onnx(model_path, onnx_path, input_dim, embed_dim)

    trt_built = False
    try:
        trt_built = onnx_to_tensorrt(onnx_path, engine_path, input_dim)
        if not trt_built:
            logger.warning("TensorRT build returned false. Continuing with ONNX-only.")
    except Exception as e:
        logger.warning("TensorRT conversion skipped due to runtime error: %s", e)

    metadata = {
        "model_name": args.model_name,
        "model_alias": args.model_alias,
        "model_version": model_version,
        "run_id": run_id,
        "source_model_path": model_path,
        "input_dim": input_dim,
        "embed_dim": embed_dim,
        "engine_built": trt_built,
    }
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("LOCAL_ONNX_PATH=%s", onnx_path)
    logger.info("LOCAL_ENGINE_PATH=%s", engine_path if trt_built else "N/A")
    logger.info("LOCAL_METADATA_PATH=%s", metadata_path)

    if args.s3_base_prefix:
        s3_prefix = _build_structured_s3_prefix(args.s3_base_prefix, args.model_name, model_version)
        uploaded = _upload_artifacts_to_s3(args.output_dir, s3_prefix, args.release_pointer, metadata)
        for key, value in uploaded.items():
            logger.info("%s=%s", key.upper(), value)


if __name__ == "__main__":
    main()
