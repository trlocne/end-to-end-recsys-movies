import os
import sys
import torch
import numpy as np
import logging
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.reranker import DeepFMReRanker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("convert_to_tensorrt")


def _parse_s3_uri(s3_uri: str):
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    bucket = parsed.netloc
    key_prefix = parsed.path.lstrip("/")
    return bucket, key_prefix


def _normalize_version(version: str) -> str:
    value = (version or "").strip().lower()
    if value.startswith("v"):
        value = value[1:]
    if value.isdigit():
        return f"v{int(value):06d}"
    return f"v{value or 'latest'}"


def _build_structured_s3_prefix(base_prefix: str, model_name: str, model_version: str) -> str:
    bucket, key_prefix = _parse_s3_uri(base_prefix)
    version = _normalize_version(model_version)
    key_prefix = key_prefix.rstrip("/")
    return f"s3://{bucket}/{key_prefix}/{model_name}/versions/{version}"


def _upload_artifacts_to_s3(output_dir: str, s3_prefix: str, pointer_name: str, metadata: dict):
    import boto3

    bucket, key_prefix = _parse_s3_uri(s3_prefix)
    key_prefix = key_prefix.rstrip("/")

    local_onnx = os.path.join(output_dir, "reranker.onnx")
    local_engine = os.path.join(output_dir, "reranker.engine")
    local_metadata = os.path.join(output_dir, "metadata.json")

    s3 = boto3.client("s3")

    uploaded = {}
    if os.path.exists(local_onnx):
        onnx_key = f"{key_prefix}/reranker.onnx"
        s3.upload_file(local_onnx, bucket, onnx_key)
        uploaded["onnx_s3_uri"] = f"s3://{bucket}/{onnx_key}"

    if os.path.exists(local_engine):
        engine_key = f"{key_prefix}/reranker.engine"
        s3.upload_file(local_engine, bucket, engine_key)
        uploaded["engine_s3_uri"] = f"s3://{bucket}/{engine_key}"

    metadata_key = f"{key_prefix}/metadata.json"
    s3.upload_file(local_metadata, bucket, metadata_key)
    uploaded["metadata_s3_uri"] = f"s3://{bucket}/{metadata_key}"

    pointer_payload = {
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "engine_built": metadata["engine_built"],
        "engine_s3_uri": uploaded.get("engine_s3_uri", ""),
        "onnx_s3_uri": uploaded.get("onnx_s3_uri", ""),
        "metadata_s3_uri": uploaded["metadata_s3_uri"],
    }
    pointer_key = f"{'/'.join(key_prefix.split('/')[:-2])}/pointers/{pointer_name}.json"
    s3.put_object(
        Bucket=bucket,
        Key=pointer_key,
        Body=json.dumps(pointer_payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    uploaded[f"{pointer_name}_pointer_s3_uri"] = f"s3://{bucket}/{pointer_key}"

    return uploaded

def pytorch_to_onnx(model_path, onnx_path, input_dim=219, embed_dim=64):
    """Converts PyTorch model to ONNX format."""
    logger.info(f"Converting {model_path} to ONNX...")
    
    model = DeepFMReRanker(input_dim=input_dim, embed_dim=embed_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    model.eval()

    dummy_input = torch.randn(1, input_dim)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }
    )
    logger.info(f"ONNX model saved to {onnx_path}")

def onnx_to_tensorrt(onnx_path, engine_path, input_dim=219):
    """Converts ONNX model to TensorRT engine."""
    import tensorrt as trt
    
    logger.info(f"Converting {onnx_path} to TensorRT engine...")
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    
    # Set memory pool limit (1GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    
    with open(onnx_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors):
                logger.error(f"ONNX parsing error: {parser.get_error(error)}")
            return False

    # Optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    
    # Set shape: min (1, dim), opt (32, dim), max (1000, dim)
    profile.set_shape(input_name, (1, input_dim), (32, input_dim), (1000, input_dim))
    config.add_optimization_profile(profile)
    
    logger.info("Building TensorRT engine (this may take some time)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        logger.error("Engine build failed")
        return False
        
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    
    logger.info(f"TensorRT engine saved to {engine_path}")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to reranker.pt")
    parser.add_argument("--output-dir", required=True, help="Directory to save conversion artifacts")
    parser.add_argument("--input-dim", type=int, default=544)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--model-name", default="reranker", help="Logical model name for S3 layout/pointers")
    parser.add_argument("--model-version", default="latest", help="Model version label for metadata/output")
    parser.add_argument(
        "--release-pointer",
        default="production",
        help="Pointer name updated for serving, e.g. production or champion",
    )
    parser.add_argument(
        "--s3-output-prefix",
        default="",
        help="Optional fixed S3 prefix to upload artifacts directly",
    )
    parser.add_argument(
        "--s3-base-prefix",
        default="",
        help="Optional S3 base prefix to auto-structure artifacts, e.g. s3://recsys-moivelens/processed/production",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    onnx_path = os.path.join(args.output_dir, "reranker.onnx")
    engine_path = os.path.join(args.output_dir, "reranker.engine")
    
    pytorch_to_onnx(args.model_path, onnx_path, args.input_dim, args.embed_dim)

    trt_built = False
    try:
        import tensorrt
        success = onnx_to_tensorrt(onnx_path, engine_path, args.input_dim)
        if success:
            trt_built = True
        else:
            logger.warning("TensorRT conversion failed. Continuing with ONNX-only artifacts.")
    except ImportError:
        logger.warning("TensorRT not installed. Skipping engine conversion.")
    except Exception as e:
        logger.warning("TensorRT conversion skipped due to runtime error: %s", e)

    metadata = {
        "model_name": args.model_name,
        "model_version": args.model_version,
        "model_path": args.model_path,
        "input_dim": args.input_dim,
        "embed_dim": args.embed_dim,
        "onnx_path": onnx_path,
        "engine_path": engine_path if trt_built else "",
        "engine_built": trt_built,
    }
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Local artifacts:")
    logger.info("ONNX_PATH=%s", onnx_path)
    logger.info("ENGINE_PATH=%s", engine_path if trt_built else "N/A")
    logger.info("METADATA_PATH=%s", metadata_path)

    resolved_s3_prefix = args.s3_output_prefix
    if not resolved_s3_prefix and args.s3_base_prefix:
        resolved_s3_prefix = _build_structured_s3_prefix(
            base_prefix=args.s3_base_prefix,
            model_name=args.model_name,
            model_version=args.model_version,
        )

    if resolved_s3_prefix:
        uploaded = _upload_artifacts_to_s3(
            args.output_dir,
            resolved_s3_prefix,
            args.release_pointer,
            metadata,
        )
        logger.info("S3 artifacts:")
        for key, value in uploaded.items():
            logger.info("%s=%s", key.upper(), value)

if __name__ == "__main__":
    main()
