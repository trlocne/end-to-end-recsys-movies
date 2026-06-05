import os
import inspect
import subprocess
import sys
from urllib.parse import urlparse

import boto3
import mlflow
import torch
from mlflow import MlflowClient

tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
model_name = os.environ["MODEL_NAME"]
prod_alias = os.environ["PRODUCTION_ALIAS"]
s3_prefix = os.environ["S3_BASE_PREFIX"].rstrip("/")

mlflow.set_tracking_uri(tracking_uri)

model_uri = f"models:/{model_name}@{prod_alias}"

print(f"[INFO] loading {model_uri}")

model = mlflow.pytorch.load_model(model_uri)
model.eval()

input_dim = model.fm.w.in_features

onnx_path = "/tmp/reranker.onnx"

_export_kw = dict(
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)
_sig = inspect.signature(torch.onnx.export)
if "dynamo" in _sig.parameters:
    _export_kw["dynamo"] = False
if "external_data" in _sig.parameters:
    _export_kw["external_data"] = False

torch.onnx.export(
    model,
    torch.randn(1, input_dim),
    onnx_path,
    **_export_kw,
)

# Newer torch.onnx.export can write weights to reranker.onnx.data. Triton only pulls
# reranker.onnx from S3 unless we also upload the sidecar, so prefer one self-contained file.


def _ensure_onnx():
    try:
        return __import__("onnx")
    except ImportError:
        print("[INFO] onnx not installed; pip installing for external_data inlining")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "onnx"],
            stdout=sys.stderr,
        )
        return __import__("onnx")


def _onnx_has_external_data_refs(onnx_mod, path: str) -> bool:
    model = onnx_mod.load(path, load_external_data=False)
    for init in model.graph.initializer:
        for entry in init.external_data:
            if entry.key == "location" and entry.value:
                return True
    return False


def _inline_onnx_external_weights(onnx_mod, path: str) -> None:
    model_proto = onnx_mod.load(path, load_external_data=True)
    tmp = path + ".single.onnx"
    onnx_mod.save_model(model_proto, tmp, save_as_external_data=False)
    os.replace(tmp, path)
    sidecar = path + ".data"
    if os.path.isfile(sidecar):
        os.remove(sidecar)


_onnx = _ensure_onnx()
try:
    _inline_onnx_external_weights(_onnx, onnx_path)
    print("[INFO] inlined ONNX external_data into single reranker.onnx")
except Exception as exc:
    print(f"[WARN] ONNX inline failed: {exc}")

_sidecar_path = onnx_path + ".data"
if _onnx_has_external_data_refs(_onnx, onnx_path) and not os.path.isfile(_sidecar_path):
    raise SystemExit(
        "Export produced reranker.onnx that still references external_data but "
        "reranker.onnx.data is missing and inlining failed. "
        "Ensure the export image can `pip install onnx` or bundle onnx, then re-run. "
        "Refusing to upload a stub that will break Triton."
    )

_bytes = os.path.getsize(onnx_path)

client = MlflowClient(tracking_uri=tracking_uri)

prod_version = client.get_model_version_by_alias(
    model_name,
    prod_alias
)

version = f"v{prod_version.version}"

parsed = urlparse(s3_prefix)

bucket = parsed.netloc
base_key = parsed.path.lstrip("/")

artifact_key = (
    f"{base_key}/"
    f"{model_name}/"
    f"{version}/"
    f"reranker.onnx"
)

pointer_key = (
    f"{base_key}/"
    f"{model_name}/"
    f"{prod_alias}.txt"
)

s3 = boto3.client("s3")

s3.upload_file(
    onnx_path,
    bucket,
    artifact_key
)

sidecar_path = onnx_path + ".data"
if os.path.isfile(sidecar_path):
    sidecar_key = artifact_key + ".data"
    s3.upload_file(sidecar_path, bucket, sidecar_key)
    print(f"[INFO] uploaded companion s3://{bucket}/{sidecar_key}")

pointer = f"""
model={model_name}
alias={prod_alias}
version={prod_version.version}
run_id={prod_version.run_id}
artifact=s3://{bucket}/{artifact_key}
""".strip()

s3.put_object(
    Bucket=bucket,
    Key=pointer_key,
    Body=pointer.encode("utf-8")
)

print(f"[INFO] uploaded s3://{bucket}/{artifact_key}")
print(f"[INFO] pointer  s3://{bucket}/{pointer_key}")

# Output ONNX path for Jenkins to capture
onnx_s3_path = f"s3://{bucket}/{artifact_key}"
# Write to workspace directory (Jenkins mounts workspace)
workspace_dir = os.getenv("WORKSPACE", "/tmp")
with open(f"{workspace_dir}/onnx_path.txt", "w") as f:
    f.write(onnx_s3_path)
print(f"[OUTPUT] ONNX_PATH={onnx_s3_path}")
