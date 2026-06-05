#!/usr/bin/env python3
"""Champion pair gate: MLflow tags (hard vs warn) + optional inference contract (GNN embed_dim vs rerank embed_dim).

Hard mismatches block auto-promote (NEEDS_APPROVAL / has_mismatch). Warn-only tag differences are reported in
``warnings`` but do not set has_mismatch.

Rerank DeepFM logs both embed_dim (FM embedding size) and input_dim (full feature width). Pairing GNN item/user
embed_dim with rerank embed_dim is meaningful; comparing GNN embed_dim to rerank input_dim is not.

Env (required): MLFLOW_TRACKING_URI, MODEL_NAME, GNN_MODEL_NAME, MODEL_ALIAS, PRODUCTION_ALIAS

Tag sync (priority):
  1. If SYNC_TAG_KEYS is non-empty (legacy): use it as the **hard** key list only; optional WARN_SYNC_TAG_KEYS for
     soft checks. No default warn keys.
  2. Else: HARD_SYNC_TAG_KEYS defaults to dataset_version,feature_schema_version; WARN_SYNC_TAG_KEYS defaults to
     pipeline_run_id. If STRICT_PIPELINE_RUN_ID is true, pipeline_run_id is promoted to the hard list.

Other: STRICT_CROSS_MODEL_EMBED_DIM; CHECK_INFERENCE_CONTRACT (default on); CHECK_RERANK_INPUT_MATCHES_GNN_EMBED;
MLFLOW_PAIR_MANIFEST_TAG; CORRELATION_ID."""
import json
import os
import sys

try:
    from mlflow import MlflowClient
except ImportError:
    print("Error: mlflow not installed", file=sys.stderr)
    sys.exit(1)


def _req(k: str) -> str:
    v = os.environ.get(k)
    if not v:
        print(f"Error: {k} not set", file=sys.stderr)
        sys.exit(1)
    return v


def _correlation() -> str:
    return os.environ.get("CORRELATION_ID", "").strip()


def _truthy(k: str, default: bool = False) -> bool:
    v = os.environ.get(k, "").lower()
    return default if not v else v in ("1", "true", "yes", "y")


def _csv_keys(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _resolve_tag_key_lists() -> tuple[list[str], list[str]]:
    """Return (hard_keys, warn_keys)."""
    legacy_raw = os.environ.get("SYNC_TAG_KEYS", "").strip()
    if legacy_raw:
        hard = _csv_keys(legacy_raw)
        warn_raw = os.environ.get("WARN_SYNC_TAG_KEYS", "").strip()
        warn = _csv_keys(warn_raw) if warn_raw else []
        return hard, warn

    hard_raw = os.environ.get("HARD_SYNC_TAG_KEYS", "").strip()
    hard = _csv_keys(hard_raw) if hard_raw else ["dataset_version", "feature_schema_version"]

    warn_raw = os.environ.get("WARN_SYNC_TAG_KEYS", "").strip()
    warn = _csv_keys(warn_raw) if warn_raw else ["pipeline_run_id"]

    if _truthy("STRICT_PIPELINE_RUN_ID"):
        if "pipeline_run_id" not in hard:
            hard.append("pipeline_run_id")
        warn = [k for k in warn if k != "pipeline_run_id"]

    return hard, warn


def _alias(client: MlflowClient, name: str, al: str):
    try:
        m = client.get_model_version_by_alias(name, al)
        return m.version, m.run_id
    except Exception:
        return None, None


def _tags_params(client: MlflowClient, run_id: str):
    try:
        r = client.get_run(run_id)
        return r.data.tags, r.data.params
    except Exception:
        return {}, {}


def _compare_tag_keys(
    keys: list[str],
    r_tags: dict,
    g_tags: dict,
    *,
    label: str,
) -> list[dict]:
    out = []
    for key in keys:
        rv, gv = r_tags.get(key), g_tags.get(key)
        print(f"[CHECK] {label} {key}: rerank={rv} gnn={gv}", file=sys.stderr)
        if rv != gv:
            out.append({"key": key, "rerank": rv, "gnn": gv})
    return out


def _check_metric_threshold(
    client: MlflowClient,
    run_id: str,
    metric_key: str,
    threshold: float,
    model_label: str,
) -> dict | None:
    try:
        r = client.get_run(run_id)
        val = r.data.metrics.get(metric_key)
        print(f"[CHECK] {model_label} {metric_key}={val} threshold={threshold}", file=sys.stderr)
        if val is None:
            print(f"[WARN] {model_label} metric {metric_key} not found in run {run_id}", file=sys.stderr)
            return None
        if val > threshold:
            return {"key": metric_key, "model": model_label, "value": val, "threshold": threshold}
    except Exception as e:
        print(f"[WARN] failed to fetch metrics for {model_label}: {e}", file=sys.stderr)
    return None


def validate_release() -> dict:
    c = MlflowClient(_req("MLFLOW_TRACKING_URI"))
    cand, prod = _req("MODEL_ALIAS"), _req("PRODUCTION_ALIAS")
    rr_n, gnn_n = _req("MODEL_NAME"), _req("GNN_MODEL_NAME")
    hard_keys, warn_keys = _resolve_tag_key_lists()

    metric_key = os.environ.get("METRIC_KEY", "val_loss").strip()
    metric_threshold_raw = os.environ.get("METRIC_THRESHOLD", "").strip()
    metric_threshold = float(metric_threshold_raw) if metric_threshold_raw else None
    mk = os.environ.get("MLFLOW_PAIR_MANIFEST_TAG", "").strip()
    if mk:
        if mk not in hard_keys:
            hard_keys = hard_keys + [mk]
        if mk in warn_keys:
            warn_keys = [k for k in warn_keys if k != mk]

    rr_v, rr_rid = _alias(c, rr_n, cand)
    gnn_v, gnn_rid = _alias(c, gnn_n, cand)
    rr_pv, _ = _alias(c, rr_n, prod)
    gnn_pv, _ = _alias(c, gnn_n, prod)

    base_fail = {
        "release_state": "FAILED",
        "require_approval": False,
        "has_mismatch": False,
        "rerank_champion": None,
        "rerank_prod": rr_pv,
        "gnn_champion": None,
        "gnn_prod": gnn_pv,
        "mismatches": [],
        "warnings": [],
        "correlation_id": _correlation(),
    }

    if rr_v is None or gnn_v is None:
        return base_fail

    g_tags, g_params = _tags_params(c, gnn_rid)
    r_tags, r_params = _tags_params(c, rr_rid)
    mismatches: list[dict] = []

    if metric_threshold is not None:
        for label, run_id in [(gnn_n, gnn_rid), (rr_n, rr_rid)]:
            m = _check_metric_threshold(c, run_id, metric_key, metric_threshold, label)
            if m:
                mismatches.append(m)

    hard_set = set(hard_keys)
    warn_only = [k for k in warn_keys if k not in hard_set]
    mismatches.extend(_compare_tag_keys(hard_keys, r_tags, g_tags, label="hard"))

    warnings: list[dict] = _compare_tag_keys(warn_only, r_tags, g_tags, label="warn")

    g_ed = g_params.get("embed_dim", "")
    r_ed = r_params.get("embed_dim", "")
    r_in = r_params.get("input_dim", "")

    if _truthy("STRICT_CROSS_MODEL_EMBED_DIM"):
        print(f"[CHECK] embed_dim legacy: gnn={g_ed} rerank={r_ed}", file=sys.stderr)
        if str(g_ed) != str(r_ed):
            mismatches.append({"key": "embed_dim", "rerank": r_ed, "gnn": g_ed})

    if _truthy("CHECK_INFERENCE_CONTRACT", True):
        if _truthy("CHECK_RERANK_INPUT_MATCHES_GNN_EMBED"):
            if r_in and g_ed:
                print(f"[CHECK] contract (legacy): gnn_embed={g_ed} rerank_input={r_in}", file=sys.stderr)
                if str(g_ed) != str(r_in):
                    mismatches.append({"key": "rerank_input_dim_vs_gnn_embed_dim", "rerank": r_in, "gnn": g_ed})
            else:
                print("[CHECK] contract legacy: skipped (missing input_dim and/or gnn embed_dim)", file=sys.stderr)
        elif r_ed and g_ed:
            print(f"[CHECK] contract: gnn_embed={g_ed} rerank_embed={r_ed}", file=sys.stderr)
            if str(g_ed) != str(r_ed):
                mismatches.append({"key": "rerank_embed_dim_vs_gnn_embed_dim", "rerank": r_ed, "gnn": g_ed})
        else:
            print("[CHECK] contract: skipped (missing embed_dim on gnn and/or rerank)", file=sys.stderr)

    bad = len(mismatches) > 0
    r_ch, g_ch = rr_pv != rr_v, gnn_pv != gnn_v
    if bad:
        state, appr = "NEEDS_APPROVAL", True
    elif r_ch and g_ch:
        state, appr = "READY_TO_PROMOTE", False
    elif r_ch or g_ch:
        state, appr = "NEEDS_APPROVAL", True
    else:
        state, appr = "NOOP", False

    return {
        "release_state": state,
        "require_approval": appr,
        "has_mismatch": bad,
        "rerank_champion": rr_v,
        "rerank_prod": rr_pv,
        "gnn_champion": gnn_v,
        "gnn_prod": gnn_pv,
        "mismatches": mismatches,
        "warnings": warnings,
        "correlation_id": _correlation(),
    }


if __name__ == "__main__":
    print(json.dumps(validate_release()))
