#!/usr/bin/env python3
"""
Promote champion models to production alias in MLflow.
After promotion, sends:
  - Grafana annotation (model-promote event)
  - Slack notification
"""
import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone

try:
    from mlflow import MlflowClient
except ImportError:
    print("Error: mlflow not installed", file=sys.stderr)
    sys.exit(1)


def get_env_or_exit(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        print(f"Error: {key} not set", file=sys.stderr)
        sys.exit(1)
    return value


def promote_model(client: MlflowClient, model_name: str,
                  champion_alias: str, production_alias: str) -> str:
    champion = client.get_model_version_by_alias(model_name, champion_alias)
    client.set_registered_model_alias(model_name, production_alias, champion.version)
    print(f"[INFO] promoted {model_name}@{champion_alias} (v{champion.version}) -> @{production_alias}")
    return champion.version


def notify_grafana(grafana_url: str, api_key: str, text: str, tags: list[str]) -> None:
    if not grafana_url or not api_key:
        print("[WARN] GRAFANA_URL or GRAFANA_API_KEY not set, skipping annotation")
        return
    payload = json.dumps({
        "dashboardUID": "",
        "panelId": 0,
        "time": int(datetime.now(timezone.utc).timestamp() * 1000),
        "tags": tags,
        "text": text,
    }).encode()
    req = urllib.request.Request(
        f"{grafana_url.rstrip('/')}/api/annotations",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"[INFO] Grafana annotation created: {resp.read().decode()}")
    except urllib.error.URLError as e:
        print(f"[WARN] Grafana annotation failed: {e}")


def notify_slack(webhook_url: str, text: str) -> None:
    if not webhook_url:
        print("[WARN] SLACK_WEBHOOK_URL not set, skipping Slack notification")
        return
    payload = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"[INFO] Slack notification sent: {resp.read().decode()}")
    except urllib.error.URLError as e:
        print(f"[WARN] Slack notification failed: {e}")


def main():
    cid = os.environ.get("CORRELATION_ID", "").strip()
    if cid:
        print(f"[CORRELATION] {cid}")

    tracking_uri = get_env_or_exit("MLFLOW_TRACKING_URI")
    rerank_name  = get_env_or_exit("MODEL_NAME")
    gnn_name     = get_env_or_exit("GNN_MODEL_NAME")
    champion_alias    = get_env_or_exit("MODEL_ALIAS")
    production_alias  = get_env_or_exit("PRODUCTION_ALIAS")

    grafana_url  = os.environ.get("GRAFANA_URL", "")
    grafana_key  = os.environ.get("GRAFANA_API_KEY", "")
    slack_url    = os.environ.get("SLACK_WEBHOOK_URL", "")

    client = MlflowClient(tracking_uri=tracking_uri)

    rerank_ver = promote_model(client, rerank_name, champion_alias, production_alias)
    gnn_ver    = promote_model(client, gnn_name,    champion_alias, production_alias)

    msg = (
        f":rocket: *Model promoted to production*\n"
        f"• `{gnn_name}` v{gnn_ver} -> @{production_alias}\n"
        f"• `{rerank_name}` v{rerank_ver} -> @{production_alias}\n"
        f"• correlation_id: `{cid or 'n/a'}`"
    )

    notify_grafana(
        grafana_url, grafana_key,
        text=f"Model promoted: {gnn_name} v{gnn_ver}, {rerank_name} v{rerank_ver}",
        tags=["model-promote", "production", gnn_name, rerank_name],
    )
    notify_slack(slack_url, msg)


if __name__ == "__main__":
    main()