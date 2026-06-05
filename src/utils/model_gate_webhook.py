import argparse
import json
import logging
from typing import Optional

import mlflow
import requests
from mlflow.tracking import MlflowClient


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("model-gate-webhook")


def _to_optional_float(raw: str) -> Optional[float]:
    value = (raw or "").strip()
    if value == "":
        return None
    return float(value)


def _to_bool(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def _get_latest_finished_run(client: MlflowClient, experiment_id: str):
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No FINISHED run found in experiment.")
    return runs[0]


def _validate_metrics(
    run_metrics: dict,
    metric_key: str,
    threshold: Optional[float],
):
    details = []
    approved = True

    metric_value = run_metrics.get(metric_key)

    if threshold is not None:
        if metric_value is None:
            approved = False
            details.append(f"missing metric '{metric_key}'")
        elif float(metric_value) > threshold:
            approved = False
            details.append(f"{metric_key}={metric_value} > {threshold}")

    reason = "approved by threshold" if approved else "; ".join(details)
    return approved, reason, metric_value


def _register_and_stage_model(
    model_name: str,
    model_artifact_path: str,
    run_id: str,
    stage: str,
    client: MlflowClient,
):
    model_uri = f"runs:/{run_id}/{model_artifact_path}"
    logger.info("Registering model from %s", model_uri)
    version = mlflow.register_model(model_uri=model_uri, name=model_name)
    client.transition_model_version_stage(
        name=model_name,
        version=version.version,
        stage=stage,
        archive_existing_versions=False,
    )
    return str(version.version)


def _send_webhook(webhook_url: str, payload: dict, timeout_seconds: int):
    logger.info("Sending webhook to Jenkins endpoint")
    response = requests.post(webhook_url, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    logger.info("Webhook sent successfully. status_code=%s", response.status_code)


def main():
    parser = argparse.ArgumentParser(description="Gate model by metrics and notify Jenkins.")
    parser.add_argument("--mlflow-uri", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--model-name", default="gnn-recsys")
    parser.add_argument("--model-stage", default="Staging")
    parser.add_argument("--model-artifact-path", default="reranker_model")
    parser.add_argument("--run-id", default="")

    parser.add_argument("--metric-key", default="val_loss")
    parser.add_argument("--metric-threshold", default="0.1")

    parser.add_argument("--jenkins-webhook-url", default="")
    parser.add_argument("--send-fail-webhook", default="false")
    parser.add_argument("--webhook-timeout-seconds", type=int, default=15)
    parser.add_argument("--pipeline-run-id", default="",
                        help="Argo workflow.uid of the training pipeline run.")
    parser.add_argument("--correlation-id", default="",
                        help="Optional trace id (often same as Argo workflow uid).")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    client = MlflowClient(tracking_uri=args.mlflow_uri)

    if args.run_id:
        run = client.get_run(args.run_id)
    else:
        # Try to get experiment by name, fallback to default experiment if not found
        experiment = client.get_experiment_by_name(args.experiment_name)
        if experiment is None:
            logger.warning(f"Experiment '{args.experiment_name}' not found, using default experiment")
            experiment_id = "0"
        else:
            experiment_id = experiment.experiment_id
        run = _get_latest_finished_run(client, experiment_id)
    run_id = run.info.run_id
    metrics = dict(run.data.metrics)
    threshold = _to_optional_float(args.metric_threshold)

    approved, reason, metric_value = _validate_metrics(
        run_metrics=metrics,
        metric_key=args.metric_key,
        threshold=threshold,
    )

    model_version = ""
    if approved:
        model_version = _register_and_stage_model(
            model_name=args.model_name,
            model_artifact_path=args.model_artifact_path,
            run_id=run_id,
            stage=args.model_stage,
            client=client,
        )

    payload = {
        "model_name": args.model_name,
        "model_version": model_version,
        "stage": args.model_stage,
        "run_id": run_id,
        "pipeline_run_id": args.pipeline_run_id,
        "correlation_id": args.correlation_id or args.pipeline_run_id,
        "status": "APPROVED" if approved else "REJECTED",
        "reason": reason,
        "metrics": {
            args.metric_key: metric_value,
        },
    }

    should_send = approved or _to_bool(args.send_fail_webhook)
    if should_send and args.jenkins_webhook_url:
        _send_webhook(
            webhook_url=args.jenkins_webhook_url,
            payload=payload,
            timeout_seconds=args.webhook_timeout_seconds,
        )
    elif should_send:
        logger.warning("Webhook URL is empty, skipping webhook send.")
    else:
        logger.info("Model rejected and fail-webhook disabled. Skipping webhook.")

    logger.info("Gate result: %s", json.dumps(payload))


if __name__ == "__main__":
    main()
