# RecSys — End-to-End Movie Recommendation System

A production-ready movie recommendation system built on MovieLens, covering the full ML lifecycle: CDC ingestion, real-time feature store, GNN + reranker training, and serving via FastAPI + Triton Inference Server. The entire pipeline runs on Kubernetes (DigitalOcean), orchestrated by Argo Workflows, and deployed GitOps-style via ArgoCD.

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Service Map](#2-service-map)
3. [Namespace Map](#3-namespace-map)
4. [Tech Stack](#4-tech-stack)
5. [Monitoring & Alerting](#5-monitoring--alerting)
6. [Prerequisites](#6-prerequisites)
7. [Cluster Bootstrap](#7-cluster-bootstrap)
8. [Data Infrastructure](#8-data-infrastructure)
9. [PostgreSQL + Debezium Setup](#9-postgresql--debezium-setup)
10. [Flink CDC Job](#10-flink-cdc-job)
11. [Training Pipeline](#11-training-pipeline)
12. [Model Gate + Promote](#12-model-gate--promote)
13. [Serving Update](#13-serving-update)
14. [Serving API](#14-serving-api)
15. [CronWorkflow — Incremental Refresh](#15-cronworkflow--incremental-refresh)
16. [Repository Structure](#16-repository-structure)
17. [Secrets & Security](#17-secrets--security)
18. [Quick Reference](#18-quick-reference)

---

## 1. Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║  REAL-TIME PATH                                                       ║
║                                                                       ║
║  PostgreSQL ──WAL──► Debezium ──► Kafka ──► Flink CDC Job            ║
║  (Aiven)             ns:infra    ns:kafka    ns:infra                 ║
║                                                  │                   ║
║                                    ┌─────────────┴──────────────┐    ║
║                                    ▼                            ▼    ║
║                             Feast Online Store            S3 Offline  ║
║                             Redis (Aiven)                 (AWS S3)   ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║  BATCH TRAINING PATH  (Argo DAG · ns:argo)                            ║
║                                                                       ║
║  Feast Offline ──► AWS Glue ──► data-prep ──► LightGCN (Ray)         ║
║       ──► Milvus index ──► DeepFM Reranker (Ray) ──► model gate      ║
║       ──► MLflow Registry  (alias: champion)  ns:mlflow              ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║  SERVING UPDATE PATH  (Argo DAG · ns:argo → ns:serving)              ║
║                                                                       ║
║  MLflow (alias:production) ──► inference_only ──► embeddings → S3    ║
║       ──► Milvus blue-green swap ──► ES index rebuild                ║
║       ──► Triton model reload ──► rollout serving pod                ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║  SERVING  (ns:serving · port 8000)                                    ║
║                                                                       ║
║  /recommend ──► Feast online (user emb) ──► Milvus ANN (LightGCN)   ║
║             ──► Elasticsearch BM25 ──► DeepFM Reranker (Triton)     ║
║             ──► MMR diversity ──► top-K results                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 2. Service Map

| Service | Namespace | Port |
|---|---|---|
| FastAPI serving | `serving` | 8000 |
| Triton Inference Server | `serving` | 8001 (gRPC) / 8002 (metrics) |
| recsys-ui | `serving` | 3000 |
| MLflow | `mlflow` | 5000 |
| Postgres (MLflow backend) | `mlflow` | 5432 |
| Argo Server | `argo` | 2746 |
| ArgoCD | `argocd` | 443 |
| Grafana | `monitoring` | 3000 |
| Prometheus | `monitoring` | 9090 |
| Alertmanager | `monitoring` | 9093 |
| Loki | `monitoring` | 3100 |
| Kafka bootstrap | `kafka` | 9092 |
| Kafka UI | `kafka` | 8080 |
| Kafka Connect | `infra` | 8083 |
| Schema Registry | `infra` | 8081 |
| Debezium UI | `infra` | 8080 |
| Flink UI | `infra` | 8081 |
| Milvus | `infra` | 19530 |
| Elasticsearch | `infra` | 9200 |
| Aiven PostgreSQL | external | 27400 |
| Aiven Redis | external | — |
| AWS S3 | external | — |

---

## 3. Namespace Map

| Namespace | Contents |
|---|---|
| `argo` | Argo Workflows, KubeRay operator, WorkflowTemplates |
| `argocd` | ArgoCD GitOps controller |
| `serving` | FastAPI, Triton, recsys-ui |
| `mlflow` | MLflow server, Postgres backend |
| `infra` | Kafka Connect/Debezium, Elasticsearch, Flink, Milvus, Schema Registry |
| `kafka` | Strimzi Kafka cluster (KRaft) |
| `monitoring` | Prometheus, Grafana, Loki, Alertmanager, Alloy |

---

## 4. Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | Argo Workflows (DAG), ArgoCD (GitOps) |
| Training | PyTorch (LightGCN, DeepFM), Ray (distributed), AWS Glue (PySpark) |
| Feature Store | Feast (offline: S3/Parquet, online: Aiven Redis) |
| Model Registry | MLflow (PostgreSQL backend, S3 artifacts) |
| Inference | Triton Inference Server (ONNX) |
| Vector Search | Milvus (ANN) |
| Text Search | Elasticsearch (BM25) |
| CDC | Debezium → Kafka (Strimzi KRaft) → Flink (PyFlink) |
| Serving API | FastAPI |
| Monitoring | Prometheus + Grafana + Loki + Alertmanager |
| Cloud | DigitalOcean Kubernetes, AWS S3 |
| External Services | Aiven PostgreSQL (CDC source), Aiven Redis (online store) |

---

## 5. Monitoring & Alerting

### Stack

```
Prometheus ──► Grafana (port 3000)
     │
     └──► Alertmanager ──► Slack channels
Loki ◄── Alloy (DaemonSet log collector)
```

### Grafana

```bash
kubectl port-forward svc/recsys-monitoring-grafana -n monitoring 3000:80
# → http://localhost:3000  (admin / recsys-grafana)
```

### Slack Alert Routing

| Receiver | Channel | Trigger |
|---|---|---|
| `critical` | `#recsys-alerts-critical` | `severity=critical` |
| `ml-team` | `#recsys-ml-alerts` | `team=ml` |
| `model-promote` | `#recsys-deploys` | After successful promote |

### Prometheus Scrape Targets

| Job | Namespace | Port | Path |
|---|---|---|---|
| `recsys-serving` | `serving` | 8000 | `/metrics` |
| `recsys-triton` | `serving` | 8002 | `/metrics` |
| `mlflow` | `mlflow` | 5000 | `/metrics` |
| `argo-workflows` | `argo` | 9090 | `/metrics` |
| `kafka` | `kafka` | 9404 | `/metrics` |

---

## 6. Prerequisites

### Local Tools

| Tool | Version |
|---|---|
| `kubectl` | ≥ 1.28 |
| `helm` | ≥ 3.12 |
| `argo` CLI | ≥ 3.5 |
| `argocd` CLI | ≥ 2.9 |
| `docker` | ≥ 24 |
| `psql` | ≥ 14 |

### Kubeconfig

```bash
export KUBECONFIG=deployments/k8s/account/seanmovies-kubeconfig.yaml
```

### Helm Repositories

```bash
helm repo add argoproj             https://argoproj.github.io/argo-helm
helm repo add kuberay              https://ray-project.github.io/kuberay-helm/
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana              https://grafana.github.io/helm-charts
helm repo add strimzi              https://strimzi.io/charts/
helm repo update
```

---

## 7. Cluster Bootstrap

### 7.1 Namespaces + RBAC

```bash
kubectl apply -f deployments/k8s/namespaces/namespaces.yaml
kubectl apply -f deployments/k8s/rbac/
```

### 7.2 ArgoCD

```bash
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/v3.4.3/manifests/install.yaml
kubectl rollout status deployment/argocd-server -n argocd

# Retrieve initial admin password
kubectl get secret argocd-initial-admin-secret -n argocd \
  -o jsonpath="{.data.password}" | base64 -d

kubectl port-forward svc/argocd-server -n argocd 8080:443
# → https://localhost:8080
```

### 7.3 Argo Workflows + KubeRay

```bash
cd deployments/helm/platform/argo-ray
helm dependency update
helm upgrade --install recsys-argo-ray . \
  -n argo --create-namespace -f values.yaml
```

### 7.4 MLflow

```bash
helm upgrade --install recsys-mlflow deployments/helm/infra/mlflow \
  -f deployments/helm/infra/mlflow/values.do.yaml \
  -n mlflow --create-namespace
```

> Use `values.do.yaml` for DigitalOcean (StorageClass: `do-block-storage`) or `values.aws.yaml` for AWS EKS (StorageClass: `ebs-sc`).

### 7.5 Monitoring

```bash
cd deployments/helm/platform/monitoring
helm dependency update
helm upgrade --install recsys-monitoring . \
  -n monitoring --create-namespace -f values.yaml \
  --wait --timeout 5m
```

### 7.6 Triton + Serving

```bash
helm upgrade --install recsys-triton   deployments/helm/platform/triton     -n serving --create-namespace -f values.yaml
helm upgrade --install recsys-ui       deployments/helm/apps/recsys-ui      -n serving --create-namespace -f values.yaml
```

### 7.7 GitOps (ArgoCD Root App)

```bash
kubectl apply -f deployments/argocd/root-app.yaml -n argocd
kubectl get applications -n argocd
```

### 7.8 WorkflowTemplates

```bash
kubectl apply -f cicd/argo/template/ -n argo
```

---

## 8. Data Infrastructure

### 8.1 Kafka (Strimzi KRaft)

```bash
kubectl create namespace kafka
kubectl apply -f 'https://strimzi.io/install/latest?namespace=kafka' -n kafka
kubectl create clusterrolebinding strimzi-cluster-operator-admin-binding-kafka \
  --clusterrole=cluster-admin \
  --serviceaccount=kafka:strimzi-cluster-operator

kubectl wait --for=condition=established crd/kafkanodepools.kafka.strimzi.io --timeout=120s
kubectl wait --for=condition=established crd/kafkas.kafka.strimzi.io --timeout=120s

kubectl apply -f deployments/k8s/infra/kafka/kafka-nodepool.yaml -n kafka
kubectl apply -f deployments/k8s/infra/kafka/kafka-cluster.yaml  -n kafka
kubectl apply -f deployments/k8s/infra/kafka/kafka-ui-deployment.yaml -n kafka
```

| Parameter | Value |
|---|---|
| Version | Kafka 4.2.0 |
| Mode | KRaft (combined broker+controller) |
| Replicas | 2 |
| Bootstrap | `kafka-kafka-bootstrap.kafka.svc.cluster.local:9092` |

CDC topics (created automatically by Debezium):

| Topic | Source Table |
|---|---|
| `recsys-cdc.public.interactions` | `public.interactions` |
| `recsys-cdc.public.users` | `public.users` |
| `recsys-cdc.public.items` | `public.items` |

### 8.2 Kafka Connect + Schema Registry + Debezium UI

```bash
kubectl apply -f deployments/k8s/infra/debezium/kafka-connect-deployment.yaml   -n infra
kubectl apply -f deployments/k8s/infra/debezium/schema-registry-deployment.yaml -n infra
kubectl apply -f deployments/k8s/infra/debezium/debezium-ui-deployment.yaml     -n infra
```

### 8.3 Milvus + Elasticsearch

```bash
kubectl apply -f deployments/k8s/infra/milvus/        -n infra
kubectl apply -f deployments/k8s/infra/elasticsearch/ -n infra
```

---

## 9. PostgreSQL + Debezium Setup

### 9.1 Create Tables and Publication

```bash
psql "postgres://avnadmin:<password>@pg-30130064-seanhcmut05.c.aivencloud.com:27400/defaultdb?sslmode=require" \
  -f scripts/create_postgres_tables.sql
```

Creates tables `public.interactions`, `public.users`, `public.items`, replication slot `debezium_docker`, and publication `dbz_publication`.

### 9.2 Apply Debezium Secret

```bash
kubectl apply -f deployments/k8s/infra/debezium/debezium-connector-secret.yaml -n infra
```

### 9.3 Register Connector

```bash
kubectl apply -f deployments/k8s/infra/debezium/debezium-connector-configmap.yaml -n infra
kubectl delete job debezium-connector-register -n infra --ignore-not-found
kubectl apply -f deployments/k8s/infra/debezium/debezium-connector-register-job.yaml -n infra
kubectl logs -n infra -l job-name=debezium-connector-register --tail=20
```

### 9.4 Verify

```bash
kubectl port-forward svc/kafka-connect -n infra 8083:8083
curl http://localhost:8083/connectors/postgres-connector/status | python3 -m json.tool
```

**Troubleshooting:**

| Error | Cause | Fix |
|---|---|---|
| `No table filters found for filtered publication` | Publication not created | Re-run `create_postgres_tables.sql` |
| `The connection attempt failed` | Wrong host/port/password | Check `debezium-connector-secret.yaml` |
| `slot_name already exists` | Stale replication slot | Script uses `WHERE NOT EXISTS`, safe to re-run |

---

## 10. Flink CDC Job

### 10.1 Install Flink Kubernetes Operator

```bash
cd flink-kubernetes-operator
kubectl apply -f helm/flink-kubernetes-operator/crds/
kubectl wait --for=condition=established crd/flinkdeployments.flink.apache.org --timeout=60s
helm upgrade --install flink-kubernetes-operator \
  helm/flink-kubernetes-operator \
  --namespace flink-system --create-namespace \
  --set webhook.create=false
```

### 10.2 ServiceAccount + RBAC

```bash
kubectl apply -f deployments/k8s/infra/flink-cluster/flink-service-account.yaml -n infra
```

### 10.3 AWS Credentials for Feast

```bash
kubectl create secret generic aws-credentials -n infra \
  --from-literal=AWS_ACCESS_KEY_ID=<key> \
  --from-literal=AWS_SECRET_ACCESS_KEY=<secret>
```

### 10.4 Deploy Flink CDC Job

```bash
kubectl apply -f deployments/k8s/infra/flink-cluster/flink-cdc-job.yaml -n infra
```

### 10.5 Verify

```bash
kubectl get flinkdeployment flink-cdc-ingest -n infra
kubectl logs -n infra -l app=flink-cdc-ingest,component=jobmanager --tail=30
kubectl port-forward service/flink-cdc-ui 8081:8081 -n infra
# → http://localhost:8081
```

**Troubleshooting:**

| Symptom | Cause | Fix |
|---|---|---|
| `no matches for kind "FlinkDeployment"` | CRDs not installed | Run step 10.1 |
| Job not starting | Operator not ready | Check `kubectl logs -n flink-system -l app=flink-kubernetes-operator` |
| Avro decode errors | Schema Registry not running | Check `schema-registry` pod in `infra` namespace |
| Feast push failed | Missing AWS credentials | Check secret `aws-credentials` in `infra` namespace |

---

## 11. Training Pipeline

### 11.1 DAG Overview

```
Feast historical retrieval
  └── AWS Glue preprocess (PySpark)
        └── data-prep (feature engineering)
              ├── train-gnn  (LightGCN · Ray)
              │     └── index-milvus + Elasticsearch
              └── train-reranker (DeepFM · Ray)
                    └── model-gate → promote → serving-update
```

### 11.2 Submit Workflow

```bash
argo submit cicd/argo/workflow/training-pipeline-submit-workflow.yaml \
  -n argo \
  -p registry=trlocne204 \
  -p num-epochs-gnn=10 \
  -p metric-threshold=0.08 \
  -p skip-tune=true \
  --watch
```

### 11.3 Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `registry` | `trlocne204` | DockerHub image prefix |
| `s3-raw-path` | `s3://recsys-moivelens/processed/dev` | Input data path |
| `mlflow-tracking-uri` | `http://mlflow.mlflow.svc.cluster.local:5000` | MLflow server |
| `metric-key` | `val_loss` | Metric evaluated in model gate |
| `metric-threshold` | `0.1` | Maximum allowed metric value to pass gate |
| `run-serving-update` | `true` | Auto-deploy after promote |
| `skip-tune` | `false` | Skip Ray Tune hyperparameter search |
| `karpenter` | `false` | Set to `true` on AWS with Karpenter |

---

## 12. Model Gate + Promote

### 12.1 Flow

```
run-train-gnn + run-train-reranker
    └── run-validate-release      (metric threshold + tag consistency check)
          └── run-promote-models  (MLflow: Staging → Production alias)
                └── Grafana annotation + Slack #recsys-deploys
                      └── run-serving-update-embedded
```

### 12.2 Notifications Secret

```bash
# Edit deployments/k8s/account/monitoring-creds-secret.yaml with actual values, then:
kubectl apply -f deployments/k8s/account/monitoring-creds-secret.yaml -n argo
```

---

## 13. Serving Update

### 13.1 Manual Trigger

```bash
argo submit -n argo --from workflowtemplate/serving-update \
  -p registry=trlocne204 \
  --watch
```

### 13.2 Update Flow

```
inference_only → extract embeddings → S3
  └── Milvus blue-green collection swap
  └── Elasticsearch index rebuild
  └── Triton model reload (gRPC model control API)
  └── kubectl rollout restart deployment/serving -n serving
```

---

## 14. Serving API

```bash
kubectl port-forward svc/serving -n serving 8000:8000
```

| Endpoint | Method | Description |
|---|---|---|
| `/v1/recommend/home` | GET | Top-K movie recommendations for a user |
| `/v1/feedback/click` | POST | Record interaction → Postgres → CDC pipeline |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

---

## 15. CronWorkflow — Incremental Refresh

Runs daily at 2:00 AM UTC to update embeddings and refresh the serving layer.

```bash
kubectl apply -f cicd/argo/workflow/serving-update-cronworkflow.yaml -n argo
kubectl get cronworkflow -n argo
```

---

## 16. Repository Structure

```
seanmovies/
├── src/                          # Python source code
│   ├── training/                 # LightGCN, DeepFM training scripts
│   ├── serving/                  # FastAPI application
│   ├── feature_stores/           # Feast repo, Flink ingest, transforms
│   └── data_prep/                # Feature engineering
├── scripts/
│   ├── create_postgres_tables.sql
│   ├── validate_release.py
│   └── promote_models.py
├── configs/                      # Training feature configurations
├── cicd/
│   └── argo/
│       ├── template/             # WorkflowTemplates
│       └── workflow/             # Submit manifests + CronWorkflow
├── deployments/
│   ├── argocd/                   # ArgoCD Application manifests
│   ├── docker/                   # Dockerfiles
│   ├── helm/
│   │   ├── apps/
│   │   │   ├── recsys-serving/
│   │   │   └── recsys-ui/
│   │   ├── infra/
│   │   │   └── mlflow/
│   │   │       ├── values.do.yaml
│   │   │       └── values.aws.yaml
│   │   └── platform/
│   │       ├── argo-ray/
│   │       ├── monitoring/
│   │       └── triton/
│   └── k8s/
│       ├── account/
│       ├── infra/
│       │   ├── debezium/
│       │   ├── elasticsearch/
│       │   ├── flink-cluster/
│       │   ├── kafka/
│       │   └── milvus/
│       ├── namespaces/
│       └── rbac/
└── flink-kubernetes-operator/
    └── helm/
        └── flink-kubernetes-operator/
            └── crds/
```

---

## 17. Secrets & Security

| Secret | Namespace | Contents |
|---|---|---|
| `aws-creds` | `mlflow`, `infra`, `argo` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` |
| `debezium-connector-secret` | `infra` | Aiven PostgreSQL credentials |
| `monitoring-creds` | `argo` | Grafana API key, Slack webhook URL |
| `postgres-creds` | `argo`, `mlflow` | PostgreSQL password (`DB_PASSWORD`) |

All `*-secret.yaml` files in the repository contain placeholders only. Never commit real credentials.

---

## 18. Quick Reference

```bash
# Kubeconfig
export KUBECONFIG=deployments/k8s/account/seanmovies-kubeconfig.yaml

# Bootstrap
kubectl apply -f deployments/k8s/namespaces/namespaces.yaml
kubectl apply -f deployments/k8s/rbac/

# Helm deployments
helm upgrade --install recsys-argo-ray   deployments/helm/platform/argo-ray   -n argo       --create-namespace -f values.yaml
helm upgrade --install recsys-mlflow     deployments/helm/infra/mlflow        -n mlflow     --create-namespace -f values.do.yaml
helm upgrade --install recsys-triton     deployments/helm/platform/triton     -n serving    --create-namespace -f values.yaml
helm upgrade --install recsys-ui         deployments/helm/apps/recsys-ui      -n serving    --create-namespace -f values.yaml
helm upgrade --install recsys-monitoring deployments/helm/platform/monitoring -n monitoring --create-namespace -f values.yaml --wait --timeout 5m

# WorkflowTemplates
kubectl apply -f cicd/argo/template/ -n argo

# Flink operator
cd flink-kubernetes-operator
kubectl apply -f helm/flink-kubernetes-operator/crds/
helm upgrade --install flink-kubernetes-operator helm/flink-kubernetes-operator \
  --namespace flink-system --create-namespace --set webhook.create=false
cd ..

# Data infrastructure
kubectl apply -f deployments/k8s/infra/kafka/kafka-nodepool.yaml -n kafka
kubectl apply -f deployments/k8s/infra/kafka/kafka-cluster.yaml  -n kafka
kubectl apply -f deployments/k8s/infra/debezium/  -n infra
kubectl apply -f deployments/k8s/infra/milvus/    -n infra
kubectl apply -f deployments/k8s/infra/elasticsearch/ -n infra
kubectl apply -f deployments/k8s/infra/flink-cluster/flink-service-account.yaml -n infra
kubectl apply -f deployments/k8s/infra/flink-cluster/flink-cdc-job.yaml -n infra

# Port-forwards
kubectl port-forward svc/argo-server                    -n argo       2746:2746
kubectl port-forward svc/recsys-monitoring-grafana      -n monitoring 3000:80
kubectl port-forward svc/recsys-monitoring-prometheus   -n monitoring 9090:9090
kubectl port-forward svc/mlflow                         -n mlflow     5000:5000
kubectl port-forward svc/kafka-connect                  -n infra      8083:8083
kubectl port-forward service/flink-cdc-ui               -n infra      8081:8081
kubectl port-forward service/kafka-ui                   -n kafka      8080:8080

# Training
argo submit cicd/argo/workflow/training-pipeline-submit-workflow.yaml \
  -n argo -p registry=trlocne204 -p skip-tune=true --watch

# Serving
kubectl port-forward svc/serving -n serving 8000:8000
curl "http://localhost:8000/v1/recommend/home?user_id=1"

# GitOps
kubectl apply -f deployments/argocd/root-app.yaml -n argocd

# Incremental refresh CronWorkflow
kubectl apply -f cicd/argo/workflow/serving-update-cronworkflow.yaml -n argo
```
