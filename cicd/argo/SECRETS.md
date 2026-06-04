# Secrets Reference — Argo Workflows

Các workflow templates **không** chứa secret values. Secrets phải được inject lúc submit hoặc tạo sẵn trên cluster.

---

## K8s Secrets cần tạo trên cluster

### 1. `aws-creds` — AWS / S3 credentials
```bash
kubectl create secret generic aws-creds \
  --from-literal=AWS_ACCESS_KEY_ID=<key> \
  --from-literal=AWS_SECRET_ACCESS_KEY=<secret> \
  --from-literal=AWS_DEFAULT_REGION=ap-southeast-1 \
  -n argo
```
> Dùng bởi: tất cả job containers (envFrom).

### 2. `postgres-creds` — Postgres password (Aiven)
```bash
kubectl create secret generic postgres-creds \
  --from-literal=DB_PASSWORD=<AVNS_...> \
  -n argo
```
> Dùng bởi: `glue-job-template` trong `training-template` (env.POSTGRES_PASSWORD).

---

## Inject lúc submit — milvus-token, es-api-key

Hai secrets này được pass qua `-p` khi submit, không lưu trong file YAML.

### Training pipeline
```bash
argo submit cicd/argo/workflow/training-pipeline-submit-workflow.yaml \
  -n argo \
  -p milvus-token="<token>" \
  -p es-api-key="<key>" \
  --watch
```

### Serving update (manual)
```bash
argo submit -n argo \
  --from workflowtemplate/serving-update \
  -p milvus-token="<token>" \
  -p es-api-key="<key>"
```

### Jenkins (post-train)
Lưu `MILVUS_TOKEN` và `ES_API_KEY` trong Jenkins Credentials, truyền qua environment block trong `Jenkinsfile.post-train`:
```groovy
withCredentials([
  string(credentialsId: 'milvus-token', variable: 'MILVUS_TOKEN'),
  string(credentialsId: 'es-api-key',   variable: 'ES_API_KEY'),
]) {
  sh 'argo submit ... -p milvus-token=$MILVUS_TOKEN -p es-api-key=$ES_API_KEY'
}
```

---

## Checklist trước khi apply template lên cluster

- [ ] `aws-creds` secret tồn tại trong namespace `argo`
- [ ] `postgres-creds` secret tồn tại trong namespace `argo`
- [ ] `milvus-token` + `es-api-key` được pass qua `-p` hoặc Jenkins Credentials
- [ ] Không commit bất kỳ secret value nào vào Git
