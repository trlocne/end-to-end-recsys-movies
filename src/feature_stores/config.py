import os

KAFKA_BOOTSTRAP_SERVERS = os.environ.get(
    "KAFKA_BOOTSTRAP_SERVERS",
    "kafka-kafka-bootstrap.kafka.svc.cluster.local:9092",
)
SCHEMA_REGISTRY_URL = os.environ.get(
    "SCHEMA_REGISTRY_URL",
    "http://schema-reg.infra.svc.cluster.local:8081",
)
KAFKA_TOPICS = [
    "recsys-cdc.public.interactions",
    "recsys-cdc.public.items",
    "recsys-cdc.public.users",
]
KAFKA_GROUP_ID  = "flink-cdc-ingest-group"
FEAST_REPO_PATH = os.environ.get("FEAST_REPO_PATH", "/app/src/feature_repo")
PARALLELISM     = int(os.environ.get("PARALLELISM", "2"))
CHECKPOINT_DIR      = os.environ.get("CHECKPOINT_DIR", "/tmp/flink_cdc_checkpoints")
S3_OFFLINE_BASE     = os.environ.get("S3_OFFLINE_BASE", "s3://recsys-moivelens/processed/feast-features")
