from prometheus_client import Counter, Histogram, Gauge

# Recommendation Requests
REC_REQUESTS = Counter(
    "rec_requests_total", 
    "Total recommendation requests", 
    ["scenario"]
)

REC_LATENCY = Histogram(
    "rec_latency_seconds", 
    "Recommendation latency in seconds", 
    ["scenario"]
)

# Cache Metrics
CACHE_HIT_TOTAL = Counter(
    "cache_hit_total", 
    "Total cache hits", 
    ["scenario"]
)

CACHE_MISS_TOTAL = Counter(
    "cache_miss_total", 
    "Total cache misses", 
    ["scenario"]
)

# External Service Metrics
FEAST_LATENCY = Histogram(
    "feast_latency_seconds", 
    "Feast feature retrieval latency in seconds",
    ["operation"]
)

FEAST_ERRORS = Counter(
    "feast_errors_total",
    "Total Feast request errors",
    ["operation"]
)

# Model Metrics
MODEL_INFERENCE_LATENCY = Histogram(
    "model_inference_latency_seconds", 
    "Reranker model inference latency in seconds"
)

# User Interaction Metrics
FEEDBACK_TOTAL = Counter(
    "rec_feedback_total",
    "Total user feedback/interactions logged",
    ["event_type"]
)

# System Metrics (Gauge example)
ACTIVE_USERS = Gauge(
    "rec_active_users_total",
    "Total number of users in memory"
)

ACTIVE_ITEMS = Gauge(
    "rec_active_items_total",
    "Total number of items in memory"
)
