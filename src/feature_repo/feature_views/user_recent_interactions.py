from datetime import timedelta

from feast import FeatureView, Field, PushSource
from feast.types import Array, Float64, Int64

from data_sources import interaction_batch_source
from entities import user_entity

interaction_push_source = PushSource(
    name="interaction_push_source",
    batch_source=interaction_batch_source,
)

interaction_fv = FeatureView(
    name="user_recent_interactions",
    entities=[user_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="recent_movie_ids", dtype=Array(Int64)),
        Field(name="recent_ratings", dtype=Array(Float64)),
    ],
    online=True,
    source=interaction_push_source,
)