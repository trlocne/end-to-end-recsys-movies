from datetime import timedelta

from feast import FeatureView, Field
from feast.types import Float64, Int64

from data_sources import item_cdc_push_source, user_cdc_push_source
from entities import movie_entity, user_entity

user_cdc_fv = FeatureView(
    name="user_cdc_features",
    entities=[user_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="avg_rating",        dtype=Float64),
        Field(name="interaction_count", dtype=Int64),
    ],
    online=True,
    source=user_cdc_push_source,
    tags={"tag": "cdc"},
)

item_cdc_fv = FeatureView(
    name="item_cdc_features",
    entities=[movie_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="avg_rating",   dtype=Float64),
        Field(name="rating_count", dtype=Int64),
    ],
    online=True,
    source=item_cdc_push_source,
    tags={"tag": "cdc"},
)
