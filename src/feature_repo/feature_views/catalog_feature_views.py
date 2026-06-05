"""Catalog feature views from Postgres items/users CDC (Flink → S3 / Redis)."""
from datetime import timedelta

from feast import FeatureView, Field
from feast.types import Array, Int64, String

from data_sources import item_catalog_push_source, user_catalog_push_source
from entities import movie_entity, user_entity

item_catalog_fv = FeatureView(
    name="item_catalog",
    entities=[movie_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="title", dtype=String),
        Field(name="genres", dtype=Array(String)),
        Field(name="tags", dtype=Array(String)),
        Field(name="release_year", dtype=Int64),
    ],
    online=True,
    source=item_catalog_push_source,
    tags={"tag": "catalog", "streaming": "true"},
)

user_catalog_fv = FeatureView(
    name="user_catalog",
    entities=[user_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="metadata_json", dtype=String),
    ],
    online=True,
    source=user_catalog_push_source,
    tags={"tag": "catalog", "streaming": "true"},
)
