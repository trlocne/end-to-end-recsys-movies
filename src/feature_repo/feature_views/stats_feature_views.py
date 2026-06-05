"""Batch + push stats and user genre feature views (used by serving + data_prep)."""
from datetime import timedelta

from feast import FeatureView, Field
from feast.types import Array, Float64, Int64, String

from data_sources import (
    movie_stats_push_source,
    user_genres_push_source,
    user_stats_push_source,
)
from entities import movie_entity, user_entity

user_stats_fv = FeatureView(
    name="user_stats",
    entities=[user_entity],
    ttl=timedelta(days=60),
    schema=[
        Field(name="avg_rating", dtype=Float64),
        Field(name="interaction_count", dtype=Int64),
        Field(name="rating_std", dtype=Float64),
    ],
    online=True,
    source=user_stats_push_source,
    tags={"tag": "stats", "streaming": "true"},
)

movie_stats_fv = FeatureView(
    name="movie_stats",
    entities=[movie_entity],
    ttl=timedelta(days=60),
    schema=[
        Field(name="year", dtype=Int64),
        Field(name="popularity", dtype=Int64),
        Field(name="avg_rating", dtype=Float64),
        Field(name="rating_std", dtype=Float64),
    ],
    online=True,
    source=movie_stats_push_source,
    tags={"tag": "stats", "streaming": "true"},
)

user_genres_fv = FeatureView(
    name="user_genres",
    entities=[user_entity],
    ttl=timedelta(days=60),
    schema=[
        Field(name="user_id", dtype=Int64),
        Field(name="top_genres", dtype=Array(String)),
        Field(name="genre_vector", dtype=Array(Int64)),
    ],
    online=True,
    source=user_genres_push_source,
    tags={"tag": "stats", "streaming": "true"},
)
