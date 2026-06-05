from datetime import timedelta
from feast import FeatureView, Field
from feast.types import Float32, Array
from entities import user_entity, movie_entity
from data_sources import user_embeddings_source, movie_embeddings_source

user_embedding_fv = FeatureView(
    name="user_embeddings",
    entities=[user_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(name="embedding", dtype=Array(Float32)),
    ],
    online=True,
    source=user_embeddings_source,
    tags={"tag": "embedding"}
)

movie_embedding_fv = FeatureView(
    name="movie_embeddings",
    entities=[movie_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(name="embedding", dtype=Array(Float32)),
    ],
    online=True,
    source=movie_embeddings_source,
    tags={"tag": "embedding"}
)