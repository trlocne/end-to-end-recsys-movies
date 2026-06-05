import os

from feast import FileSource, PushSource
from feast.data_format import ParquetFormat

S3_BUCKET = os.environ.get("S3_BUCKET", "s3://recsys-moivelens/processed")

user_stats_source = FileSource(
    name="user_stats_source",
    path=f"{S3_BUCKET}/feast-features/user_stats/",
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    created_timestamp_column=None,
)

movie_stats_source = FileSource(
    name="movie_stats_source",
    path=f"{S3_BUCKET}/feast-features/movie_stats/",
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    created_timestamp_column=None,
)

user_top_genres_source = FileSource(
    name="user_top_genres_source",
    path=f"{S3_BUCKET}/feast-features/user_top_genres/",
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    created_timestamp_column=None,
)

user_embeddings_source = FileSource(
    name="user_embeddings_source",
    path=f"{S3_BUCKET}/embedding/user-embedding/",
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    created_timestamp_column=None,
)

movie_embeddings_source = FileSource(
    name="movie_embeddings_source",
    path=f"{S3_BUCKET}/embedding/movie-embedding/",
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    created_timestamp_column=None,
)

interaction_batch_source = FileSource(
    name="interaction_batch_source",
    path=f"{S3_BUCKET}/feast-features/user_recent_interactions/",
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    created_timestamp_column=None,
)

tag_source = FileSource(
    name="tag_source",
    path=f"{S3_BUCKET}/feast-features/tags/",
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    created_timestamp_column=None,
)

item_catalog_source = FileSource(
    name="item_catalog_source",
    path=f"{S3_BUCKET}/feast-features/item_catalog/",
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    created_timestamp_column=None,
)

user_catalog_source = FileSource(
    name="user_catalog_source",
    path=f"{S3_BUCKET}/feast-features/user_catalog/",
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    created_timestamp_column=None,
)


user_cdc_push_source = PushSource(
    name="user_cdc_push_source",
    batch_source=user_stats_source,
)

item_cdc_push_source = PushSource(
    name="item_cdc_push_source",
    batch_source=movie_stats_source,
)

user_stats_push_source = PushSource(
    name="user_stats_push_source",
    batch_source=user_stats_source,
)

movie_stats_push_source = PushSource(
    name="movie_stats_push_source",
    batch_source=movie_stats_source,
)

user_genres_push_source = PushSource(
    name="user_genres_push_source",
    batch_source=user_top_genres_source,
)

item_catalog_push_source = PushSource(
    name="item_catalog_push_source",
    batch_source=item_catalog_source,
)

user_catalog_push_source = PushSource(
    name="user_catalog_push_source",
    batch_source=user_catalog_source,
)