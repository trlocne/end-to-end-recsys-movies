import json
import logging

from pyflink.common import Duration, Types, WatermarkStrategy
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream import RuntimeExecutionMode, StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import (
    KafkaOffsetResetStrategy,
    KafkaOffsetsInitializer,
    KafkaSource,
)


from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_GROUP_ID,
    KAFKA_TOPICS,
    PARALLELISM,
    S3_OFFLINE_BASE,
)
from transforms import (
    CDCParser,
    CDCTimestampAssigner,
    FeastPushFunction,
    InteractionFilter,
    ItemCatalogFromCdc,
    ItemFilter,
    ItemRatingReducer,
    ItemToFeature,
    RecentInteractionsReducer,
    S3BatchSink,
    UserCatalogFromCdc,
    UserFilter,
    UserRatingReducer,
    UserRecentToFeature,
    UserToFeature,
    interaction_to_item_acc,
    interaction_to_recent_acc,
    interaction_to_user_acc,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("PyFlinkCDCFeast")

def main():
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_runtime_mode(RuntimeExecutionMode.STREAMING)
    env.set_parallelism(PARALLELISM)
    env.enable_checkpointing(60_000)
    kafka_source = (
        KafkaSource.builder()
        .set_bootstrap_servers(KAFKA_BOOTSTRAP_SERVERS)
        .set_topics(*KAFKA_TOPICS)
        .set_group_id(KAFKA_GROUP_ID)
        .set_starting_offsets(KafkaOffsetsInitializer.committed_offsets(KafkaOffsetResetStrategy.LATEST))
        .set_value_only_deserializer(SimpleStringSchema("ISO-8859-1"))
        .build()
    )

    raw_stream = env.from_source(
        kafka_source,
        WatermarkStrategy.no_watermarks(),
        "Kafka CDC Source",
    )

    parsed_stream = raw_stream.flat_map(
        CDCParser(),
        output_type=Types.TUPLE([
            Types.STRING(),  # topic_type
            Types.STRING(),  # entity_id
            Types.STRING(),  # payload JSON
            Types.LONG(),    # ts_ms
        ]),
    ).name("parse_cdc")

    interaction_stream = parsed_stream.filter(InteractionFilter()).name("filter_interactions")
    user_stream        = parsed_stream.filter(UserFilter()).name("filter_users")
    item_stream        = parsed_stream.filter(ItemFilter()).name("filter_items")

    watermarked_interactions = (
        interaction_stream
        .assign_timestamps_and_watermarks(
            WatermarkStrategy
            .for_bounded_out_of_orderness(Duration.of_seconds(10))
            .with_idleness(Duration.of_seconds(30))
            .with_timestamp_assigner(CDCTimestampAssigner())
        )
        .name("watermark_interactions")
    )

    user_features_stream = (
        watermarked_interactions
        .map(interaction_to_user_acc, output_type=Types.TUPLE([
            Types.STRING(), Types.STRING(), Types.STRING(), Types.LONG()
        ]))
        .filter(lambda x: bool(x[1]))
        .key_by(lambda x: x[1])               # key = user_id
        .reduce(UserRatingReducer())
        .key_by(lambda x: x[1])               # re-key để giữ cùng subtask
        .map(UserToFeature(), output_type=Types.STRING())
        .name("user_feature_aggregation")
    )

    item_features_stream = (
        watermarked_interactions
        .map(interaction_to_item_acc, output_type=Types.TUPLE([
            Types.STRING(), Types.STRING(), Types.STRING(), Types.LONG()
        ]))
        .filter(lambda x: bool(x[1]))         # bỏ item_id rỗng
        .key_by(lambda x: x[1])               # key = item_id
        .reduce(ItemRatingReducer())
        .key_by(lambda x: x[1])               # re-key để giữ cùng subtask
        .map(ItemToFeature(), output_type=Types.STRING())
        .name("item_feature_aggregation")
    )
    recent_features_stream = (
        watermarked_interactions
        .map(interaction_to_recent_acc, output_type=Types.TUPLE([
            Types.STRING(), Types.STRING(), Types.STRING(), Types.LONG()
        ]))
        .filter(lambda x: bool(x[1]))
        .key_by(lambda x: x[1])                       # key = user_id
        .process(RecentInteractionsReducer(), output_type=Types.TUPLE([
            Types.STRING(), Types.STRING(), Types.STRING(), Types.LONG()
        ]))
        .map(UserRecentToFeature(), output_type=Types.STRING())
        .name("recent_interactions_aggregation")
    )

    item_catalog_stream = (
        item_stream
        .map(ItemCatalogFromCdc(), output_type=Types.STRING())
        .name("item_catalog_from_cdc")
    )
    user_catalog_stream = (
        user_stream
        .map(UserCatalogFromCdc(), output_type=Types.STRING())
        .name("user_catalog_from_cdc")
    )

    user_push_stream = (
        user_features_stream
        .map(FeastPushFunction("user", "user_cdc_push_source"), output_type=Types.STRING())
        .name("feast_push_user")
    )
    user_push_stream.name("sink_user_online")

    item_push_stream = (
        item_features_stream
        .map(FeastPushFunction("item", "item_cdc_push_source"), output_type=Types.STRING())
        .name("feast_push_item")
    )
    item_push_stream.name("sink_item_online")

    recent_push_stream = (
        recent_features_stream
        .map(FeastPushFunction("recent", "interaction_push_source"), output_type=Types.STRING())
        .name("feast_push_recent")
    )
    recent_push_stream.name("sink_recent_online")

    item_catalog_push = (
        item_catalog_stream
        .map(FeastPushFunction("item_catalog", "item_catalog_push_source"), output_type=Types.STRING())
        .name("feast_push_item_catalog")
    )
    item_catalog_push.name("sink_item_catalog_online")

    user_catalog_push = (
        user_catalog_stream
        .map(FeastPushFunction("user_catalog", "user_catalog_push_source"), output_type=Types.STRING())
        .name("feast_push_user_catalog")
    )
    user_catalog_push.name("sink_user_catalog_online")

    # ── S3 offline sinks (for Feast offline store / get_historical_features) ──
    user_features_stream.map(
        S3BatchSink(f"{S3_OFFLINE_BASE}/user_stats/"), output_type=Types.STRING()
    ).name("sink_user_offline_s3")

    item_features_stream.map(
        S3BatchSink(f"{S3_OFFLINE_BASE}/movie_stats/"), output_type=Types.STRING()
    ).name("sink_item_offline_s3")

    recent_features_stream.map(
        S3BatchSink(f"{S3_OFFLINE_BASE}/user_recent_interactions/"), output_type=Types.STRING()
    ).name("sink_recent_offline_s3")

    item_catalog_stream.map(
        S3BatchSink(f"{S3_OFFLINE_BASE}/item_catalog/"), output_type=Types.STRING()
    ).name("sink_item_catalog_offline_s3")

    user_catalog_stream.map(
        S3BatchSink(f"{S3_OFFLINE_BASE}/user_catalog/"), output_type=Types.STRING()
    ).name("sink_user_catalog_offline_s3")

    env.execute("CDC Feature Engineering → Feast Pipeline")


if __name__ == "__main__":
    main()


