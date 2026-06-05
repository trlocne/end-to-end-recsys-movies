import argparse
import logging
import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from feast import FeatureStore

current_dir = os.path.dirname(os.path.abspath(__file__))
if "steps" in current_dir:
    project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
    if project_root not in sys.path:
        sys.path.append(project_root)

from src.data.features import FeatureStore as LegacyFeatureStore
from src.utils.s3 import download_from_s3, upload_to_s3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("incremental_feature_extractor")


def load_config(config_path: str) -> Dict:
    """Load configuration from S3 or local path."""
    if config_path.startswith("s3://"):
        local_config = "/tmp/incremental_config.yaml"
        download_from_s3(config_path, local_config)
        config_path = local_config
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from {config_path}")
    return config


def load_features_from_feast(feast_config: Dict, user_ids: List[int], item_ids: List[int]) -> Dict:
    """Load features from Feast for incremental processing. CDC views override batch where available."""
    try:
        feast_repo_path = feast_config.get('repo_path', '/app/src/feature_repo')
        fs = FeatureStore(repo_path=feast_repo_path)

        user_entity_rows = [{"user_id": uid} for uid in user_ids]
        item_entity_rows = [{"movie_id": iid} for iid in item_ids]

        user_features = fs.get_online_features(
            features=["user_stats:avg_rating", "user_stats:interaction_count", "user_stats:rating_std"],
            entity_rows=user_entity_rows,
        ).to_dict()

        item_features = fs.get_online_features(
            features=["movie_stats:avg_rating", "movie_stats:popularity", "movie_stats:rating_std"],
            entity_rows=item_entity_rows,
        ).to_dict()

        try:
            user_cdc = fs.get_online_features(
                features=["user_cdc_features:avg_rating", "user_cdc_features:interaction_count"],
                entity_rows=user_entity_rows,
            ).to_dict()
            if user_cdc.get("user_cdc_features__avg_rating"):
                user_features["user_stats__avg_rating"] = user_cdc["user_cdc_features__avg_rating"]
            if user_cdc.get("user_cdc_features__interaction_count"):
                user_features["user_stats__interaction_count"] = user_cdc["user_cdc_features__interaction_count"]

            item_cdc = fs.get_online_features(
                features=["item_cdc_features:avg_rating", "item_cdc_features:rating_count"],
                entity_rows=item_entity_rows,
            ).to_dict()
            if item_cdc.get("item_cdc_features__avg_rating"):
                item_features["movie_stats__avg_rating"] = item_cdc["item_cdc_features__avg_rating"]
            if item_cdc.get("item_cdc_features__rating_count"):
                item_features["movie_stats__popularity"] = item_cdc["item_cdc_features__rating_count"]
        except Exception as cdc_e:
            logger.warning(f"CDC feature override skipped (non-critical): {cdc_e}")

        logger.info(f"Loaded features from Feast: {len(user_ids)} users, {len(item_ids)} items")
        return {
            'user_features': user_features,
            'item_features': item_features
        }
    except Exception as e:
        logger.error(f"Failed to load features from Feast: {e}")
        raise


def get_new_entities_from_feast(feast_config: Dict, since_timestamp: str) -> Dict:
    """Get new users/items that had updates since timestamp."""
    try:
        feast_repo_path = feast_config.get('repo_path', '/app/src/feature_repo')
        fs = FeatureStore(repo_path=feast_repo_path)
        
        # Get historical features to find new entities
        since_dt = datetime.fromisoformat(since_timestamp) if since_timestamp else datetime.now() - timedelta(hours=2)
        
        # Query Feast for features updated since timestamp
        # Note: This would require Feast to support time-travel queries properly
        # For now, return empty and rely on downstream logic
        
        logger.info(f"Querying Feast for entities since {since_dt}")
        return {'new_user_ids': [], 'new_item_ids': []}
        
    except Exception as e:
        logger.error(f"Failed to query Feast for new entities: {e}")
        return {'new_user_ids': [], 'new_item_ids': []}


def load_last_timestamp(timestamp_path: str, pg_config: Optional[Dict] = None) -> Optional[str]:
    """Load last processed timestamp from PostgreSQL or S3/local file."""
    if pg_config:
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=pg_config['host'],
                port=pg_config['port'],
                database=pg_config['database'],
                user=pg_config['user'],
                password=pg_config['password'],
                sslmode='require'
            )
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp FROM incremental_state WHERE component = %s", ('feature_extractor',))
            result = cursor.fetchone()
            conn.close()
            if result:
                return result[0]
            return None
        except Exception as e:
            logger.warning(f"Could not load last timestamp from PostgreSQL: {e}")
            return None
    elif timestamp_path.startswith("s3://"):
        try:
            local_timestamp = "/tmp/last_processed_timestamp.txt"
            download_from_s3(timestamp_path, local_timestamp)
            with open(local_timestamp, 'r') as f:
                return f.read().strip()
        except Exception as e:
            logger.warning(f"Could not load last timestamp: {e}")
            return None
    else:
        try:
            with open(timestamp_path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            logger.warning(f"Could not load last timestamp: {e}")
            return None


def save_last_timestamp(timestamp_path: str, timestamp: str, pg_config: Optional[Dict] = None):
    """Save last processed timestamp to PostgreSQL or S3/local file."""
    if pg_config:
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=pg_config['host'],
                port=pg_config['port'],
                database=pg_config['database'],
                user=pg_config['user'],
                password=pg_config['password'],
                sslmode='require'
            )
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO incremental_state (component, timestamp, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (component) 
                DO UPDATE SET timestamp = %s, updated_at = NOW()
            """, ('feature_extractor', timestamp, timestamp))
            conn.commit()
            conn.close()
            logger.info(f"Saved timestamp to PostgreSQL: {timestamp}")
        except Exception as e:
            logger.warning(f"Could not save timestamp to PostgreSQL: {e}")
    elif timestamp_path.startswith("s3://"):
        local_timestamp = "/tmp/last_processed_timestamp.txt"
        with open(local_timestamp, 'w') as f:
            f.write(timestamp)
        upload_to_s3(local_timestamp, timestamp_path)
    else:
        with open(timestamp_path, 'w') as f:
            f.write(timestamp)


def consume_from_kafka(kafka_config: Dict, max_messages: int = 1000) -> pd.DataFrame:
    """Consume new interactions from Kafka topic."""
    from kafka import KafkaConsumer
    consumer = KafkaConsumer(
        kafka_config['topic'],
        bootstrap_servers=kafka_config['bootstrap_servers'],
        security_protocol=kafka_config.get('security_protocol', 'SSL'),
        ssl_cafile=kafka_config.get('ssl_cafile'),
        ssl_certfile=kafka_config.get('ssl_certfile'),
        ssl_keyfile=kafka_config.get('ssl_keyfile'),
        value_deserializer=lambda m: m.decode('utf-8'),
        auto_offset_reset='latest',
        enable_auto_commit=False,
        group_id=kafka_config.get('consumer_group', 'incremental-feature-extractor')
    )
    
    interactions = []
    message_count = 0
    
    logger.info(f"Starting to consume from Kafka topic: {kafka_config['topic']}")
    
    try:
        for message in consumer:
            try:
                # Parse JSON message
                import json
                data = json.loads(message.value)
                interactions.append(data)
                message_count += 1
                
                if message_count >= max_messages:
                    logger.info(f"Reached max messages limit: {max_messages}")
                    break
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse message: {e}")
                continue
    except KeyboardInterrupt:
        logger.info("Consumer interrupted")
    finally:
        consumer.close()
    
    if interactions:
        df = pd.DataFrame(interactions)
        logger.info(f"Consumed {len(df)} interactions from Kafka")
        return df
    else:
        logger.warning("No messages consumed from Kafka")
        return pd.DataFrame()


def load_feast_features(config: Dict, new_user_ids: List[int], new_item_ids: List[int]) -> Dict:
    """Load user and item features from Feast for new users/items."""
    if not config.get('feast', {}).get('enabled', False):
        logger.info("Feast disabled, skipping feature loading")
        return {'user_features': {}, 'item_features': {}}
    
    feast_repo_path = config['feast']['repo_path']
    
    # Download Feast config if needed
    if feast_repo_path.startswith("s3://"):
        import tempfile
        from src.utils.s3 import download_from_s3
        
        tmp_dir = tempfile.mkdtemp()
        logger.info(f"Downloading Feast config from S3 to {tmp_dir}")
        local_config_path = os.path.join(tmp_dir, "feature_store.yaml")
        
        try:
            download_from_s3(os.path.join(feast_repo_path, "feature_store.yaml"), local_config_path)
            feast_repo_path = tmp_dir
        except Exception as e:
            logger.error(f"Failed to download Feast config from S3: {e}")
            return {'user_features': {}, 'item_features': {}}
    
    try:
        # Create user/item maps for Feast
        users_map = pd.DataFrame({'user_id': new_user_ids, 'user_idx': range(len(new_user_ids))})
        items_map = pd.DataFrame({'item_id': new_item_ids, 'item_idx': range(len(new_item_ids))})
        
        logger.info("Loading features from Feast...")
        feature_store = FeatureStore(feast_repo_path=feast_repo_path)
        feature_store.load(
            items_map=items_map,
            users_map=users_map,
        )
        
        user_features = feature_store.get_user_features()
        item_features = feature_store.get_item_features()
        
        logger.info(f"Loaded Feast features for {len(user_features)} users, {len(item_features)} items")
        return {'user_features': user_features, 'item_features': item_features}
        
    except Exception as e:
        logger.warning(f"Failed to load Feast features: {e}")
        return {'user_features': {}, 'item_features': {}}


def extract_incremental_features(
    new_interactions: pd.DataFrame,
    config: Dict,
    static_features: Optional[Dict] = None
) -> Dict:
    """Extract features for new users/items."""
    
    # Get unique new users and items
    new_user_ids = new_interactions['user_id'].unique().tolist()
    new_item_ids = new_interactions['item_id'].unique().tolist()
    
    logger.info(f"Processing {len(new_user_ids)} new users, {len(new_item_ids)} new items")
    
    feast_features = load_feast_features(config, new_user_ids, new_item_ids)
    
    from src.data.features import build_bipartite_graph
    
    user_id_to_idx = {uid: idx for idx, uid in enumerate(new_user_ids)}
    item_id_to_idx = {iid: idx for idx, iid in enumerate(new_item_ids)}
    
    new_interactions['user_idx'] = new_interactions['user_id'].map(user_id_to_idx)
    new_interactions['item_idx'] = new_interactions['item_id'].map(item_id_to_idx)
    
    edge_index, edge_weight = build_bipartite_graph(
        new_interactions,
        num_users=len(new_user_ids),
        num_items=len(new_item_ids)
    )
    
    edge_index_sparse = torch.sparse_coo_tensor(
        edge_index,
        edge_weight,
        size=(len(new_user_ids), len(new_item_ids))
    )
    
    logger.info(f"Created sparse edge_index with {edge_index_sparse.nnz()} non-zero elements")
    
    delta_features = {
        'edge_index_delta': edge_index_sparse,
        'new_user_ids': new_user_ids,
        'new_item_ids': new_item_ids,
        'new_interactions': new_interactions,
        'feast_user_features': feast_features['user_features'],
        'feast_item_features': feast_features['item_features'],
        'timestamp': datetime.now().isoformat(),
        'num_new_users': len(new_user_ids),
        'num_new_items': len(new_item_ids)
    }
    
    return delta_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True, help="Path to config file (S3 or local)")
    parser.add_argument("--delta-output-path", required=True, help="Output path for delta features")
    parser.add_argument("--use-postgres", action="store_true", help="Use PostgreSQL for timestamp storage")
    args = parser.parse_args()
    
    config = load_config(args.config_path)
    
    # Check if incremental is enabled
    if not config.get('incremental', {}).get('enabled', False):
        logger.info("Incremental mode disabled, exiting")
        return
    
    # Get PostgreSQL config if enabled
    pg_config = None
    if args.use_postgres or config.get('postgres', {}).get('enabled', False):
        pg_config = config.get('postgres', {})
        logger.info("Using PostgreSQL for timestamp storage")
    
    last_timestamp = load_last_timestamp(
        config['feature_paths'].get('last_timestamp', ''),
        pg_config=pg_config
    )
    current_timestamp = datetime.now().isoformat()
    
    logger.info(f"Last processed: {last_timestamp}")
    logger.info(f"Current timestamp: {current_timestamp}")
    
    # Get new entities from Feast (users/items with updates since last run)
    feast_config = config.get('feast', {})
    new_entities = get_new_entities_from_feast(feast_config, last_timestamp)
    
    new_user_ids = new_entities['new_user_ids']
    new_item_ids = new_entities['new_item_ids']
    
    logger.info(f"Found {len(new_user_ids)} new users, {len(new_item_ids)} new items")
    
    # Check minimum thresholds
    min_users = config['incremental'].get('min_new_users', 10)
    min_items = config['incremental'].get('min_new_items', 10)
    
    if len(new_user_ids) < min_users and len(new_item_ids) < min_items:
        logger.info(f"Below threshold (users: {len(new_user_ids)} < {min_users}, items: {len(new_item_ids)} < {min_items}), skipping")
        return
    
    feast_features = load_features_from_feast(feast_config, new_user_ids, new_item_ids)

    from src.data.features import build_bipartite_graph
    
    # Create dummy interactions for delta graph (in real implementation, get from Feast)
    # For now, use placeholder data
    dummy_interactions = pd.DataFrame({
        'user_id': new_user_ids[:min(10, len(new_user_ids))] * min(5, len(new_item_ids)),
        'item_id': new_item_ids[:min(5, len(new_item_ids))] * min(10, len(new_user_ids)),
        'rating': 4.0,
        'timestamp': current_timestamp
    })
    
    user_id_to_idx = {uid: idx for idx, uid in enumerate(new_user_ids)}
    item_id_to_idx = {iid: idx for idx, iid in enumerate(new_item_ids)}
    
    dummy_interactions['user_idx'] = dummy_interactions['user_id'].map(user_id_to_idx)
    dummy_interactions['item_idx'] = dummy_interactions['item_id'].map(item_id_to_idx)
    
    edge_index, edge_weight = build_bipartite_graph(
        dummy_interactions,
        num_users=len(new_user_ids),
        num_items=len(new_item_ids)
    )
    
    edge_index_sparse = torch.sparse_coo_tensor(
        edge_index,
        edge_weight,
        size=(len(new_user_ids), len(new_item_ids))
    )
    
    logger.info(f"Created sparse edge_index with {edge_index_sparse.nnz()} non-zero elements")
    
    delta_features = {
        'edge_index_delta': edge_index_sparse,
        'new_user_ids': new_user_ids,
        'new_item_ids': new_item_ids,
        'feast_user_features': feast_features['user_features'],
        'feast_item_features': feast_features['item_features'],
        'timestamp': current_timestamp,
        'num_new_users': len(new_user_ids),
        'num_new_items': len(new_item_ids)
    }
    
    local_delta_path = "/tmp/features_delta.pt"
    torch.save(delta_features, local_delta_path)
    
    if args.delta_output_path.startswith("s3://"):
        upload_to_s3(local_delta_path, args.delta_output_path)
    else:
        os.makedirs(os.path.dirname(args.delta_output_path), exist_ok=True)
        import shutil
        shutil.copy(local_delta_path, args.delta_output_path)
    
    logger.info(f"Saved delta features to {args.delta_output_path}")
    
    # Update last timestamp
    save_last_timestamp(
        config['feature_paths'].get('last_timestamp', ''),
        current_timestamp,
        pg_config=pg_config
    )
    logger.info(f"Updated last timestamp to {current_timestamp}")
    
    logger.info("Incremental feature extraction completed successfully")


if __name__ == "__main__":
    main()
