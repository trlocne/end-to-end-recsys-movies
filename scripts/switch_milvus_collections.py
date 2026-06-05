#!/usr/bin/env python3
"""
Switch Milvus collections for blue-green deployment.
This script updates the serving configuration to point to new collections
and provides rollback capability.
"""
import sys
import os
import argparse
from datetime import datetime
from pymilvus import connections, Collection, utility
import yaml


def switch_collections(milvus_uri, milvus_token, movie_collection, user_collection):
    """
    Switch serving to use new Milvus collections (blue-green deployment).
    
    Args:
        milvus_uri: Milvus server URI
        milvus_token: Milvus authentication token
        movie_collection: New movie embeddings collection name
        user_collection: New user embeddings collection name
    """
    try:
        connections.connect("default", uri=milvus_uri, token=milvus_token)
        print(f"Connected to Milvus at {milvus_uri}")
        for col_name in [movie_collection, user_collection]:
            try:
                col = Collection(col_name)
                col.load()
                num_entities = col.num_entities
                print(f"✓ Collection {col_name}: {num_entities} entities")
                if num_entities == 0:
                    raise ValueError(f"Collection {col_name} is empty")
            except Exception as e:
                print(f"✗ Collection {col_name} verification failed: {e}")
                raise
        
        config_path = "/teamspace/studios/this_studio/configs/serving_config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{config_path}.backup_{timestamp}"
            with open(backup_path, 'w') as f:
                yaml.dump(config, f)
            print(f"✓ Backed up config to {backup_path}")
            
            old_movie_col = config.get('milvus', {}).get('movie_collection', 'movie_embeddings')
            old_user_col = config.get('milvus', {}).get('user_collection', 'user_embeddings')
            
            config['milvus']['movie_collection'] = movie_collection
            config['milvus']['user_collection'] = user_collection
            
            config['milvus']['rollback'] = {
                'previous_movie_collection': old_movie_col,
                'previous_user_collection': old_user_col,
                'switched_at': datetime.now().isoformat()
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            print(f"✓ Updated serving config:")
            print(f"  Movie collection: {old_movie_col} → {movie_collection}")
            print(f"  User collection: {old_user_col} → {user_collection}")
            print(f"✓ Rollback info saved in config")
            
        except FileNotFoundError:
            print(f"⚠ Config file not found at {config_path}")
            print("Creating new config file...")
            config = {
                'milvus': {
                    'movie_collection': movie_collection,
                    'user_collection': user_collection,
                    'rollback': {
                        'previous_movie_collection': 'movie_embeddings',
                        'previous_user_collection': 'user_embeddings',
                        'switched_at': datetime.now().isoformat()
                    }
                }
            }
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            print(f"✓ Created new config at {config_path}")
        
        print("\n✓ Collection switch completed successfully")
        print(f"New movie collection: {movie_collection}")
        print(f"New user collection: {user_collection}")
        
    except Exception as e:
        print(f"✗ Collection switch failed: {e}")
        raise


def rollback_collections(milvus_uri, milvus_token, config_path=None):
    """
    Rollback to previous Milvus collections.
    
    Args:
        milvus_uri: Milvus server URI
        milvus_token: Milvus authentication token
        config_path: Path to serving config file
    """
    if config_path is None:
        config_path = "/teamspace/studios/this_studio/configs/serving_config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        rollback_info = config.get('milvus', {}).get('rollback', {})
        
        if not rollback_info:
            print("✗ No rollback information found in config")
            return False
        
        old_movie_col = rollback_info.get('previous_movie_collection')
        old_user_col = rollback_info.get('previous_user_collection')
        
        if not old_movie_col or not old_user_col:
            print("✗ Previous collection names not found in rollback info")
            return False
        
        # Switch back
        config['milvus']['movie_collection'] = old_movie_col
        config['milvus']['user_collection'] = old_user_col
        
        # Update rollback info
        current_movie = config['milvus']['movie_collection']
        current_user = config['milvus']['user_collection']
        config['milvus']['rollback'] = {
            'previous_movie_collection': current_movie,
            'previous_user_collection': current_user,
            'switched_at': datetime.now().isoformat(),
            'rollback_at': datetime.now().isoformat()
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        print(f"✓ Rolled back to:")
        print(f"  Movie collection: {old_movie_col}")
        print(f"  User collection: {old_user_col}")
        return True
        
    except Exception as e:
        print(f"✗ Rollback failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Switch Milvus collections for blue-green deployment")
    parser.add_argument("--milvus-uri", required=True, help="Milvus server URI")
    parser.add_argument("--milvus-token", required=True, help="Milvus authentication token")
    parser.add_argument("--movie-collection", required=True, help="New movie embeddings collection name")
    parser.add_argument("--user-collection", required=True, help="New user embeddings collection name")
    parser.add_argument("--rollback", action="store_true", help="Rollback to previous collections")
    parser.add_argument("--config-path", help="Path to serving config file")
    
    args = parser.parse_args()
    
    if args.rollback:
        success = rollback_collections(
            milvus_uri=args.milvus_uri,
            milvus_token=args.milvus_token,
            config_path=args.config_path
        )
        sys.exit(0 if success else 1)
    else:
        switch_collections(
            milvus_uri=args.milvus_uri,
            milvus_token=args.milvus_token,
            movie_collection=args.movie_collection,
            user_collection=args.user_collection
        )
