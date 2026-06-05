import argparse
import pandas as pd
import numpy as np
import sys
import os
import time
import logging
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType
from elasticsearch import Elasticsearch, helpers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
if "steps" in current_dir:
    project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
    if project_root not in sys.path:
        sys.path.append(project_root)
from src.utils.s3 import download_from_s3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-path", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--milvus-uri", default="https://in03-a2ab3e1a0468768.serverless.aws-eu-central-1.cloud.zilliz.com")
    parser.add_argument("--milvus-token", default="replace-this-with-your-api-key")
    parser.add_argument("--milvus-collection", default="movie_embeddings")
    parser.add_argument("--milvus-connect-retries", type=int, default=2)
    parser.add_argument("--milvus-retry-delay-sec", type=float, default=5.0)
    parser.add_argument("--es-uri", default="https://9f90eb58646243f7ab374c9abe1df15b.us-central1.gcp.cloud.es.io:443")
    parser.add_argument("--es-api-key", default="YOUR_API_KEY")
    parser.add_argument("--es-index", default="recsys-movies")
    args = parser.parse_args()

    from src.utils.s3 import download_from_s3, download_folder_from_s3
    
    if args.embedding_path.endswith('.parquet') and not args.embedding_path.endswith('/'):
        download_from_s3(args.embedding_path, "item_embeddings.parquet")
    else:
        download_folder_from_s3(args.embedding_path, "item_embeddings.parquet")
        
    if args.metadata_path.endswith('.parquet') and not args.metadata_path.endswith('/'):
        download_from_s3(args.metadata_path, "movies_metadata.parquet")
    else:
        download_folder_from_s3(args.metadata_path, "movies_metadata.parquet")

    embeddings_df = pd.read_parquet("item_embeddings.parquet")
    metadata_df = pd.read_parquet("movies_metadata.parquet")

    logger.info(f"Connecting to Milvus at {args.milvus_uri}...")
    connections.connect("default", uri=args.milvus_uri, token=args.milvus_token)
    
    dim = len(embeddings_df.iloc[0]["embedding"])
    
    if utility.has_collection(args.milvus_collection):
        collection = Collection(args.milvus_collection)
        collection.drop()
    
    fields = [
        FieldSchema(name="movie_id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, "Movie Recommendation Embeddings")
    collection = Collection(args.milvus_collection, schema)
    
    data = [
        embeddings_df["movie_id"].tolist(),
        embeddings_df["embedding"].tolist()
    ]
    collection.insert(data)
    collection.flush()
    
    index_params = {
        "metric_type": "IP", 
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    logger.info("Milvus indexing complete.")

    logger.info(f"Connecting to Elasticsearch at {args.es_uri}...")
    es = Elasticsearch(args.es_uri, api_key=args.es_api_key)
    
    actions = [
        {
            "_index": args.es_index,
            "_id": row["movie_id"],
            "_source": {
                "title": str(row.get("title", "")),
                "genres": str(row.get("genres", "")).replace("|", " "),
                "tags": str(row.get("tag", "")),
            }
        }
        for _, row in metadata_df.iterrows()
    ]
    
    helpers.bulk(es, actions)
    logger.info("Elasticsearch indexing complete.")

if __name__ == "__main__":
    main()
