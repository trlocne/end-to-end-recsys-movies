import os
import boto3
from botocore.exceptions import ClientError
from urllib.parse import urlparse

def parse_s3_path(s3_path):
    parsed = urlparse(s3_path)
    return parsed.netloc, parsed.path.lstrip('/')

def upload_to_s3(local_path, s3_path):
    bucket, key = parse_s3_path(s3_path)
    s3 = boto3.client('s3')
    s3.upload_file(local_path, bucket, key)
    print(f"Uploaded {local_path} to {s3_path}")

def download_from_s3(s3_path, local_path):
    bucket, key = parse_s3_path(s3_path)
    s3 = boto3.client('s3')
    dirname = os.path.dirname(local_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    try:
        s3.download_file(bucket, key, local_path)
    except ClientError as e:
        err = e.response.get("Error", {}) if e.response else {}
        code = err.get("Code", "") or ""
        http_status = (
            e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if e.response
            else None
        )
        missing = (
            code in ("NoSuchKey", "NotFound", "404")
            or http_status == 404
        )
        if missing:
            raise FileNotFoundError(
                f"S3 object does not exist: s3://{bucket}/{key} "
                f"(404/NoSuchKey — fix USER_EMB_S3 / ITEM_EMB_S3 / SERVING_CONFIG_S3 or run training/serving-update to write paths)."
            ) from e
        raise
    print(f"Downloaded {s3_path} to {local_path}")

def download_folder_from_s3(s3_path, local_dir):
    bucket, prefix = parse_s3_path(s3_path)
    if not prefix.endswith('/'):
        prefix += '/'
    
    s3 = boto3.resource('s3')
    bucket_obj = s3.Bucket(bucket)
    
    os.makedirs(local_dir, exist_ok=True)
    
    for obj in bucket_obj.objects.filter(Prefix=prefix):
        relative_path = os.path.relpath(obj.key, prefix)
        if relative_path == '.':
            continue
            
        target_path = os.path.join(local_dir, relative_path)
        dirname = os.path.dirname(target_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        
        if not obj.key.endswith('/'):
            bucket_obj.download_file(obj.key, target_path)
            
    print(f"Downloaded folder {s3_path} to {local_dir}")

def upload_folder_to_s3(local_dir, s3_path):
    bucket, prefix = parse_s3_path(s3_path)
    if prefix and not prefix.endswith('/'):
        prefix += '/'
    
    s3 = boto3.client('s3')
    
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_dir)
            s3_key = os.path.join(prefix, relative_path)
            s3.upload_file(local_file_path, bucket, s3_key)
            
    print(f"Uploaded folder {local_dir} to {s3_path}")
