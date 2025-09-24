#!/usr/bin/env python3
import argparse
import concurrent.futures
import os
import pathlib
import sys

import boto3
from botocore.exceptions import ClientError

# Workaround for MissingContentLength on some S3-compatible services
os.environ.setdefault("AWS_REQUEST_CHECKSUM_CALCULATION", "when_required")
os.environ.setdefault("AWS_RESPONSE_CHECKSUM_VALIDATION", "when_required")


def upload_file(s3, local_path, bucket, key):
    try:
        s3.upload_file(local_path, bucket, key)
        print(f"✔ {key}")
    except ClientError as e:
        print(f"✘ {key} failed: {e}")


def upload_folder(local_folder, bucket, endpoint_url, prefix="", workers=10):
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    # Collect all (local_path, s3_key) pairs
    to_upload = []
    for root, _, files in os.walk(local_folder):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, local_folder)
            key = os.path.join(prefix, rel_path).replace("\\", "/")
            to_upload.append((local_path, key))

    # Upload in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(upload_file, s3, lp, bucket, key) for lp, key in to_upload
        ]
        # wait for all to finish (errors are already printed in upload_file)
        concurrent.futures.wait(futures)


def main():
    """
    Upload a folder to an S3-compatible bucket in parallel.

    Usage:
        python sync_to_s3.py <local_folder> <s3_path>
    Arguments:
        local_folder: Path to the local folder to upload.
        s3_path: S3 path formatted s3://bucket/prefix/<name>. The local folder name is inserted in <name>

    Environment Variables:
        S3_ENDPOINT_URL: The endpoint URL of the S3-compatible service.
        AWS_ACCESS_KEY_ID: Your AWS access key ID.
        AWS_SECRET_ACCESS_KEY: Your AWS secret access key.

    Example:
        python sync_to_s3.py /path/to/local/folder s3://mybucket/prefix/
    """
    endpoint_url = os.getenv("S3_ENDPOINT_URL")
    if not endpoint_url:
        sys.exit("Error: S3_ENDPOINT_URL not set")

    p = argparse.ArgumentParser(
        description="Upload a folder to an S3-compatible bucket in parallel"
    )
    p.add_argument("local_folder", help="Path to local folder")
    p.add_argument("s3_path", help="Target S3 path (s3://bucket/prefix/)")
    p.add_argument(
        "--workers",
        "-w",
        type=int,
        default=10,
        help="Number of parallel upload threads (default: 10)",
    )
    args = p.parse_args()

    if not os.path.isdir(args.local_folder):
        sys.exit(f"Error: {args.local_folder} is not a directory")

    # Parse the S3 path
    if not args.s3_path.startswith("s3://"):
        sys.exit("Error: s3_path must start with 's3://'")
    s3_path = args.s3_path[5:]  # Remove 's3://'
    if "/" not in s3_path:
        sys.exit("Error: s3_path must contain a bucket name and a prefix")
    bucket, prefix = s3_path.split("/", 1)
    prefix = prefix.rstrip("/")  # Remove trailing slash if present
    if not prefix.endswith("/"):
        prefix += "/"  # Ensure prefix ends with a slash
    name = pathlib.Path(args.local_folder).name
    prefix = prefix + name + "/"

    upload_folder(
        args.local_folder, bucket, endpoint_url, prefix=prefix, workers=args.workers
    )


if __name__ == "__main__":
    # test
    main()
