#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument(
        "--source-root",
        default="/root/SkyReels-V2/result",
        help="Local result root to mirror into S3.",
    )
    parser.add_argument(
        "--only-dir",
        action="append",
        default=[],
        help="Optional result subdirectory name(s) under source-root to sync.",
    )
    return parser.parse_args()


def iter_files(source_root: Path, only_dirs: list[str]):
    if only_dirs:
        roots = [source_root / item for item in only_dirs]
    else:
        roots = [source_root]

    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                yield path


def should_upload(s3, bucket: str, key: str, local_size: int) -> bool:
    try:
        response = s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return True
        raise
    return int(response["ContentLength"]) != local_size


def main():
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))

    uploaded = 0
    skipped = 0

    for path in iter_files(source_root, args.only_dir):
        rel = path.relative_to(source_root.parent)
        key = f"{args.prefix.rstrip('/')}/{rel.as_posix()}"
        size = path.stat().st_size
        if should_upload(s3, args.bucket, key, size):
            print(f"UPLOAD {path} -> s3://{args.bucket}/{key}")
            s3.upload_file(str(path), args.bucket, key)
            uploaded += 1
        else:
            print(f"SKIP   {path}")
            skipped += 1

    print(f"DONE uploaded={uploaded} skipped={skipped}")


if __name__ == "__main__":
    main()
