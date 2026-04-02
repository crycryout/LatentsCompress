#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError


EXPORTS = (
    (Path("/workspace/video_bench/codec_compare"), "codec_compare"),
    (Path("/root/LatentsCompress/examples/vbench_codec/skyreels_long_lossless"), "repo_examples/skyreels_long_lossless"),
    (Path("/root/LatentsCompress/examples/vbench_codec/wan64_xor"), "repo_examples/wan64_xor"),
    (Path("/root/LatentsCompress/examples/vbench_codec/skyreels_long_and_wan64_lossless_report.md"), "repo_examples"),
    (Path("/root/LatentsCompress/scripts/compare_single_lossless_long_latent_codecs.py"), "repo_scripts"),
    (Path("/root/LatentsCompress/scripts/compare_single_long_latent_xor_codecs.py"), "repo_scripts"),
    (Path("/root/LatentsCompress/scripts/eval_wan64_xor_codecs.py"), "repo_scripts"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload compression experiment artifacts to S3.")
    parser.add_argument("--bucket", default="video-latents")
    parser.add_argument("--prefix", default="compression_results_2026-04-02")
    return parser.parse_args()


def remote_size_if_exists(client, bucket: str, key: str) -> int | None:
    try:
        meta = client.head_object(Bucket=bucket, Key=key)
        return int(meta["ContentLength"])
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise


def iter_files(root: Path):
    if not root.exists():
        return
    if root.is_file():
        yield root, root.name
        return
    for path in root.rglob("*"):
        if path.is_file():
            yield path, path.relative_to(root).as_posix()


def main() -> int:
    args = parse_args()
    client = boto3.session.Session(region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")).client("s3")
    client.head_bucket(Bucket=args.bucket)

    cfg = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=8,
        use_threads=True,
    )

    uploaded = skipped = 0
    uploaded_keys: list[str] = []
    for root, subprefix in EXPORTS:
        for path, rel in iter_files(root):
            key = f"{args.prefix}/{subprefix}/{rel}"
            size = path.stat().st_size
            remote_size = remote_size_if_exists(client, args.bucket, key)
            if remote_size == size:
                skipped += 1
                continue
            client.upload_file(str(path), args.bucket, key, Config=cfg)
            uploaded += 1
            uploaded_keys.append(key)

    print(
        json.dumps(
            {
                "bucket": args.bucket,
                "prefix": args.prefix,
                "uploaded": uploaded,
                "skipped_existing": skipped,
                "uploaded_keys_sample": uploaded_keys[:20],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
