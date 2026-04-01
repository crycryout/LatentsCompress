#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


DIMENSION_TO_SUITE = {
    "subject_consistency": "subject_consistency",
    "background_consistency": "scene",
    "temporal_flickering": "temporal_flickering",
    "motion_smoothness": "subject_consistency",
    "dynamic_degree": "subject_consistency",
    "aesthetic_quality": "overall_consistency",
    "imaging_quality": "overall_consistency",
    "object_class": "object_class",
    "multiple_objects": "multiple_objects",
    "human_action": "human_action",
    "color": "color",
    "spatial_relationship": "spatial_relationship",
    "scene": "scene",
    "temporal_style": "temporal_style",
    "appearance_style": "appearance_style",
    "overall_consistency": "overall_consistency",
}


def load_prompts(prompt_dir: Path, suite_name: str) -> list[str]:
    path = prompt_dir / f"{suite_name}.txt"
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [p for p in prompts if p]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vbench-root",
        default="/workspace/VBench",
        help="Path to the cloned VBench repo.",
    )
    parser.add_argument(
        "--samples-per-dimension",
        type=int,
        default=20,
        help="How many prompts to sample for each VBench dimension.",
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=42,
        help="Seed for prompt sampling.",
    )
    parser.add_argument(
        "--generation-seed",
        type=int,
        default=42,
        help="Fixed generation seed to attach to each item.",
    )
    parser.add_argument(
        "--output",
        default="/workspace/video_bench/manifests/wan22_vbench_16x20_seed42.json",
        help="Output manifest path.",
    )
    args = parser.parse_args()

    prompt_dir = Path(args.vbench_root) / "prompts" / "prompts_per_dimension"
    rng = random.Random(args.selection_seed)
    rows = []

    for dimension, suite_name in DIMENSION_TO_SUITE.items():
        suite_prompts = load_prompts(prompt_dir, suite_name)
        if len(suite_prompts) < args.samples_per_dimension:
            raise ValueError(
                f"{suite_name} only has {len(suite_prompts)} prompts, "
                f"cannot sample {args.samples_per_dimension}."
            )

        sampled_indices = sorted(
            rng.sample(range(len(suite_prompts)), args.samples_per_dimension)
        )
        for sample_rank, prompt_index in enumerate(sampled_indices):
            prompt = suite_prompts[prompt_index]
            rows.append(
                {
                    "dimension": dimension,
                    "prompt_suite": suite_name,
                    "sample_rank": sample_rank,
                    "prompt_index_in_suite": prompt_index,
                    "prompt": prompt,
                    "generation_seed": args.generation_seed,
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output_path)
    print(f"items={len(rows)}")


if __name__ == "__main__":
    main()
