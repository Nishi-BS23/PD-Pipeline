#!/usr/bin/env python3
"""Audio segmentation script for PD vs HC cohorts.

This script is the .py equivalent of audio_segmentation.ipynb and creates:
    segments/{PD,HC}/{early,middle,late}/*.flac

Default behavior:
- Read cohort labels from final_selected.xlsx.
- Scan raw FLAC files under mpower_voice_data_flac* directories.
- Keep only files with duration >= 10 seconds.
- Split into fixed windows: early (0-3s), middle (3-7s), late (7-10s).
"""

from __future__ import annotations

import argparse
import glob
import random
from pathlib import Path

import soundfile as sf

from shared.cohort import assign_class_from_filename, find_raw_flac_files, load_cohort_map
from shared.audio_utils import duration_seconds

SEGMENTS_MODES = {
    "segment": {
        "early": (0.0, 3.0),
        "middle": (3.0, 7.0),
        "late": (7.0, 10.0),
    },
    "full": {
        "full": (0.0, 10.0),
    },
}
COHORTS = ["PD", "HC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment raw FLAC into clips based on mode.")
    parser.add_argument(
        "--mode",
        default="segment",
        choices=["segment", "full"],
        help="Segmentation mode: 'segment' creates early/middle/late, 'full' creates 0-10s full clip.",
    )
    parser.add_argument(
        "--xlsx-path",
        default="final_selected.xlsx",
        help="Path to cohort metadata Excel file.",
    )
    parser.add_argument(
        "--raw-glob",
        default="mpower_voice_data_flac*",
        help="Glob (relative to project root) used to locate raw data roots.",
    )
    parser.add_argument(
        "--output-root",
        default="segments",
        help="Directory where segmented FLAC files are saved.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=10.0,
        help="Minimum source duration in seconds required for segmentation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing segment files.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Balanced cap per class: N means use up to N PD and N HC files (0 = no cap).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --max-per-class is set.",
    )
    return parser.parse_args()


def resolve_raw_roots(project_root: Path, raw_glob: str) -> list[str]:
    roots = sorted(glob.glob(str(project_root / raw_glob)))
    resolved: list[str] = []

    def safe_is_dir(path: Path) -> bool:
        try:
            return path.is_dir()
        except OSError:
            return False

    for root in roots:
        root_path = Path(root)
        nested = root_path / "mpower_voice_data_flac"
        if safe_is_dir(nested):
            resolved.append(str(nested))
        elif safe_is_dir(root_path):
            resolved.append(str(root_path))
    return resolved


def ensure_output_dirs(output_root: Path, segments: dict) -> None:
    for cohort in COHORTS:
        for seg_name in segments:
            (output_root / cohort / seg_name).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    xlsx_path = Path(args.xlsx_path)
    if not xlsx_path.is_absolute():
        xlsx_path = project_root / xlsx_path
    if not xlsx_path.is_file():
        raise FileNotFoundError(f"Cohort file not found: {xlsx_path}")

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root

    raw_roots = resolve_raw_roots(project_root, args.raw_glob)
    if not raw_roots:
        raise FileNotFoundError(
            f"No raw audio roots found for pattern '{args.raw_glob}' under {project_root}"
        )

    segments = SEGMENTS_MODES[args.mode]

    print("=" * 60)
    print(f"Audio Segmentation (.py) — mode={args.mode}")
    print("=" * 60)
    print(f"Project root : {project_root}")
    print(f"Cohort file  : {xlsx_path}")
    print(f"Raw roots    : {raw_roots}")
    print(f"Output root  : {output_root}")
    print(f"Segments     : {list(segments.keys())}")
    print(f"Min duration : {args.min_duration:.1f}s")
    print(f"Overwrite    : {args.overwrite}")
    print(f"Max/class    : {args.max_per_class if args.max_per_class > 0 else 'all'}")
    print(f"Seed         : {args.seed}")

    cohort_map = load_cohort_map(str(xlsx_path))
    all_flac_files = find_raw_flac_files(raw_roots)

    ensure_output_dirs(output_root, segments)

    cohort_files: dict[str, list[str]] = {"PD": [], "HC": []}
    unmatched_files = 0
    discarded_short = 0
    discarded_error = 0

    for fpath in all_flac_files:
        cohort = assign_class_from_filename(fpath, cohort_map)
        if cohort not in COHORTS:
            unmatched_files += 1
            continue
        try:
            dur = duration_seconds(fpath)
        except Exception:
            discarded_error += 1
            continue
        if dur < args.min_duration:
            discarded_short += 1
            continue
        cohort_files[cohort].append(fpath)

    print("\nScan summary")
    print(f"Total FLAC files found : {len(all_flac_files)}")
    print(f"Unmatched (not PD/HC)  : {unmatched_files}")
    print(f"Discarded (<min sec)   : {discarded_short}")
    print(f"Unreadable             : {discarded_error}")
    print(f"Valid PD files         : {len(cohort_files['PD'])}")
    print(f"Valid HC files         : {len(cohort_files['HC'])}")

    if args.max_per_class > 0:
        rng = random.Random(args.seed)
        for cohort in COHORTS:
            files = cohort_files[cohort]
            if len(files) > args.max_per_class:
                cohort_files[cohort] = sorted(rng.sample(files, args.max_per_class))

        print("\nBalanced cap applied")
        print(f"Selected PD files      : {len(cohort_files['PD'])}")
        print(f"Selected HC files      : {len(cohort_files['HC'])}")

    saved_counts = {"PD": 0, "HC": 0}
    skipped_existing = 0
    failed_files = 0

    for cohort in COHORTS:
        for fpath in cohort_files[cohort]:
            try:
                data, samplerate = sf.read(fpath, always_2d=True)
            except Exception:
                failed_files += 1
                continue

            stem = Path(fpath).stem
            file_skipped = False
            for seg_name, (start_s, end_s) in segments.items():
                start_sample = int(start_s * samplerate)
                end_sample = int(end_s * samplerate)
                segment_data = data[start_sample:end_sample]

                if segment_data.shape[0] == 0:
                    file_skipped = True
                    break

                out_name = f"{stem}_{seg_name}.flac"
                out_path = output_root / cohort / seg_name / out_name

                if out_path.exists() and not args.overwrite:
                    continue

                sf.write(str(out_path), segment_data, samplerate)

            if file_skipped:
                failed_files += 1
                continue

            # If all files already existed, count as skipped-existing for transparency.
            if all(
                (output_root / cohort / seg / f"{stem}_{seg}.flac").exists()
                for seg in segments
            ) and not args.overwrite:
                skipped_existing += 1

            saved_counts[cohort] += 1

    print("\nWrite summary")
    print(f"Processed PD files      : {saved_counts['PD']}")
    print(f"Processed HC files      : {saved_counts['HC']}")
    print(f"Skipped existing files  : {skipped_existing}")
    print(f"Failed during slicing   : {failed_files}")
    print(f"Segments root           : {output_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
