"""HuBERT embedding extraction pipeline.

Usage:
    python HuBERT/pipeline.py --mode segment   # uses segments/ directory
    python HuBERT/pipeline.py --mode full       # uses raw mPower FLAC files

Mirrors Wav2Vec2/pipeline.py exactly; only the model class and output paths differ.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared.writer import EmbeddingDatasetWriter, FullAudioWriter
from HuBERT.model import Prediction

# ---------------------------------------------------------------------------
# Paths (parallel to Wav2Vec2 paths)
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_NAME   = "facebook/hubert-base-ls960"

SEG_ROOT     = os.path.join(PROJECT_ROOT, "segments")
EMB_ROOT     = os.path.join(PROJECT_ROOT, "HuBERT", "embeddings")
AGG_ROOT     = os.path.join(PROJECT_ROOT, "HuBERT", "embeddings_aggregated")
META_PATH    = os.path.join(PROJECT_ROOT, "HuBERT", "metadata.csv")
META_AGG     = os.path.join(PROJECT_ROOT, "HuBERT", "metadata_aggregated.csv")

EMB_FULL     = os.path.join(PROJECT_ROOT, "HuBERT", "embeddings_full")
META_FULL    = os.path.join(PROJECT_ROOT, "HuBERT", "metadata_full.csv")

XLSX_PATH    = os.path.join(PROJECT_ROOT, "final_selected.xlsx")

CLASSES  = ["PD", "HC"]
SEGMENTS = ["early", "middle", "late"]


def run_aggregation(emb_root: str, agg_root: str, strategy: str = "concat") -> pd.DataFrame:
    """Combine early+middle+late HuBERT embeddings per subject.

    Mirrors Wav2Vec2/pipeline.py run_aggregation — same logic, different paths.
    """
    import tqdm
    os.makedirs(agg_root, exist_ok=True)
    records = []
    for cls in CLASSES:
        early_dir = os.path.join(emb_root, cls, "early")
        if not os.path.isdir(early_dir):
            continue
        stems = sorted(
            os.path.splitext(f)[0].removesuffix("_early")
            for f in os.listdir(early_dir)
            if f.endswith(".npy") and not f.endswith("_seq.npy")
        )
        out_cls = os.path.join(agg_root, cls)
        os.makedirs(out_cls, exist_ok=True)
        for orig_stem in tqdm.tqdm(stems, desc=cls, unit="file"):
            vecs, skip = [], False
            for seg in SEGMENTS:
                path = os.path.join(emb_root, cls, seg, f"{orig_stem}_{seg}.npy")
                if not os.path.isfile(path):
                    skip = True
                    break
                vecs.append(np.load(path))
            if skip:
                continue
            agg = np.concatenate(vecs) if strategy == "concat" else np.stack(vecs).mean(0)
            out = os.path.join(out_cls, f"{orig_stem}_{strategy}.npy")
            np.save(out, agg)
            records.append({
                "original_stem":  orig_stem,
                "class":          cls,
                "strategy":       strategy,
                "embedding_dim":  agg.shape[0],
                "embedding_path": out,
            })
    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract HuBERT embeddings (segment or full-audio mode)."
    )
    parser.add_argument(
        "--mode", choices=["segment", "full"], default="segment",
        help="'segment' reads from segments/ (early/middle/late); "
             "'full' reads raw FLAC from mpower_voice_data_flac-*/"
    )
    parser.add_argument("--gpu",      type=int, default=0,
                        help="GPU index (ignored if no CUDA)")
    parser.add_argument("--use-feat", action="store_true",
                        help="Use CNN features (z) instead of transformer output (c)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  HuBERT Embedding Pipeline  [mode={args.mode}]")
    print(f"{'='*60}")
    print(f"  Model : {MODEL_NAME}")

    if args.mode == "segment":
        writer = EmbeddingDatasetWriter(
            input_root  = SEG_ROOT,
            output_root = EMB_ROOT,
            model_fname = MODEL_NAME,
            gpu         = args.gpu,
            use_feat    = args.use_feat,
        )
        print(f"\n{writer}")
        writer.require_output_dirs()
        print("\nLoading model...")
        writer.set_model(Prediction(MODEL_NAME, args.gpu))
        print("Extracting embeddings...")
        writer.write_features()
        print("Done.\n")
        writer.write_metadata(META_PATH)

        print("\nAggregating segments (concat)...")
        agg_df = run_aggregation(EMB_ROOT, AGG_ROOT, strategy="concat")
        agg_df.to_csv(META_AGG, index=False)
        print(f"Aggregated {len(agg_df)} recordings  ->  {META_AGG}")

    else:
        raw_roots = sorted(glob.glob(os.path.join(PROJECT_ROOT, "mpower_voice_data_flac*")))
        if not raw_roots:
            print(f"ERROR: No mpower_voice_data_flac* directory found under {PROJECT_ROOT}")
            sys.exit(1)
        print(f"  Raw data roots: {raw_roots}")

        writer = FullAudioWriter(
            raw_roots   = raw_roots,
            output_root = EMB_FULL,
            model_fname = MODEL_NAME,
            xlsx_path   = XLSX_PATH,
            gpu         = args.gpu,
            use_feat    = args.use_feat,
        )
        print(f"\n{writer}")
        writer.require_output_dirs()
        print("\nLoading model...")
        writer.set_model(Prediction(MODEL_NAME, args.gpu))
        print("Extracting embeddings (full audio)...")
        writer.write_features()
        print("Done.\n")
        writer.write_metadata(META_FULL)


if __name__ == "__main__":
    main()
