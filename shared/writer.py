# Adapted from pipeline.ipynb EmbeddingDatasetWriter class (L111-220 of fairseq reference).
#
# Two writers are provided:
#   EmbeddingDatasetWriter  — segment mode (reads from segments/ directory)
#   FullAudioWriter         — full mode    (reads from raw mpower FLAC files)
#
# Both share the same interface: require_output_dirs(), write_features(), write_metadata().
# Resume logic is identical: skip if .npy output already exists.

import csv
import glob
import os
import random
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import tqdm

from shared.audio_utils import read_audio, duration_seconds
from shared.cohort import (
    assign_class_from_filename,
    find_raw_flac_files,
    load_cohort_map,
)

CLASSES   = ["PD", "HC"]
SEGMENTS  = ["early", "middle", "late"]
MIN_DURATION_FULL = 3.0   # seconds — minimum duration to include in full mode


class BaseWriter(ABC):
    """Common interface for both segment-wise and full-audio embedding writers."""

    def __init__(self, output_root: str, model_fname: str,
                 gpu: int = 0, verbose: bool = True, use_feat: bool = False):
        self.output_root = output_root
        self.model_fname = model_fname
        self.verbose      = verbose
        self.use_feat     = use_feat
        self.metadata: list[dict] = []
        # model is injected after construction so that model.py can pass the
        # appropriate Prediction class without coupling shared/ to model code.
        self._model = None

    def set_model(self, prediction_instance) -> None:           # L138 equivalent
        self._model = prediction_instance

    def _progress(self, iterable, **kwargs):                    # L151-154
        return tqdm.tqdm(iterable, **kwargs) if self.verbose else iterable

    @abstractmethod
    def require_output_dirs(self) -> None:
        ...

    @abstractmethod
    def write_features(self) -> None:
        ...

    def write_metadata(self, path: str) -> None:
        fields = ["filename", "class", "embedding_path", "seq_path"]
        # add optional segment field if present
        if self.metadata and "segment" in self.metadata[0]:
            fields = ["filename", "class", "segment", "original_stem",
                      "embedding_path", "seq_path"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.metadata)
        print(f"Metadata  ->  {path}")


# ---------------------------------------------------------------------------
# Segment-wise writer (mirrors pipeline.ipynb EmbeddingDatasetWriter exactly)
# ---------------------------------------------------------------------------

class EmbeddingDatasetWriter(BaseWriter):
    """Write embeddings from pre-segmented FLAC files (segments/ directory).

    Adapted from:
        external/fairseq/examples/wav2vec/wav2vec_featurize.py  L111-220

    Original class structure preserved:
        __init__             L123-148  model + dataset bookkeeping
        _progress            L151-154  optional tqdm wrapper
        require_output_path  L156-158  os.makedirs
        input_fnames         L189-191  glob *.flac per (cls, seg)
        write_features       L196-214  core extraction loop
        __repr__             L216-220
    """

    def __init__(self, input_root: str, output_root: str, model_fname: str,  # L123
                 gpu: int = 0, verbose: bool = True, use_feat: bool = False,
                 max_per_class: Optional[int] = None, seed: int = 42):
        super().__init__(output_root, model_fname, gpu, verbose, use_feat)
        assert os.path.isdir(input_root), f"Not found: {input_root}"
        self.input_root = input_root
        self.max_per_class = max_per_class
        self.seed = seed

    def _input_fnames(self, cls: str, seg: str) -> list[str]:              # mirrors L189-191
        return sorted(glob.glob(os.path.join(self.input_root, cls, seg, "*.flac")))

    def _output_dir(self, cls: str, seg: str) -> str:
        return os.path.join(self.output_root, cls, seg)

    def require_output_dirs(self) -> None:                                  # mirrors L156-158
        for cls in CLASSES:
            for seg in SEGMENTS:
                os.makedirs(self._output_dir(cls, seg), exist_ok=True)

    def write_features(self) -> None:                                       # L196-214
        assert self._model is not None, "Call set_model() before write_features()"
        rng = random.Random(self.seed)

        for cls in CLASSES:
            selected_stems = None
            if self.max_per_class and self.max_per_class > 0:
                early_files = self._input_fnames(cls, "early")
                stems = [os.path.splitext(os.path.basename(f))[0].removesuffix("_early")
                         for f in early_files]
                if len(stems) > self.max_per_class:
                    stems = rng.sample(stems, self.max_per_class)
                selected_stems = set(stems)

            for seg in SEGMENTS:
                fnames  = self._input_fnames(cls, seg)
                if selected_stems is not None:
                    fnames = [
                        f for f in fnames
                        if os.path.splitext(os.path.basename(f))[0].removesuffix(f"_{seg}") in selected_stems
                    ]
                out_dir = self._output_dir(cls, seg)
                skipped = 0
                for fpath in self._progress(fnames, desc=f"{cls}/{seg}", unit="file"):
                    stem          = os.path.splitext(os.path.basename(fpath))[0]
                    mean_path     = os.path.join(out_dir, f"{stem}.npy")
                    seq_path      = os.path.join(out_dir, f"{stem}_seq.npy")
                    original_stem = "_".join(stem.split("_")[:-1])

                    # resume: skip if both outputs already exist
                    if os.path.isfile(mean_path) and os.path.isfile(seq_path):
                        skipped += 1
                        self.metadata.append({
                            "filename": os.path.basename(fpath), "class": cls,
                            "segment": seg, "original_stem": original_stem,
                            "embedding_path": mean_path, "seq_path": seq_path,
                        })
                        continue

                    wav, sr   = read_audio(fpath)                           # L210
                    z, c      = self._model(wav, sr)                        # L211
                    features  = z if self.use_feat else c                   # L212
                    mean_vec  = features.mean(axis=0)
                    np.save(mean_path, mean_vec)
                    np.save(seq_path,  features)
                    self.metadata.append({
                        "filename": os.path.basename(fpath), "class": cls,
                        "segment": seg, "original_stem": original_stem,
                        "embedding_path": mean_path, "seq_path": seq_path,
                    })

                if skipped:
                    print(f"  {cls}/{seg}: {skipped} files already done, skipped.")

    def __repr__(self) -> str:                                              # L216-220
        n = sum(len(self._input_fnames(c, s)) for c in CLASSES for s in SEGMENTS)
        return (f"EmbeddingDatasetWriter ({n} files)\n"
                f"\tinput_root  : {self.input_root}\n"
                f"\toutput_root : {self.output_root}\n"
                f"\tmodel       : {self.model_fname}")


# ---------------------------------------------------------------------------
# Full-audio writer (new — reads raw mPower FLAC, no temporal segmentation)
# ---------------------------------------------------------------------------

class FullAudioWriter(BaseWriter):
    """Write embeddings from raw (unsegmented) mPower FLAC files.

    Cohort labels are resolved from final_selected.xlsx using the same
    logic as audio_segmentation.ipynb. Files shorter than MIN_DURATION_FULL
    seconds are skipped (no minimum imposed beyond basic readability).

    Output structure:
        embeddings_full/
        ├── PD/  {stem}_full.npy   {stem}_full_seq.npy
        └── HC/  ...
    """

    def __init__(self, raw_roots: list[str], output_root: str, model_fname: str,
                 xlsx_path: str, gpu: int = 0, verbose: bool = True,
                 use_feat: bool = False,
                 min_duration: float = MIN_DURATION_FULL,
                 max_per_class: Optional[int] = None,
                 seed: int = 42):
        super().__init__(output_root, model_fname, gpu, verbose, use_feat)
        self.raw_roots    = raw_roots
        self.xlsx_path    = xlsx_path
        self.min_duration = min_duration
        self.max_per_class = max_per_class
        self.seed = seed

    def require_output_dirs(self) -> None:
        for cls in CLASSES:
            os.makedirs(os.path.join(self.output_root, cls), exist_ok=True)

    def _collect_files(self) -> dict[str, list[str]]:
        """Return {cls: [fpath, ...]} after cohort assignment and duration filter."""
        cohort_map = load_cohort_map(self.xlsx_path)
        all_files  = find_raw_flac_files(self.raw_roots)

        by_class: dict[str, list[str]] = {cls: [] for cls in CLASSES}
        skipped_no_label = 0
        skipped_short    = 0

        for fpath in all_files:
            cls = assign_class_from_filename(fpath, cohort_map)
            if cls is None:
                skipped_no_label += 1
                continue
            try:
                dur = duration_seconds(fpath)
            except Exception:
                continue
            if dur < self.min_duration:
                skipped_short += 1
                continue
            by_class[cls].append(fpath)

        print(f"  Raw files found   : {len(all_files)}")
        print(f"  No cohort label   : {skipped_no_label}")
        print(f"  Too short (<{self.min_duration}s): {skipped_short}")
        for cls in CLASSES:
            if self.max_per_class and self.max_per_class > 0 and len(by_class[cls]) > self.max_per_class:
                rng = random.Random(self.seed)
                by_class[cls] = sorted(rng.sample(by_class[cls], self.max_per_class))
            print(f"  {cls}              : {len(by_class[cls])} files")
        return by_class

    def write_features(self) -> None:
        assert self._model is not None, "Call set_model() before write_features()"
        by_class = self._collect_files()

        for cls in CLASSES:
            out_dir = os.path.join(self.output_root, cls)
            skipped = 0
            for fpath in self._progress(by_class[cls], desc=cls, unit="file"):
                stem      = os.path.splitext(os.path.basename(fpath))[0]
                mean_path = os.path.join(out_dir, f"{stem}_full.npy")
                seq_path  = os.path.join(out_dir, f"{stem}_full_seq.npy")

                # resume
                if os.path.isfile(mean_path) and os.path.isfile(seq_path):
                    skipped += 1
                    self.metadata.append({
                        "filename": os.path.basename(fpath), "class": cls,
                        "embedding_path": mean_path, "seq_path": seq_path,
                    })
                    continue

                try:
                    wav, sr = read_audio(fpath)
                except RuntimeError as exc:
                    print(f"  WARNING: {exc} — skipped")
                    continue

                z, c     = self._model(wav, sr)
                features = z if self.use_feat else c
                mean_vec = features.mean(axis=0)
                np.save(mean_path, mean_vec)
                np.save(seq_path,  features)
                self.metadata.append({
                    "filename": os.path.basename(fpath), "class": cls,
                    "embedding_path": mean_path, "seq_path": seq_path,
                })

            if skipped:
                print(f"  {cls}: {skipped} files already done, skipped.")

    def __repr__(self) -> str:
        return (f"FullAudioWriter\n"
                f"\traw_roots   : {self.raw_roots}\n"
                f"\toutput_root : {self.output_root}\n"
                f"\tmodel       : {self.model_fname}")
