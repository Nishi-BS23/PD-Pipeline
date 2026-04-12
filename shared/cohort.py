# Cohort assignment logic extracted from audio_segmentation.ipynb.
#
# Reads final_selected.xlsx and builds a {record_id -> "PD"|"HC"} map.
# Uses the same approach as the segmentation notebook: split filename on "_"
# to extract the record_id prefix.

import glob
import os
import pandas as pd
from typing import Optional


def load_cohort_map(xlsx_path: str) -> dict[str, str]:
    """Return a mapping {record_id: cohort_label} where cohort_label is 'PD' or 'HC'.

    Reads final_selected.xlsx and extracts cohort groupings using the same
    column structure assumed by audio_segmentation.ipynb.
    """
    df = pd.read_excel(xlsx_path, engine="openpyxl")

    # Normalise column names to lowercase for safety
    df.columns = [c.strip().lower() for c in df.columns]

    # The notebook groups by a column that distinguishes PD from HC.
    # Common column names seen in mPower studies:
    #   'professional_diagnosis', 'are_caretaker', 'health_history', ...
    # We look for a column that encodes the cohort label.
    # Based on audio_segmentation.ipynb the relevant column appears to be a
    # binary/categorical field; we check for standard possibilities.
    cohort_col = None
    for candidate in ["cohort", "label", "group", "class", "category",
                       "professional_diagnosis"]:
        if candidate in df.columns:
            cohort_col = candidate
            break

    if cohort_col is None:
        # Fall back: use the first non-recordid/non-fileid column
        id_like = {"recordid", "fileid", "record_id", "file_id"}
        for col in df.columns:
            if col not in id_like:
                cohort_col = col
                break

    if cohort_col is None:
        raise ValueError(
            f"Cannot identify cohort column in {xlsx_path}. "
            f"Columns present: {list(df.columns)}"
        )

    record_id_col = None
    for candidate in ["recordid", "record_id", "healthcode", "health_code"]:
        if candidate in df.columns:
            record_id_col = candidate
            break
    if record_id_col is None:
        raise ValueError(
            f"Cannot identify record-id column in {xlsx_path}. "
            f"Columns present: {list(df.columns)}"
        )

    cohort_map: dict[str, str] = {}
    for _, row in df.iterrows():
        rid = str(row[record_id_col]).strip()
        raw_label = str(row[cohort_col]).strip()
        # Normalise to 'PD' or 'HC'
        if raw_label in ("1", "True", "true", "yes", "PD", "pd"):
            label = "PD"
        else:
            label = "HC"
        cohort_map[rid] = label

    return cohort_map


def assign_class_from_filename(fname: str, cohort_map: dict[str, str]) -> Optional[str]:
    """Return 'PD' or 'HC' for a given FLAC filename, or None if not found.

    Filename pattern: {record_id}_{file_id}.flac
    Tries progressively shorter prefixes until a match is found.
    """
    stem = os.path.splitext(os.path.basename(fname))[0]
    parts = stem.split("_")
    # Try longest prefix first (record_id may contain hyphens/underscores)
    for n in range(len(parts), 0, -1):
        candidate = "_".join(parts[:n])
        if candidate in cohort_map:
            return cohort_map[candidate]
    return None


def find_raw_flac_files(raw_roots: list[str]) -> list[str]:
    """Glob all *.flac files under one or more raw data root directories."""
    files: list[str] = []
    for root in raw_roots:
        files.extend(sorted(glob.glob(os.path.join(root, "**", "*.flac"), recursive=True)))
        files.extend(sorted(glob.glob(os.path.join(root, "*.flac"))))
    # deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique
