# Adapted from pipeline.ipynb read_audio cell.
#
# Original (L25-31, fairseq/examples/wav2vec/wav2vec_featurize.py):
#   wav, sr = sf.read(fname)
#   assert sr == 16e3
#   return wav, 16e3
#
# Changes vs original:
#   - always_2d=True + mono mix  (mPower files may be multi-channel)
#   - resample_poly to 16 kHz    (mPower FLAC files are 44.1 kHz)
#   - min_duration_s param for filtering short files

import numpy as np
import soundfile as sf
from math import gcd
from scipy.signal import resample_poly

TARGET_SR = 16_000


def read_audio(fname: str) -> tuple[np.ndarray, int]:
    """Read an audio file, mix to mono, and resample to TARGET_SR (16 kHz).

    Returns (wav_float32, sample_rate).
    Raises RuntimeError if the file cannot be read.
    """
    try:
        wav, sr = sf.read(fname, always_2d=True)
    except Exception as exc:
        raise RuntimeError(f"Could not read {fname}: {exc}") from exc

    # mono mix
    if wav.shape[1] > 1:
        wav = wav.mean(axis=1)
    else:
        wav = wav[:, 0]

    wav = wav.astype(np.float32)

    # resample if needed
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        wav = resample_poly(wav, TARGET_SR // g, sr // g).astype(np.float32)

    return wav, TARGET_SR


def duration_seconds(fname: str) -> float:
    """Return duration in seconds without loading the full audio array."""
    info = sf.info(fname)
    return info.frames / info.samplerate
