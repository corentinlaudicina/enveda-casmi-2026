"""Flag the clearly wrong or useless train spectra, so they can be left out of training.

A spectrum is dropped if any of these holds:
- |precursor_error_ppm| > MAX_PRECURSOR_ERROR_PPM: the precursor doesn't match the labelled molecule's mass,
  so the label (structure or adduct) is probably wrong;
- fewer than MIN_PEAKS peaks: almost no information;
- a peak more than MAX_PEAK_ABOVE_PRECURSOR Da above the precursor m/z: a fragment can't be heavier than its
  precursor, so the spectrum is contaminated or chimeric (the margin leaves room for the precursor's isotope peaks).
"""

from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

MAX_PRECURSOR_ERROR_PPM = 20.0
MIN_PEAKS = 3
MAX_PEAK_ABOVE_PRECURSOR = 3.0  # Da

BATCH_SIZE = 100_000


def clean_spectrum_mask(train_path: Path) -> np.ndarray:
    """Boolean array, one entry per row of train.parquet: True if the spectrum passes all three checks.

    Reads the file batch by batch (the peak lists are large), in row order.
    """
    parquet_file = pq.ParquetFile(train_path)
    is_clean_batches = []

    batches = parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["precursor_mz", "precursor_error_ppm", "num_peaks", "ms2_mzs"],
    )
    for batch in batches:
        spectra = pl.from_arrow(batch)
        highest_peak_above_precursor = pl.col("ms2_mzs").list.max() - pl.col("precursor_mz")

        is_clean = spectra.select(
            (pl.col("precursor_error_ppm").abs() <= MAX_PRECURSOR_ERROR_PPM).fill_null(True)
            & (pl.col("num_peaks") >= MIN_PEAKS)
            & (highest_peak_above_precursor <= MAX_PEAK_ABOVE_PRECURSOR).fill_null(True)
        ).to_series()
        is_clean_batches.append(is_clean.to_numpy())

    return np.concatenate(is_clean_batches)
