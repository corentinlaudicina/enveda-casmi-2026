"""Turn every train spectrum into fixed-size peak arrays, the MLP input, and link it to its fingerprint.

Each spectrum keeps its MAX_PEAKS strongest peaks below MAX_MZ, stored as
- bin index: floor(m/z / BIN_WIDTH), so 10,000 bins of 0.1 Da from 0 to 1000
- intensity: sqrt of the normalized intensity (compresses the many weak peaks vs the base peak)
Spectra with fewer peaks are padded with bin 0 / intensity 0, which add nothing when binned.

train.parquet is read batch by batch, without the label-only and metadata columns.

Output: fingerprints_MLP/library/spectrum_arrays.npz with
- peak_bins (n_spectra, MAX_PEAKS) int16
- peak_intensities (n_spectra, MAX_PEAKS) float16
- fp_index (n_spectra,) int32: row of the molecule in molecules.parquet / morgan_fingerprints.npy
"""

from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_DIR / "data" / "enveda-CASMI26-molecule-id-mass-spectra" / "train.parquet"
LIBRARY_DIR = Path(__file__).resolve().parent / "library"

MAX_MZ = 1000.0
BIN_WIDTH = 0.1
N_BINS = int(MAX_MZ / BIN_WIDTH)  # 10,000
MAX_PEAKS = 100
BATCH_SIZE = 100_000


def spectrum_to_peaks(mzs: np.ndarray, intensities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the MAX_PEAKS strongest peaks below MAX_MZ; return (bins, sqrt intensities), each of length MAX_PEAKS."""
    below_max = mzs < MAX_MZ
    mzs = mzs[below_max]
    intensities = intensities[below_max]

    if len(mzs) > MAX_PEAKS:
        strongest = np.argsort(intensities)[-MAX_PEAKS:]
        mzs = mzs[strongest]
        intensities = intensities[strongest]

    bins = np.zeros(MAX_PEAKS, dtype=np.int16)
    sqrt_intensities = np.zeros(MAX_PEAKS, dtype=np.float16)
    n_peaks = len(mzs)
    bins[:n_peaks] = (mzs / BIN_WIDTH).astype(np.int16)
    sqrt_intensities[:n_peaks] = np.sqrt(intensities)
    return bins, sqrt_intensities


def build_spectrum_arrays(train_path: Path) -> dict[str, np.ndarray]:
    """Read train.parquet sequentially and convert every spectrum with spectrum_to_peaks."""
    # SMILES -> fp_index, from the fingerprint library built by build_fingerprint_library.py
    molecules = pl.read_parquet(LIBRARY_DIR / "molecules.parquet")
    smiles_to_fp_index = dict(zip(molecules["normalized_smiles"], molecules["fp_index"]))

    parquet_file = pq.ParquetFile(train_path)
    n_spectra = parquet_file.metadata.num_rows

    peak_bins = np.zeros((n_spectra, MAX_PEAKS), dtype=np.int16)
    peak_intensities = np.zeros((n_spectra, MAX_PEAKS), dtype=np.float16)
    fp_index = np.zeros(n_spectra, dtype=np.int32)

    row = 0  # position of the current spectrum in train.parquet
    batches = parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["normalized_smiles", "ms2_mzs", "ms2_normalized_intensities"],
    )
    for batch in batches:
        smiles_column = batch.column("normalized_smiles").to_pylist()

        # The list columns as one flat array of values + offsets: spectrum i is values[offsets[i]:offsets[i + 1]]
        mz_column = batch.column("ms2_mzs")
        intensity_column = batch.column("ms2_normalized_intensities")
        all_mzs = mz_column.flatten().to_numpy()
        all_intensities = intensity_column.flatten().to_numpy()
        offsets = mz_column.offsets.to_numpy()
        offsets = offsets - offsets[0]

        for i in range(batch.num_rows):
            start = offsets[i]
            end = offsets[i + 1]
            bins, sqrt_intensities = spectrum_to_peaks(all_mzs[start:end], all_intensities[start:end])
            peak_bins[row] = bins
            peak_intensities[row] = sqrt_intensities
            fp_index[row] = smiles_to_fp_index[smiles_column[i]]
            row += 1

        print(f"{row:>9,} / {n_spectra:,} spectra")

    return {"peak_bins": peak_bins, "peak_intensities": peak_intensities, "fp_index": fp_index}


if __name__ == "__main__":
    arrays = build_spectrum_arrays(TRAIN_PATH)
    np.savez(LIBRARY_DIR / "spectrum_arrays.npz", **arrays)
    print(f"Saved {len(arrays['fp_index']):,} spectra to {LIBRARY_DIR / 'spectrum_arrays.npz'}")
