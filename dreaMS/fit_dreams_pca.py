"""PCA of the DreaMS embeddings, fitted once, for the reduced embedding input of train_MLP_combined.py.

The full embedding is 1024 dense numbers next to a binned spectrum with ~50 non-zero bins: it may dominate the MLP's first layer. A PCA keeps most of its information in fewer numbers (dreaMS/exploration_embeddings.ipynb: 90% of the variance in ~210 components, nearest-neighbour search almost unchanged with 256).

- Fitted on N_FIT random training spectra with an embedding: the train split of train_MLP.py (no validation or  natural-product test molecules), whatever their quality, as the training spectra the model sees.
- Exact PCA: mean, covariance matrix (1024 x 1024), eigenvectors sorted by decreasing variance.
- N_COMPONENTS components are saved. The first k of them are the PCA with k components, so one fit serves every k up to N_COMPONENTS (256, 128, 64).

Output: fingerprints_MLP/library/dreams_pca.npz (dreams_inputs.PCA_PATH)
- mean: float64 (1024,), mean embedding of the fit sample
- components: float64 (N_COMPONENTS, 1024), unit-length directions, by decreasing variance
- explained_variance: float64 (N_COMPONENTS,), variance of the sample along each component
- total_variance: float64 scalar, total variance of the sample (all 1024 directions)

Usage: python fit_dreams_pca.py (from dreaMS/, in the project .venv)
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fingerprints_MLP"))
from build_spectrum_arrays import TRAIN_PATH  # noqa: E402
from dreams_inputs import PCA_PATH  # noqa: E402  (fingerprints_MLP/library/dreams_pca.npz)
from train_MLP import LIBRARY_DIR, SEED, split_spectra_by_molecule  # noqa: E402
from train_MLP_dreams import embeddings_of_rows  # noqa: E402
N_FIT = 200_000
N_COMPONENTS = 256


def main():
    fp_index = np.load(LIBRARY_DIR / "spectrum_arrays.npz")["fp_index"]
    ingest_lib = pl.scan_parquet(TRAIN_PATH).select("ingest_lib").collect()["ingest_lib"]
    train_indices, _ = split_spectra_by_molecule(fp_index, ingest_lib, np.random.default_rng(SEED))

    # a random sample of training spectra, read in row order (one part file at a time)
    rng = np.random.default_rng(SEED)
    sample_rows = np.sort(rng.choice(train_indices, size=N_FIT, replace=False))
    embeddings, has_embedding = embeddings_of_rows(sample_rows)
    embeddings = embeddings[has_embedding].astype(np.float64)
    print(f"fit sample: {len(embeddings):,} training spectra with an embedding")

    mean = embeddings.mean(axis=0)
    centred = embeddings - mean
    covariance = centred.T @ centred / (len(centred) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)  # ascending order
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    total_variance = eigenvalues.sum()
    cumulative = np.cumsum(eigenvalues) / total_variance
    for k in (20, 64, 128, 211, 256):
        print(f"  {k:>3} components: {cumulative[k - 1]:.1%} of the variance")

    np.savez(
        PCA_PATH,
        mean=mean,
        components=eigenvectors[:, :N_COMPONENTS].T,
        explained_variance=eigenvalues[:N_COMPONENTS],
        total_variance=total_variance,
    )
    print(f"Saved {N_COMPONENTS} components to {PCA_PATH}")


if __name__ == "__main__":
    main()
