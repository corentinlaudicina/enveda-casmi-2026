"""Check that dreams_lite.py reproduces the official DreaMS embeddings stored by embed_train.py.

Takes N_PER_PART random spectra from each of PARTS, embeds them with dreams_lite (no dreams package), and compares
with the stored embeddings (float16): cosine similarity and largest absolute difference per spectrum.
Runs in the main environment (no dreams package needed), from the dreaMS folder:
    ../.venv/bin/python verify_dreams_lite.py
"""

import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from dreams_lite import embed, load_embedder, preprocess

PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_DIR / "data" / "enveda-CASMI26-molecule-id-mass-spectra" / "train.parquet"
EMBEDDINGS_DIR = Path("/Volumes/HDLAUDICINA/enveda-CASMI26-molecule-id-mass-spectra/dreaMS/dreams_train_embeddings")
WEIGHTS_PATH = Path(__file__).resolve().parent / "dreams_cache" / "dreams_embedding_weights.pt"
PARTS = ["part_0000000.npz", "part_1048576.npz"]  # an enveda-180 part, and the part where enveda-180 ends
N_PER_PART = 500
SEED = 0


def main():
    rng = np.random.default_rng(SEED)
    row_ids = []
    stored = []
    for part_name in PARTS:
        part = np.load(EMBEDDINGS_DIR / part_name)
        chosen = np.sort(rng.choice(len(part["row_ids"]), size=N_PER_PART, replace=False))
        row_ids.append(part["row_ids"][chosen].astype(np.int64))
        stored.append(part["embeddings"][chosen].astype(np.float32))
    row_ids = np.concatenate(row_ids)
    stored = np.concatenate(stored)

    spectra = (
        pl.scan_parquet(TRAIN_PATH)
        .with_row_index("row_id")
        .filter(pl.col("row_id").is_in(row_ids.tolist()))
        .select("row_id", "ingest_lib", "ms2_mzs", "ms2_normalized_intensities", "precursor_mz")
        .collect()
    )
    assert np.array_equal(spectra["row_id"].to_numpy(), row_ids)
    print(f"{spectra.height} spectra, libraries: {spectra['ingest_lib'].value_counts().sort('ingest_lib').rows()}")

    prepared = []
    for row in spectra.iter_rows(named=True):
        prepared.append(preprocess(row["ms2_mzs"], row["ms2_normalized_intensities"], row["precursor_mz"]))

    embedder = load_embedder(WEIGHTS_PATH, device="cpu")
    print(f"torch {torch.__version__}, numpy {np.__version__}")
    started = time.time()
    lite = embed(embedder, prepared)
    print(f"embedded on CPU in {time.time() - started:.1f} s ({len(prepared) / (time.time() - started):.0f} spectra/s)")

    cosine = (lite * stored).sum(axis=1) / np.linalg.norm(stored, axis=1)
    largest_difference = np.abs(lite - stored).max(axis=1)
    print(f"cosine similarity lite vs official: min {cosine.min():.6f}, median {np.median(cosine):.6f}")
    print(f"largest absolute difference per spectrum: median {np.median(largest_difference):.2e}, "
          f"max {largest_difference.max():.2e} (float16 storage alone gives up to ~1e-4)")
    worst = np.argsort(cosine)[:3]
    print("worst spectra (row_id, cosine):", [(int(row_ids[i]), round(float(cosine[i]), 6)) for i in worst])


if __name__ == "__main__":
    main()
