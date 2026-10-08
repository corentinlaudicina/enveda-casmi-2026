"""Reference DreaMS embeddings of test.parquet, computed with the OFFICIAL dreams package, for the Kaggle test notebook.

The Kaggle notebook embeds the same spectra with dreams_lite.py and compares with this file, so the check does not
depend on dreams_lite being right. Runs on CPU (the MPS device may be busy with embed_train.py), in the DreaMS
environment, from the dreaMS folder:
    .venv-dreams/bin/python make_test_reference.py

Output: ../kaggle_submission/dreams_lite_dataset/test_reference_embeddings.npz
- spectrum_ids: test.parquet's spectrum_id, in file order
- embeddings: (n_spectra, 1024) float32, unit length
"""

import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from embed_dreams import DreamsEmbedder

PROJECT_DIR = Path(__file__).resolve().parent.parent
TEST_PATH = PROJECT_DIR / "data" / "enveda-CASMI26-molecule-id-mass-spectra" / "test.parquet"
OUTPUT_PATH = PROJECT_DIR / "kaggle_submission" / "dreams_lite_dataset" / "test_reference_embeddings.npz"
BATCH_SIZE = 16  # the official model builds a (batch, 101, 101, 980) tensor: keep it small on CPU


def main():
    test = pl.read_parquet(TEST_PATH)
    embedder = DreamsEmbedder()
    embedder.model.to("cpu")

    prepared = []
    for row in test.iter_rows(named=True):
        prepared.append(embedder.prepare(row["ms2_mzs"], row["ms2_normalized_intensities"], row["precursor_mz"]))

    started = time.time()
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(prepared), BATCH_SIZE):
            batch = torch.from_numpy(np.stack(prepared[start:start + BATCH_SIZE]))
            embeddings.append(embedder.model(batch).numpy())
    embeddings = np.concatenate(embeddings)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    print(f"{len(embeddings)} test spectra embedded on CPU in {time.time() - started:.0f} s")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT_PATH, spectrum_ids=test["spectrum_id"].to_numpy(), embeddings=embeddings.astype(np.float32))
    print(f"saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
