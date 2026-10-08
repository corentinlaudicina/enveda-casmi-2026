"""train_MLP.py with the DreaMS embedding of each spectrum as input, instead of the binned spectrum.
Saves models/mlp_dreams.pt.

Needs the embeddings of dreaMS/embed_train.py (one part file per block of train rows, in EMBEDDINGS_DIR).

- Input: the spectrum's DreaMS embedding (1024 numbers, unit length) times EMBEDDING_SCALE,
  followed by the N_METADATA_FEATURES measurement features of metadata_features.py.
  DreaMS is not told the adduct or the collision energy, so the metadata features stay useful.
- Spectra without an embedding are left out of training and validation: those of the rare adducts that
  embed_train.py skipped, and those of the parts not embedded yet (training can start before the embedding
  run is finished, on the parts done so far).
- Everything else as in train_MLP.py: same molecule split (the split happens before the spectra without an
  embedding are dropped, so the validation molecules are the same), quality filter, model size, loss and schedule.

Usage: python train_MLP_dreams.py
"""

from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

from build_spectrum_arrays import TRAIN_PATH
from dreams_inputs import EMBEDDING_SCALE, EMBEDDING_SIZE  # 1024 numbers, x 32 so that each is ~1 in size
from metadata_features import METADATA_COLUMNS, N_METADATA_FEATURES, metadata_features
from morgan_generator import N_BITS
from spectrum_quality import clean_spectrum_mask
from train_MLP import (
    DEVICE, DROPOUT, HIDDEN_SIZE, LIBRARY_DIR, MODEL_DIR, SEED, split_spectra_by_molecule, train_model,
)

EMBEDDINGS_DIR = Path("/Volumes/HDLAUDICINA/enveda-CASMI26-molecule-id-mass-spectra/dreaMS/dreams_train_embeddings")
MODEL_PATH = MODEL_DIR / "mlp_dreams.pt"


def build_model() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(EMBEDDING_SIZE + N_METADATA_FEATURES, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN_SIZE, N_BITS),  # logits; the sigmoid is inside the loss
    )


def load_embeddings(n_spectra: int) -> tuple[np.ndarray, np.ndarray]:
    """(embeddings (n_spectra, EMBEDDING_SIZE) float16, has_embedding (n_spectra,) bool), indexed by train row.

    Rows without an embedding are zeros, and has_embedding is False for them. About 5 GB for all of train.
    """
    embeddings, has_embedding = embeddings_of_rows(np.arange(n_spectra))
    print(f"DreaMS embeddings: {has_embedding.sum():,} of {n_spectra:,} train spectra")
    return embeddings, has_embedding


def embeddings_of_rows(row_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(embeddings (n, EMBEDDING_SIZE) float16, has_embedding (n,) bool) of the given train rows, in their order.

    Like load_embeddings, but reads only the parts holding these rows and keeps only these rows: for evaluation.
    """
    embeddings = np.zeros((len(row_indices), EMBEDDING_SIZE), dtype=np.float16)
    has_embedding = np.zeros(len(row_indices), dtype=bool)

    for part_path in embedding_part_paths():
        part = np.load(part_path)
        part_row_ids = part["row_ids"].astype(np.int64)  # ascending
        in_part = np.isin(row_indices, part_row_ids)
        if not in_part.any():
            continue
        positions = np.searchsorted(part_row_ids, row_indices[in_part])
        embeddings[in_part] = part["embeddings"][positions]
        has_embedding[in_part] = True
    return embeddings, has_embedding


def embedding_part_paths() -> list[Path]:
    """The finished part files of dreaMS/embed_train.py."""
    part_paths = sorted(EMBEDDINGS_DIR.glob("part_*.npz"))
    return [path for path in part_paths if not path.name.endswith(".tmp.npz")]  # a part being written


class DreamsBatcher:
    """Holds the embeddings and metadata features on CPU and builds (inputs, targets) tensors for a set of spectra.

    Same interface as train_MLP.SpectrumBatcher, so train_model can use it.
    """

    def __init__(self, metadata: np.ndarray):
        self.fp_index = np.load(LIBRARY_DIR / "spectrum_arrays.npz")["fp_index"]  # molecule of each spectrum
        self.packed_fingerprints = np.load(LIBRARY_DIR / "morgan_fingerprints.npy")
        self.metadata = metadata  # (n_spectra, N_METADATA_FEATURES), indexed by train row
        self.embeddings, self.has_embedding = load_embeddings(len(metadata))

    def get_batch(self, spectrum_indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.embeddings[spectrum_indices].astype(np.float32) * EMBEDDING_SCALE
        metadata = self.metadata[spectrum_indices].astype(np.float32)
        inputs = torch.from_numpy(np.concatenate([embeddings, metadata], axis=1)).to(DEVICE)

        packed = self.packed_fingerprints[self.fp_index[spectrum_indices]]
        bits = np.unpackbits(packed, axis=1, count=N_BITS)
        targets = torch.from_numpy(bits.astype(np.float32)).to(DEVICE)
        return inputs, targets


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    # per-spectrum columns of train.parquet (no peak lists), in train row order
    spectrum_info = pl.scan_parquet(TRAIN_PATH).select(["ingest_lib"] + METADATA_COLUMNS).collect()
    metadata = metadata_features(spectrum_info)

    batcher = DreamsBatcher(metadata)
    train_indices, validation_indices = split_spectra_by_molecule(batcher.fp_index, spectrum_info["ingest_lib"], rng)

    # drop the clearly wrong or useless spectra from training (validation is enveda-180: already clean)
    is_clean = clean_spectrum_mask(TRAIN_PATH)
    n_before = len(train_indices)
    train_indices = train_indices[is_clean[train_indices]]
    print(f"quality filter: kept {len(train_indices):,} of {n_before:,} train spectra")

    # only spectra with an embedding
    n_train_before = len(train_indices)
    n_validation_before = len(validation_indices)
    train_indices = train_indices[batcher.has_embedding[train_indices]]
    validation_indices = validation_indices[batcher.has_embedding[validation_indices]]
    print(f"with a DreaMS embedding: {len(train_indices):,} of {n_train_before:,} train spectra, "
          f"{len(validation_indices):,} of {n_validation_before:,} validation spectra")
    print(f"device {DEVICE} | {len(train_indices):,} train spectra, {len(validation_indices):,} validation spectra")

    train_model(build_model(), batcher, train_indices, validation_indices, rng, MODEL_PATH)


if __name__ == "__main__":
    main()
