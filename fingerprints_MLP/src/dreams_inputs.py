"""DreaMS embedding -> the embedding columns of a combined model's input (train_MLP_combined.py).

Shared by training (train_MLP_combined.py) and prediction (predict.py, the Kaggle kernel), so both build these columns
the same way. Light on purpose: numpy, torch and the input constants only, so it ships in the Kaggle dataset with
predict.py and none of the training scripts.

A combined model's input: [binned spectrum (N_BINS), metadata features (N_METADATA_FEATURES), embedding columns,
has-embedding flag (1)]. The embedding columns are either
- the full embedding x EMBEDDING_SCALE (EMBEDDING_SIZE columns), or
- its first k PCA coordinates (dreaMS/fit_dreams_pca.py, k <= 256), scaled so that their mean square is 1.
The number of columns tells which (pca_dimensions_of).

Which spectra get an embedding: those whose adduct was embedded for training (EMBEDDED_ADDUCTS, the selection of
dreaMS/embed_train.py) with a charge sign matching the polarity. Test spectra must follow the same rule: the model
learned the flag-0 input for the others.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from build_spectrum_arrays import N_BINS
from metadata_features import N_METADATA_FEATURES
from train_MLP import DEVICE

EMBEDDING_SIZE = 1024
# The components of a unit vector of 1024 numbers are ~0.03 in size. Times sqrt(1024) = 32 they are ~1, like the
# other input features. A fixed constant, not a statistic of train, so train and test are scaled the same way.
EMBEDDING_SCALE = 32.0
N_OTHER_INPUTS = N_BINS + N_METADATA_FEATURES  # binned spectrum + metadata, as in train_MLP.py

# The PCA of dreaMS/fit_dreams_pca.py. The Kaggle kernel points it to its dataset folder before predicting.
PCA_PATH = Path(__file__).resolve().parent / "library" / "dreams_pca.npz"

# Adducts with at least 1,000 train spectra: the ones dreaMS/embed_train.py embedded (its selected_rows.parquet)
EMBEDDED_ADDUCTS = {
    "[M+H]+", "[M+Na]+", "[M+NH4]+", "[M+K]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+2H]2+", "[M]+", "[2M+H]+", "[2M+Na]+",
    "[M-H]-", "[M+CH2O2-H]-", "[M+C2H4O2-H]-", "[M+Cl]-", "[2M-H]-", "[2M+Na-2H]-",
}


def gets_embedding(adducts: list[str], ionization_modes: list[str]) -> np.ndarray:
    """(n,) bool: which spectra get a DreaMS embedding, by the rule of dreaMS/embed_train.py.

    Common adduct, and a charge sign matching the polarity ("[M-H]-" in positive mode is a mislabel: no embedding).
    """
    gets = []
    for adduct, mode in zip(adducts, ionization_modes):
        sign_matches = (adduct.endswith("+") and mode == "positive") or (adduct.endswith("-") and mode == "negative")
        gets.append(adduct in EMBEDDED_ADDUCTS and sign_matches)
    return np.array(gets, dtype=bool)


def n_inputs(n_pca_dimensions: int | None) -> int:
    """Input width of a combined model: binned spectrum, metadata, embedding columns, has-embedding flag."""
    n_embedding_columns = EMBEDDING_SIZE if n_pca_dimensions is None else n_pca_dimensions
    return N_OTHER_INPUTS + n_embedding_columns + 1


def pca_dimensions_of(n_model_inputs: int) -> int | None:
    """The PCA dimensions of a combined model with this input width (None: full embedding). Inverse of n_inputs."""
    n_embedding_columns = n_model_inputs - N_OTHER_INPUTS - 1
    return None if n_embedding_columns == EMBEDDING_SIZE else n_embedding_columns


@lru_cache
def pca_projection(n_pca_dimensions: int) -> tuple[np.ndarray, np.ndarray]:
    """(mean (EMBEDDING_SIZE,), projection (EMBEDDING_SIZE, k)) of the PCA at PCA_PATH, first k components.

    The projection includes the scale: one constant for all k coordinates, so that their mean square over the fit
    sample is 1. The first coordinates keep their larger spread (they carry more of the embedding).
    """
    pca = np.load(PCA_PATH)
    components = pca["components"][:n_pca_dimensions]
    scale = 1.0 / np.sqrt(pca["explained_variance"][:n_pca_dimensions].mean())
    return pca["mean"].astype(np.float32), (components.T * scale).astype(np.float32)


def embedding_inputs(embeddings: np.ndarray, use_embedding: np.ndarray, n_pca_dimensions: int | None) -> torch.Tensor:
    """The embedding columns and the has-embedding flag (the last input columns), on DEVICE.

    embeddings: (n, EMBEDDING_SIZE) raw unit-length embeddings, use_embedding: (n,) bool. Rows with use_embedding
    False get zeros and flag 0 (a missing embedding is zeros, whose PCA coordinates would not be zero).
    """
    embeddings = embeddings.astype(np.float32)
    if n_pca_dimensions is None:
        columns = embeddings * EMBEDDING_SCALE
    else:
        mean, projection = pca_projection(n_pca_dimensions)
        columns = (embeddings - mean) @ projection
    columns[~use_embedding] = 0.0
    flag = use_embedding.astype(np.float32)[:, None]
    return torch.from_numpy(np.concatenate([columns, flag], axis=1)).to(DEVICE)
