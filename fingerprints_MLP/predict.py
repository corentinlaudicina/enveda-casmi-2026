"""Use the trained MLP: spectra in, fingerprint bit probabilities out.

- load_model: the saved weights of train_MLP.py, train_MLP_negatives.py, train_MLP_resampled.py or train_MLP_combined.py
- spectra_to_arrays: a polars frame of spectra (train or test format) -> the compact arrays the model reads
- predict_probabilities: those arrays (+ the DreaMS embeddings for a combined model) -> (n_spectra, N_BITS) probabilities
"""

from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

from build_spectrum_arrays import MAX_PEAKS, spectrum_to_peaks
from metadata_features import metadata_features
from train_MLP import BATCH_SIZE, DEVICE, MODEL_PATH, build_inputs, build_model
from dreams_inputs import N_OTHER_INPUTS, embedding_inputs, pca_dimensions_of


def load_model(model_path: Path = MODEL_PATH) -> nn.Module:
    state_dict = torch.load(model_path, map_location=DEVICE)
    n_inputs = state_dict["0.weight"].shape[1]  # binned spectrum + metadata, plus the DreaMS columns for a combined model
    model = build_model(n_inputs)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()  # turns dropout off
    return model


def uses_dreams_embedding(model: nn.Module) -> bool:
    """True for a model of train_MLP_combined.py, which also reads the DreaMS embedding (full or PCA-reduced)."""
    return model[0].in_features > N_OTHER_INPUTS


def spectra_to_arrays(spectra: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(peak_bins, peak_intensities, metadata) for a frame with the peak lists and the METADATA_COLUMNS."""
    n_spectra = spectra.height
    peak_bins = np.zeros((n_spectra, MAX_PEAKS), dtype=np.int16)
    peak_intensities = np.zeros((n_spectra, MAX_PEAKS), dtype=np.float16)

    mz_lists = spectra["ms2_mzs"].to_list()
    intensity_lists = spectra["ms2_normalized_intensities"].to_list()
    for i in range(n_spectra):
        bins, sqrt_intensities = spectrum_to_peaks(np.array(mz_lists[i]), np.array(intensity_lists[i]))
        peak_bins[i] = bins
        peak_intensities[i] = sqrt_intensities

    metadata = metadata_features(spectra)
    return peak_bins, peak_intensities, metadata


@torch.no_grad()
def predict_probabilities(
    model: nn.Module, peak_bins: np.ndarray, peak_intensities: np.ndarray, metadata: np.ndarray,
    embeddings: np.ndarray | None = None, has_embedding: np.ndarray | None = None,
) -> np.ndarray:
    """Probability of each fingerprint bit, shape (n_spectra, N_BITS), float32.

    embeddings (n_spectra, EMBEDDING_SIZE) and has_embedding (n_spectra,) bool: only for a combined model
    (uses_dreams_embedding), which needs them.
    """
    probabilities = []
    for start in range(0, len(peak_bins), BATCH_SIZE):
        end = start + BATCH_SIZE
        inputs = build_inputs(peak_bins[start:end], peak_intensities[start:end], metadata[start:end])
        if uses_dreams_embedding(model):
            n_pca_dimensions = pca_dimensions_of(model[0].in_features)  # None: the full embedding
            extra_inputs = embedding_inputs(embeddings[start:end], has_embedding[start:end], n_pca_dimensions)
            inputs = torch.cat([inputs, extra_inputs], dim=1)
        probabilities.append(torch.sigmoid(model(inputs)).cpu().numpy())
    return np.concatenate(probabilities)
