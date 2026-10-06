"""Use the trained MLP: spectra in, fingerprint bit probabilities out.

- load_model: the saved weights of train_MLP.py
- spectra_to_arrays: a polars frame of spectra (train or test format) -> the compact arrays the model reads
- predict_probabilities: those arrays -> (n_spectra, N_BITS) probabilities
"""

from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

from build_spectrum_arrays import MAX_PEAKS, spectrum_to_peaks
from metadata_features import metadata_features
from train_MLP import BATCH_SIZE, DEVICE, MODEL_PATH, build_inputs, build_model


def load_model(model_path: Path = MODEL_PATH) -> nn.Module:
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()  # turns dropout off
    return model


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
    model: nn.Module, peak_bins: np.ndarray, peak_intensities: np.ndarray, metadata: np.ndarray
) -> np.ndarray:
    """Probability of each fingerprint bit, shape (n_spectra, N_BITS), float32."""
    probabilities = []
    for start in range(0, len(peak_bins), BATCH_SIZE):
        end = start + BATCH_SIZE
        inputs = build_inputs(peak_bins[start:end], peak_intensities[start:end], metadata[start:end])
        probabilities.append(torch.sigmoid(model(inputs)).cpu().numpy())
    return np.concatenate(probabilities)
