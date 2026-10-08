"""DreaMS spectrum embeddings with only torch and numpy: no `dreams` package, no pinned dependencies.

A re-implementation of the forward pass of DreaMS's pre-trained embedding model (`PreTrainedModel.from_name(DREAMS_EMBEDDING)`, a ContrastiveHead on the DreaMS backbone), for running where the dreams package can't be installed (Kaggle, Python 3.13). It reproduces embed_dreams.py: same preprocessing, same layers, same weights (dreams_embedding_weights.pt, extracted from the official checkpoint by extract_dreams_weights.py). verify_dreams_lite.py checks it against the stored embeddings. Written by Claude.

Model, with the checkpoint's settings:
- input: one row per peak, (m/z, intensity); row 0 is the precursor (intensity 1.1), then the 100 strongest peaks in their original order, intensities relative to the strongest, zero rows as padding (101 rows)
- each row -> 1024 numbers: a small layer on (m/z / 1000, intensity) -> 44, and Fourier features of the m/z (cos and sin at 5,997 fixed frequencies) through a 5-layer feed-forward network -> 980
- 7 pre-norm transformer layers (8 heads, no biases). The attention gets an extra bias from the m/z: the sum of the Fourier-feature differences between the two peaks
- output: a linear layer on the precursor row's final vector, scaled to unit length
"""

from math import ceil, pi

import numpy as np
import torch
from torch import nn
from torch.nn import functional

N_HIGHEST_PEAKS = 100
PRECURSOR_INTENSITY = 1.1
MAX_MZ = 1000.0  # DataFormatA.max_mz: m/z scale of the peak layer and top Fourier frequency
MIN_MZ_PERIOD = 1e-4  # DataFormatA.max_tbxic_stdev: smallest Fourier period
D_PEAK = 44
D_FOURIER = 980
D_MODEL = D_PEAK + D_FOURIER  # 1024
FOURIER_HIDDEN = 512
FOURIER_DEPTH = 5
N_LAYERS = 7
N_HEADS = 8


def preprocess(mzs, intensities, precursor_mz: float) -> np.ndarray:
    """One spectrum -> (N_HIGHEST_PEAKS + 1, 2) float32 array of (m/z, intensity) rows, as DreaMS's SpectrumPreprocessor."""
    mzs = np.asarray(mzs, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)

    # the strongest peaks, kept in their original (m/z) order
    strongest = np.sort(np.argsort(intensities)[-N_HIGHEST_PEAKS:])
    mzs = mzs[strongest]
    intensities = intensities[strongest] / intensities[strongest].max()

    spectrum = np.zeros((N_HIGHEST_PEAKS + 1, 2), dtype=np.float32)
    spectrum[0] = [precursor_mz, PRECURSOR_INTENSITY]
    spectrum[1:len(mzs) + 1, 0] = mzs
    spectrum[1:len(mzs) + 1, 1] = intensities
    return spectrum


def fourier_frequencies() -> torch.Tensor:
    """The initial frequencies of the "lin_float_int" strategy: fine ones for the decimals, 1/i for the integer part.

    Only their number matters here: the checkpoint's frequencies moved during fine-tuning, and load_embedder
    replaces these with them.
    """
    fine = [1 / (MIN_MZ_PERIOD * i) for i in range(2, ceil(1 / MIN_MZ_PERIOD), 2)]
    coarse = [1 / i for i in range(2, ceil(MAX_MZ))]
    return torch.tensor(fine + coarse)


class TransformerLayer(nn.Module):
    """Pre-norm layer: x + attention(norm(x)), then x + feed_forward(norm(x)). No biases in the projections."""

    def __init__(self):
        super().__init__()
        self.attention_norm = nn.LayerNorm(D_MODEL)
        self.qkvo_weights = nn.Parameter(torch.empty(4 * D_MODEL, D_MODEL))  # query, key, value, output stacked
        self.feed_forward_norm = nn.LayerNorm(D_MODEL)
        self.feed_forward_in = nn.Linear(D_MODEL, 4 * D_MODEL, bias=False)
        self.feed_forward_out = nn.Linear(4 * D_MODEL, D_MODEL, bias=False)

    def attention(self, x: torch.Tensor, attention_bias: torch.Tensor, is_padding: torch.Tensor) -> torch.Tensor:
        n_spectra, n_rows, _ = x.shape
        head_dim = D_MODEL // N_HEADS
        query, key, value = functional.linear(x, self.qkvo_weights[:3 * D_MODEL]).chunk(3, dim=-1)

        # (n_spectra, n_rows, D_MODEL) -> (n_spectra, N_HEADS, n_rows, head_dim)
        query = query.reshape(n_spectra, n_rows, N_HEADS, head_dim).transpose(1, 2)
        key = key.reshape(n_spectra, n_rows, N_HEADS, head_dim).transpose(1, 2)
        value = value.reshape(n_spectra, n_rows, N_HEADS, head_dim).transpose(1, 2)

        weights = query @ key.transpose(-2, -1) * head_dim ** -0.5 + attention_bias.unsqueeze(1)
        # As in DreaMS: the padding rows are masked as queries (their whole row), not as keys
        weights = weights.masked_fill(is_padding.unsqueeze(1).unsqueeze(-1), -1e9)
        weights = torch.softmax(weights, dim=-1)

        output = (weights @ value).transpose(1, 2).reshape(n_spectra, n_rows, D_MODEL)
        return functional.linear(output, self.qkvo_weights[3 * D_MODEL:])

    def forward(self, x: torch.Tensor, attention_bias: torch.Tensor, is_padding: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), attention_bias, is_padding)
        x = x + self.feed_forward_out(functional.relu(self.feed_forward_in(self.feed_forward_norm(x))))
        return x


class DreamsEmbedder(nn.Module):
    """Preprocessed spectra (n_spectra, 101, 2) -> unit-length embeddings (n_spectra, 1024)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("frequencies", fourier_frequencies().unsqueeze(0))  # (1, 5997)
        self.peak_layer = nn.Linear(2, D_PEAK)

        fourier_layers = []
        n_inputs = 2 * self.frequencies.shape[1]
        for depth in range(FOURIER_DEPTH):
            n_outputs = D_FOURIER if depth == FOURIER_DEPTH - 1 else FOURIER_HIDDEN
            fourier_layers += [nn.Linear(n_inputs, n_outputs), nn.ReLU()]
            n_inputs = n_outputs
        self.fourier_network = nn.Sequential(*fourier_layers)

        self.layers = nn.ModuleList([TransformerLayer() for _ in range(N_LAYERS)])
        self.final_norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        mzs = spectra[..., [0]]  # (n_spectra, n_rows, 1)
        is_padding = spectra[..., 0] == 0

        peak_vectors = functional.relu(self.peak_layer(spectra / torch.tensor([MAX_MZ, 1.0], device=spectra.device)))
        angles = 2 * pi * mzs @ self.frequencies
        fourier_vectors = self.fourier_network(torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1))
        x = torch.cat([peak_vectors, fourier_vectors], dim=-1)

        # DreaMS adds, between rows i and j, the sum over the 980 numbers of (fourier_i - fourier_j) = sum_i - sum_j.
        # Computed from the sums, without DreaMS's (n_rows x n_rows x 980) tensor of differences.
        fourier_sums = fourier_vectors.sum(dim=-1)
        attention_bias = fourier_sums.unsqueeze(2) - fourier_sums.unsqueeze(1)

        for layer in self.layers:
            x = layer(x, attention_bias, is_padding)
        x = self.final_norm(x)

        embeddings = self.head(x[:, 0])  # the precursor row
        return embeddings / embeddings.norm(dim=1, keepdim=True)


def load_embedder(weights_path, device: str = "cpu") -> DreamsEmbedder:
    embedder = DreamsEmbedder()
    embedder.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return embedder.to(device).eval()


@torch.inference_mode()
def embed(embedder: DreamsEmbedder, spectra: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
    """Preprocessed spectra (from preprocess) -> (n_spectra, 1024) float32 unit-length embeddings."""
    device = next(embedder.parameters()).device
    embeddings = []
    for start in range(0, len(spectra), batch_size):
        batch = torch.from_numpy(np.stack(spectra[start:start + batch_size])).to(device)
        embeddings.append(embedder(batch).float().cpu().numpy())
    return np.concatenate(embeddings)
