"""train_MLP_negatives.py with fresh decoys at every batch. Saves models/mlp_resampled_seed<N>.pt.

train_MLP_negatives.py shows each molecule the same 32 negatives at every epoch (library/negatives.npz), so the
model can memorise them instead of learning to separate look-alikes in general. Here, every time a spectrum is in a
batch, N_NEGATIVES new decoys are drawn at random from its molecule's mass window (± PPM_TOLERANCE) in the decoy bank
(library/negative_bank.npz, from build_negative_bank.py: ~4M pool structures, sorted by mass).

- decoys whose fingerprint equals the true one (the molecule itself, its stereoisomers, collisions) are masked
- a window with fewer than N_NEGATIVES bank structures gives all of them, each once, as its real decoys (a larger
  window is drawn from with replacement: repeats are rare there); an empty window gives none, and that spectrum is
  left out of the ranking loss (see ranking_loss in train_MLP_negatives.py)
- validation is unchanged: the fixed negatives of negatives.npz, so the MRR is comparable with train_MLP_negatives.py

Same split, model, data, schedule, loss and training seeds as train_MLP_negatives.py (prepare_training and
train_with_ranking_loss there): the two differ only by the decoys.

Usage: python train_MLP_resampled.py [training seed, default 0]
"""

import numpy as np
import polars as pl
import torch

from morgan_generator import N_BITS
from build_negatives import N_NEGATIVES
from rank_candidates import mass_windows
from train_MLP import DEVICE, LIBRARY_DIR, build_model
from train_MLP_negatives import (
    NegativesBatcher, model_path, prepare_training, train_with_ranking_loss, training_seed_from_command_line,
)

MODEL_NAME = "mlp_resampled"
BANK_PATH = LIBRARY_DIR / "negative_bank.npz"


class ResampledBatcher(NegativesBatcher):
    """NegativesBatcher (fixed negatives, for validation) that can also draw fresh decoys from the bank."""

    def __init__(self, metadata: np.ndarray):
        super().__init__(metadata)
        bank = np.load(BANK_PATH)
        self.bank_fingerprints = bank["fingerprints"]  # (n_bank, N_BITS / 8), bit-packed, sorted by mass

        # each molecule's mass window in the bank: rows window_start .. window_start + window_size - 1
        molecule_masses = pl.read_parquet(LIBRARY_DIR / "molecules.parquet")["exact_mass"].to_numpy()
        self.window_start, window_end = mass_windows(bank["masses"], molecule_masses)
        self.window_size = window_end - self.window_start
        print(f"decoy bank: {len(self.bank_fingerprints):,} structures | window size per molecule: median "
              f"{np.median(self.window_size):.0f}, below {N_NEGATIVES}: {np.mean(self.window_size < N_NEGATIVES):.1%}")

    def draw_negatives(self, spectrum_indices: np.ndarray, targets: torch.Tensor,
                       rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """(negative_bits (n, N_NEGATIVES, N_BITS) float, is_real (n, N_NEGATIVES) bool) on DEVICE, freshly drawn."""
        molecules = self.fp_index[spectrum_indices]
        starts = self.window_start[molecules][:, None]
        sizes = self.window_size[molecules][:, None]

        # uniform random rows of each window (with replacement)
        offsets = np.floor(rng.random((len(spectrum_indices), N_NEGATIVES)) * sizes).astype(np.int64)
        # a window smaller than N_NEGATIVES: every row once instead (slots beyond its size are masked below)
        slots = np.arange(N_NEGATIVES)[None, :]
        offsets = np.where(sizes < N_NEGATIVES, slots, offsets)
        rows = np.minimum(starts + offsets, len(self.bank_fingerprints) - 1)
        packed = torch.from_numpy(self.bank_fingerprints[rows]).to(DEVICE)
        negative_bits = self.byte_to_bits[packed.long()].reshape(len(spectrum_indices), N_NEGATIVES, N_BITS)

        # real = inside a window that has room for this slot, and not a twin of the true fingerprint
        in_window = torch.from_numpy(slots < sizes).to(DEVICE)
        is_twin = (negative_bits == targets.unsqueeze(1)).all(dim=2)
        return negative_bits, in_window & ~is_twin


def main():
    training_seed = training_seed_from_command_line()
    batcher, train_indices, selection_indices, rng = prepare_training(ResampledBatcher, training_seed)
    train_with_ranking_loss(build_model(), batcher, train_indices, selection_indices, rng,
                            model_path(MODEL_NAME, training_seed), fresh_decoys=True)


if __name__ == "__main__":
    main()
