"""train_MLP.py plus a ranking loss against same-mass look-alikes. Saves models/mlp_negatives.pt.

train_MLP.py only teaches the MLP to get each fingerprint bit right. At retrieval time, though, the prediction has to
beat ~10,000 pool structures of the same mass. So a second loss shows it some of them during training:

- candidates of a spectrum: its true fingerprint + the N_NEGATIVES same-mass pool structures of its molecule
  (library/negatives.npz, from build_negatives.py)
- score of a candidate: soft Tanimoto between the predicted probabilities and its fingerprint (the "tanimoto" scorer
  of rank_candidates.py, used for retrieval)
- ranking loss: cross-entropy of "the true structure is the best", a softmax over the candidates' scores / TEMPERATURE
- total loss = binary cross entropy + RANKING_WEIGHT x ranking loss. Binary cross entropy keeps the bits sensible; 
  the ranking loss teaches the model to separate look-alikes.

Same split, model, data and schedule as train_MLP.py (imported from it). Validation reports the BCE and Tanimoto of
train_MLP.py, and the ranking metrics among each validation molecule's own candidates: ranking loss, top-1 and MRR.
The model of the epoch with the best validation MRR is saved.

Usage: python train_MLP_negatives.py
"""

from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional

from build_spectrum_arrays import TRAIN_PATH
from metadata_features import METADATA_COLUMNS, metadata_features
from morgan_generator import N_BITS
from spectrum_quality import clean_spectrum_mask
from train_MLP import (
    BATCH_SIZE, DEVICE, LEARNING_RATE, LIBRARY_DIR, MODEL_DIR, N_EPOCHS, SEED,
    SpectrumBatcher, build_model, constant_baseline, evaluate, split_spectra_by_molecule,
)

MODEL_PATH = MODEL_DIR / "mlp_negatives.pt"
NEGATIVES_PATH = LIBRARY_DIR / "negatives.npz"
RANKING_WEIGHT = 0.01  # lambda
TEMPERATURE = 0.05  # Tanimoto scores of look-alikes differ by a few hundredths: a small temperature sharpens the softmax

class NegativesBatcher(SpectrumBatcher):
    """SpectrumBatcher that also gives the fingerprints of each spectrum's negatives (those of its molecule)."""

    def __init__(self, metadata: np.ndarray):
        super().__init__(metadata)
        negatives = np.load(NEGATIVES_PATH)
        self.negative_fingerprints = negatives["fingerprints"]  # (n_molecules, N_NEGATIVES, N_BITS / 8), bit-packed
        self.n_negatives = negatives["n_negatives"]
        self.n_slots = self.negative_fingerprints.shape[1]

        # byte value -> its 8 bits, most significant first (the order of np.unpackbits): unpacks on DEVICE
        byte_values = np.arange(256, dtype=np.uint8)[:, None]
        self.byte_to_bits = torch.from_numpy(np.unpackbits(byte_values, axis=1).astype(np.float32)).to(DEVICE)

    def get_negatives(self, spectrum_indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """(negative_bits (n, n_slots, N_BITS) float, is_real (n, n_slots) bool) on DEVICE.

        Rows of molecules with fewer than n_slots negatives are padded with zeros; is_real marks the real ones.
        """
        molecules = self.fp_index[spectrum_indices]
        packed = torch.from_numpy(self.negative_fingerprints[molecules]).to(DEVICE)
        negative_bits = self.byte_to_bits[packed.long()].reshape(len(spectrum_indices), self.n_slots, N_BITS)

        n_negatives = torch.from_numpy(self.n_negatives[molecules].astype(np.int64)).to(DEVICE)
        slots = torch.arange(self.n_slots, device=DEVICE)
        is_real = slots[None, :] < n_negatives[:, None]
        return negative_bits, is_real


def soft_tanimoto(probabilities: torch.Tensor, candidate_bits: torch.Tensor) -> torch.Tensor:
    """Tanimoto between each row's probabilities (n, N_BITS) and each of its candidates (n, n_candidates, N_BITS).

    Same formula as rank_candidates.tanimoto_scores. Returns (n, n_candidates).
    """
    intersection = torch.bmm(candidate_bits, probabilities.unsqueeze(2)).squeeze(2)
    union = probabilities.sum(dim=1, keepdim=True) + candidate_bits.sum(dim=2) - intersection
    return intersection / union.clamp(min=1e-6)


def candidate_scores(logits: torch.Tensor, targets: torch.Tensor, negative_bits: torch.Tensor, is_real: torch.Tensor) -> torch.Tensor:
    """Soft-Tanimoto scores / TEMPERATURE of the candidates, the true structure in column 0. Padded slots get -inf."""
    candidates = torch.cat([targets.unsqueeze(1), negative_bits], dim=1)
    scores = soft_tanimoto(torch.sigmoid(logits), candidates) / TEMPERATURE

    truth_is_real = torch.ones(len(targets), 1, dtype=torch.bool, device=DEVICE)
    is_real_candidate = torch.cat([truth_is_real, is_real], dim=1)
    return scores.masked_fill(~is_real_candidate, float("-inf"))


def ranking_loss(scores: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of "column 0 (the true structure) is the best candidate", averaged over the rows."""
    truth_column = torch.zeros(len(scores), dtype=torch.long, device=DEVICE)
    return functional.cross_entropy(scores, truth_column)


@torch.no_grad()
def evaluate_ranking(model: nn.Module, batcher: NegativesBatcher, indices: np.ndarray) -> dict[str, float]:
    """Mean ranking loss, top-1 and MRR of the true structure among its candidates, over the given spectra."""
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_reciprocal_rank = 0.0
    for start in range(0, len(indices), BATCH_SIZE):
        batch_indices = indices[start:start + BATCH_SIZE]
        inputs, targets = batcher.get_batch(batch_indices)
        negative_bits, is_real = batcher.get_negatives(batch_indices)
        scores = candidate_scores(model(inputs), targets, negative_bits, is_real)

        # rank of the truth = 1 + number of real negatives that score strictly higher (padded slots are -inf)
        ranks = 1 + (scores[:, 1:] > scores[:, :1]).sum(dim=1)
        total_loss += ranking_loss(scores).item() * len(batch_indices)
        total_top1 += (ranks == 1).sum().item()
        total_reciprocal_rank += (1.0 / ranks).sum().item()
    model.train()
    return {
        "ranking loss": total_loss / len(indices),
        "top-1": total_top1 / len(indices),
        "MRR": total_reciprocal_rank / len(indices),
    }


def run_training(batcher_class, model_path: Path, fresh_decoys: bool) -> None:
    """The whole training run, shared by this script and train_MLP_resampled.py.

    batcher_class: NegativesBatcher (the fixed negatives of negatives.npz) or ResampledBatcher (fresh decoys from the bank).
    fresh_decoys: True to draw new decoys for every batch (batcher.draw_negatives), False to use the fixed ones.
    """
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    # per-spectrum columns of train.parquet (no peak lists), in the same row order as the spectrum arrays
    spectrum_info = pl.scan_parquet(TRAIN_PATH).select(["ingest_lib"] + METADATA_COLUMNS).collect()
    metadata = metadata_features(spectrum_info)

    batcher = batcher_class(metadata)
    train_indices, validation_indices = split_spectra_by_molecule(batcher.fp_index, spectrum_info["ingest_lib"], rng)

    # drop the clearly wrong or useless spectra from training (validation is enveda-180: already clean)
    is_clean = clean_spectrum_mask(TRAIN_PATH)
    n_before = len(train_indices)
    train_indices = train_indices[is_clean[train_indices]]
    print(f"quality filter: kept {len(train_indices):,} of {n_before:,} train spectra")
    print(f"device {DEVICE} | {len(train_indices):,} train spectra, {len(validation_indices):,} validation spectra")
    if fresh_decoys:
        decoys = "fresh decoys per spectrum and batch from the bank"
    else:
        decoys = f"{batcher.n_slots} fixed negatives per molecule"
    print(f"ranking loss: {decoys}, weight {RANKING_WEIGHT}, temperature {TEMPERATURE}")

    baseline_k, baseline_tanimoto = constant_baseline(batcher, train_indices, validation_indices)
    print(f"constant baseline: top {baseline_k} most frequent bits | validation Tanimoto {baseline_tanimoto:.3f}")

    model = build_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce_function = nn.BCEWithLogitsLoss()

    # cosine decay of the learning rate, stepped after every batch
    steps_per_epoch = int(np.ceil(len(train_indices) / BATCH_SIZE))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS * steps_per_epoch)

    validation_ranking = evaluate_ranking(model, batcher, validation_indices)
    print(f"untrained model | validation MRR among candidates {validation_ranking['MRR']:.3f} "
          f"(random ranking ~{np.mean(1 / np.arange(1, batcher.n_slots + 2)):.3f})")

    MODEL_DIR.mkdir(exist_ok=True)
    best_mrr = 0.0

    for epoch in range(1, N_EPOCHS + 1):
        shuffled = rng.permutation(train_indices)
        running_bce = 0.0
        running_ranking = 0.0
        n_steps = 0

        for start in range(0, len(shuffled), BATCH_SIZE):
            batch_indices = shuffled[start:start + BATCH_SIZE]
            inputs, targets = batcher.get_batch(batch_indices)
            if fresh_decoys:
                negative_bits, is_real = batcher.draw_negatives(batch_indices, targets, rng)
            else:
                negative_bits, is_real = batcher.get_negatives(batch_indices)

            logits = model(inputs)
            bce = bce_function(logits, targets)
            ranking = ranking_loss(candidate_scores(logits, targets, negative_bits, is_real))
            loss = bce + RANKING_WEIGHT * ranking

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_bce += bce.item()
            running_ranking += ranking.item()
            n_steps += 1
            if n_steps % 500 == 0:
                print(f"  epoch {epoch} step {n_steps:>5} | train BCE {running_bce / 500:.4f} | "
                      f"ranking loss {running_ranking / 500:.3f}", flush=True)
                running_bce = 0.0
                running_ranking = 0.0

        validation_bce, validation_tanimoto = evaluate(model, batcher, validation_indices, bce_function)
        best_threshold = max(validation_tanimoto, key=validation_tanimoto.get)
        validation_ranking = evaluate_ranking(model, batcher, validation_indices)
        print(
            f"epoch {epoch} | lr {scheduler.get_last_lr()[0]:.1e} | validation BCE {validation_bce:.4f} | "
            f"Tanimoto {validation_tanimoto[best_threshold]:.3f} at threshold {best_threshold} | "
            f"ranking loss {validation_ranking['ranking loss']:.3f} | "
            f"top-1 {validation_ranking['top-1']:.3f} | MRR {validation_ranking['MRR']:.3f}"
        )

        if validation_ranking["MRR"] > best_mrr:
            best_mrr = validation_ranking["MRR"]
            torch.save(model.state_dict(), model_path)
            print(f"  saved model to {model_path}")

    print(f"Best validation MRR among candidates {best_mrr:.3f}")


def main():
    run_training(NegativesBatcher, MODEL_PATH, fresh_decoys=False)


if __name__ == "__main__":
    main()
