"""train_MLP_resampled.py with the binned spectrum AND the DreaMS embedding as input. Saves models/mlp_combined_seed<N>.pt.

DreaMS keeps only the 100 strongest peaks and summarises them; the binned spectrum keeps every peak position.
Giving the model both lets it use whichever helps.

- Input: [binned spectrum (N_BINS), metadata features (N_METADATA_FEATURES), DreaMS embedding columns, has-embedding
  flag (1)]. The embedding columns are either the full embedding x EMBEDDING_SCALE (1024 columns), or its first k
  PCA coordinates (dreaMS/fit_dreams_pca.py, k <= 256), scaled so that their mean square is 1 like the other
  features: fewer, denser columns that may weigh less against the binned spectrum in the first layer. They are built
  by dreams_inputs.py, shared with predict.py.
- Spectra without an embedding (rare adducts, skipped by dreaMS/embed_train.py) are kept: their embedding is zeros
  and their flag 0. A test spectrum of a rare adduct can then be predicted too.
- Embedding dropout: during training, each spectrum loses its embedding (zeros, flag 0) with probability
  EMBEDDING_DROPOUT, so the model also learns to do without it instead of leaning on it entirely. Validation and
  prediction use every embedding there is.
- Loss: binary cross-entropy + ranking loss against fresh same-mass decoys drawn from the decoy bank at every batch,
  as in train_MLP_resampled.py (decoys identical to the true fingerprint are masked)
- Same train and validation spectra, model size, schedule, quality filter, validation and training seeds as
  train_MLP_resampled.py (prepare_training and train_with_ranking_loss of train_MLP_negatives.py): the two differ
  only by the embedding input, so mlp_combined_seed<N>.pt vs mlp_resampled_seed<N>.pt measures the embedding alone.
  For that, every spectrum dreaMS/embed_train.py selected must be embedded: the script stops otherwise (with missing
  parts, flag 0 would mark whole libraries, train.parquet being sorted by library).
- At the end, the validation MRR of the saved model is computed again with every embedding removed: how much the
  model relies on the embedding.

About 10 GB of RAM: peak arrays 1 GB, embeddings 5 GB, fixed negatives 2.3 GB, decoy bank 1 GB.

Usage: python train_MLP_combined.py [training seed, default 0] [PCA dimensions, default: full embedding]
  e.g. python train_MLP_combined.py 0 256 -> models/mlp_combined_pca256_seed0.pt
"""

import sys
from functools import partial

import numpy as np
import polars as pl
import torch

from dreams_inputs import EMBEDDING_SIZE, PCA_PATH, embedding_inputs, n_inputs
from train_MLP import DEVICE, build_model
from train_MLP_dreams import EMBEDDINGS_DIR, load_embeddings
from train_MLP_negatives import (
    evaluate_ranking, model_path, prepare_training, train_with_ranking_loss, training_seed_from_command_line,
)
from train_MLP_resampled import ResampledBatcher

MODEL_NAME = "mlp_combined"
EMBEDDING_DROPOUT = 0.15  # probability that a training spectrum is shown without its embedding


class CombinedBatcher(ResampledBatcher):
    """ResampledBatcher whose inputs also carry each spectrum's DreaMS embedding and has-embedding flag."""

    def __init__(self, metadata: np.ndarray, n_pca_dimensions: int | None):
        super().__init__(metadata)
        self.n_pca_dimensions = n_pca_dimensions  # None: the full embedding
        self.embeddings, self.has_embedding = load_embeddings(len(metadata))
        check_all_selected_spectra_embedded(self.has_embedding)
        self.embeddings_switched_on = True  # False: every spectrum is given without its embedding (ablation)

    def get_batch(self, spectrum_indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Validation batch: every embedding there is (unless switched off)."""
        use_embedding = self.has_embedding[spectrum_indices] & self.embeddings_switched_on
        return self.batch_with_embeddings(spectrum_indices, use_embedding)

    def get_training_batch(self, spectrum_indices: np.ndarray, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """Training batch: each embedding is dropped with probability EMBEDDING_DROPOUT."""
        is_kept = rng.random(len(spectrum_indices)) >= EMBEDDING_DROPOUT
        use_embedding = self.has_embedding[spectrum_indices] & is_kept
        return self.batch_with_embeddings(spectrum_indices, use_embedding)

    def batch_with_embeddings(self, spectrum_indices: np.ndarray, use_embedding: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = super().get_batch(spectrum_indices)  # binned spectrum + metadata
        extra_inputs = embedding_inputs(self.embeddings[spectrum_indices], use_embedding, self.n_pca_dimensions)
        return torch.cat([inputs, extra_inputs], dim=1), targets


def check_all_selected_spectra_embedded(has_embedding: np.ndarray) -> None:
    """Stop if some spectrum that dreaMS/embed_train.py selected has no embedding yet (an unfinished part)."""
    selected_rows = pl.read_parquet(EMBEDDINGS_DIR / "selected_rows.parquet")["row_id"].to_numpy()
    n_missing = int((~has_embedding[selected_rows]).sum())
    if n_missing > 0:
        raise SystemExit(f"{n_missing:,} selected spectra have no DreaMS embedding yet: finish dreaMS/embed_train.py first")
    print(f"every selected spectrum is embedded; {(~has_embedding).sum():,} spectra (rare adducts) have none: flag 0")


def pca_dimensions_from_command_line() -> int | None:
    """The PCA dimensions given after the training seed (python train_MLP_combined.py 0 256), None by default."""
    return int(sys.argv[2]) if len(sys.argv) > 2 else None


def main():
    training_seed = training_seed_from_command_line()
    n_pca_dimensions = pca_dimensions_from_command_line()
    batcher_factory = partial(CombinedBatcher, n_pca_dimensions=n_pca_dimensions)
    batcher, train_indices, selection_indices, rng = prepare_training(batcher_factory, training_seed)
    print(f"with a DreaMS embedding: {batcher.has_embedding[train_indices].mean():.1%} of train spectra, "
          f"{batcher.has_embedding[selection_indices].mean():.1%} of validation spectra | "
          f"embedding dropout {EMBEDDING_DROPOUT}")

    if n_pca_dimensions is None:
        print(f"embedding input: the full embedding ({EMBEDDING_SIZE} columns)")
        model_name = MODEL_NAME
    else:
        pca = np.load(PCA_PATH)
        kept = pca["explained_variance"][:n_pca_dimensions].sum() / pca["total_variance"]
        print(f"embedding input: {n_pca_dimensions} PCA coordinates ({kept:.1%} of the embedding variance)")
        model_name = f"{MODEL_NAME}_pca{n_pca_dimensions}"

    path = model_path(model_name, training_seed)
    model = build_model(n_inputs(n_pca_dimensions))
    best_mrr = train_with_ranking_loss(model, batcher, train_indices, selection_indices, rng, path, fresh_decoys=True)

    # how much the saved model relies on the embedding
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    batcher.embeddings_switched_on = False
    without_embedding = evaluate_ranking(model, batcher, selection_indices)
    print(f"Validation MRR of the saved model: {best_mrr:.3f} with the embeddings, "
          f"{without_embedding['MRR']:.3f} without any")


if __name__ == "__main__":
    main()
