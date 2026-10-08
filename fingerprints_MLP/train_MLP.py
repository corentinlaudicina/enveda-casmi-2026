"""Train an MLP that predicts a molecule's Morgan fingerprint (2048 bits) from one binned MS/MS spectrum.

Needs the outputs of build_fingerprint_library.py and build_spectrum_arrays.py in fingerprints_MLP/library/.

- Input: the spectrum's peaks binned into N_BINS bins of 0.1 Da (max sqrt intensity per bin),
  followed by the N_METADATA_FEATURES measurement features of metadata_features.py
- Output: one logit per fingerprint bit; each bit is a yes/no question ("is this substructure present?"),
  so the loss is binary cross-entropy summed over independent bits.
- Split: by molecule (inchikey14), so validation molecules are never seen in training,
- Validation metric: mean Tanimoto similarity between the predicted fingerprint (bits with p > threshold) and the
  true one, for each threshold in THRESHOLDS. Compared to a constant baseline: the k most frequent training bits,
  predicted for every spectrum.
- Data: training spectra from TRAIN_LIBRARIES that pass the quality checks of spectrum_quality.py;
  validation spectra from VALIDATION_LIBRARIES only (enveda-180: clean, test-like, comparable across runs).
  Molecules of enveda-np-examples are never trained on, from any library: they are the natural-product
  test set of evaluate_ranking.py.
- Learning rate: cosine decay from LEARNING_RATE to 0 over all training steps.
- The model of the epoch with the best validation Tanimoto (any threshold) is saved. Validation here is the selection
  half of the validation molecules; the other half is kept for evaluate_ranking.py (split_selection_and_test).
"""

from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

from build_spectrum_arrays import N_BINS, TRAIN_PATH
from metadata_features import METADATA_COLUMNS, N_METADATA_FEATURES, metadata_features
from spectrum_quality import clean_spectrum_mask
from morgan_generator import N_BITS

LIBRARY_DIR = Path(__file__).resolve().parent / "library"
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "mlp_all_libraries.pt"  # the enveda-180-only model is models/mlp.pt

TRAIN_LIBRARIES = [
    "enveda-180", "pluskal_ms2", "riken", "gnps", "massbank", "mona",
    "spectraverse", "msdial", "drug_plus", "masaryk",
]
VALIDATION_LIBRARIES = ["enveda-180"]
NATURAL_PRODUCT_TEST_LIBRARY = "enveda-np-examples"  # its molecules are excluded from training
VALIDATION_FRACTION = 0.05  # fraction of molecules held out
HIDDEN_SIZE = 1024
DROPOUT = 0.2
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
N_EPOCHS = 20
THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]  # bit is predicted on if p > threshold
SEED = 0
 
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Apple GPU
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def build_model(n_inputs: int = N_BINS + N_METADATA_FEATURES) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_inputs, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN_SIZE, N_BITS),  # logits; the sigmoid is inside the loss
    )


def split_spectra_by_molecule(
    fp_index: np.ndarray, ingest_lib: pl.Series, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train, validation) spectrum indices, holding out VALIDATION_FRACTION of the inchikey14s.

    A held-out molecule is removed from training in every library, not only in VALIDATION_LIBRARIES,
    and so are the molecules of NATURAL_PRODUCT_TEST_LIBRARY.
    """
    molecules = pl.read_parquet(LIBRARY_DIR / "molecules.parquet")

    unique_inchikey14 = molecules["inchikey14"].unique().sort().to_numpy()
    n_validation = int(len(unique_inchikey14) * VALIDATION_FRACTION)
    validation_inchikey14 = rng.choice(unique_inchikey14, size=n_validation, replace=False)

    # molecule-level flag (indexed by fp_index), then looked up for every spectrum
    molecule_is_validation = molecules["inchikey14"].is_in(validation_inchikey14).to_numpy()
    spectrum_is_validation = molecule_is_validation[fp_index]

    # molecules of the natural-product test library, wherever their spectra come from
    natural_product_spectra = (ingest_lib == NATURAL_PRODUCT_TEST_LIBRARY).to_numpy()
    natural_product_inchikey14 = molecules["inchikey14"].gather(fp_index[natural_product_spectra]).unique()
    molecule_is_natural_product = molecules["inchikey14"].is_in(natural_product_inchikey14.implode()).to_numpy()
    spectrum_is_natural_product = molecule_is_natural_product[fp_index]

    in_train_libraries = ingest_lib.is_in(TRAIN_LIBRARIES).to_numpy()
    in_validation_libraries = ingest_lib.is_in(VALIDATION_LIBRARIES).to_numpy()

    train_indices = np.flatnonzero(in_train_libraries & ~spectrum_is_validation & ~spectrum_is_natural_product)
    validation_indices = np.flatnonzero(in_validation_libraries & spectrum_is_validation)
    return train_indices, validation_indices


def split_selection_and_test(fp_index: np.ndarray, validation_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (selection, test) spectrum indices: two halves of the validation molecules (by inchikey14).

    The training scripts pick their best epoch on the selection half; evaluate_ranking.py reports on the test half,
    so the reported number is not the one the checkpoint was chosen on. Same halves in every script (own generator).
    """
    molecules = pl.read_parquet(LIBRARY_DIR / "molecules.parquet")
    validation_inchikey14 = molecules["inchikey14"].gather(fp_index[validation_indices])

    unique_inchikey14 = validation_inchikey14.unique().sort().to_numpy()
    rng = np.random.default_rng(SEED)
    test_inchikey14 = rng.choice(unique_inchikey14, size=len(unique_inchikey14) // 2, replace=False)

    is_test = validation_inchikey14.is_in(test_inchikey14).to_numpy()
    return validation_indices[~is_test], validation_indices[is_test]


def build_inputs(peak_bins: np.ndarray, peak_intensities: np.ndarray, metadata: np.ndarray) -> torch.Tensor:
    """Model input on DEVICE: the dense binned spectra followed by the metadata features.

    peak_bins and peak_intensities are (n_spectra, MAX_PEAKS) arrays from spectrum_to_peaks,
    metadata is (n_spectra, N_METADATA_FEATURES) from metadata_features.
    """
    bins = torch.from_numpy(peak_bins.astype(np.int64)).to(DEVICE)
    intensities = torch.from_numpy(peak_intensities.astype(np.float32)).to(DEVICE)

    # dense spectrum: each bin gets the max intensity of the peaks falling into it
    binned_spectra = torch.zeros(len(peak_bins), N_BINS, device=DEVICE)
    binned_spectra.scatter_reduce_(dim=1, index=bins, src=intensities, reduce="amax")

    metadata_tensor = torch.from_numpy(metadata.astype(np.float32)).to(DEVICE)
    return torch.cat([binned_spectra, metadata_tensor], dim=1)


class SpectrumBatcher:
    """Holds the compact peak arrays and metadata features on CPU and builds dense (inputs, targets) tensors for a set of spectra."""

    def __init__(self, metadata: np.ndarray):
        arrays = np.load(LIBRARY_DIR / "spectrum_arrays.npz")
        self.peak_bins = arrays["peak_bins"]
        self.peak_intensities = arrays["peak_intensities"]
        self.fp_index = arrays["fp_index"]
        self.packed_fingerprints = np.load(LIBRARY_DIR / "morgan_fingerprints.npy")
        self.metadata = metadata  # (n_spectra, N_METADATA_FEATURES), same row order as the peak arrays

    def get_batch(self, spectrum_indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = build_inputs(
            self.peak_bins[spectrum_indices],
            self.peak_intensities[spectrum_indices],
            self.metadata[spectrum_indices],
        )

        packed = self.packed_fingerprints[self.fp_index[spectrum_indices]]
        bits = np.unpackbits(packed, axis=1, count=N_BITS)
        targets = torch.from_numpy(bits.astype(np.float32)).to(DEVICE)
        return inputs, targets


def tanimoto(predicted_bits: torch.Tensor, true_bits: torch.Tensor) -> torch.Tensor:
    """Tanimoto similarity per row between two 0/1 matrices: |A and B| / |A or B|."""
    intersection = (predicted_bits * true_bits).sum(dim=1)
    union = ((predicted_bits + true_bits) > 0).float().sum(dim=1)
    return intersection / union.clamp(min=1)


def constant_baseline(batcher: SpectrumBatcher, train_indices: np.ndarray, validation_indices: np.ndarray) -> tuple[int, float]:
    """Best mean validation Tanimoto when predicting the same k most frequent training bits for every spectrum.

    Returns (best k, its Tanimoto). The model has to beat this to show it uses the spectrum at all.
    """
    train_molecules = np.unique(batcher.fp_index[train_indices])
    train_bits = np.unpackbits(batcher.packed_fingerprints[train_molecules], axis=1, count=N_BITS)
    bit_frequency = train_bits.mean(axis=0)
    bits_by_frequency = np.argsort(bit_frequency)[::-1]

    packed = batcher.packed_fingerprints[batcher.fp_index[validation_indices]]
    validation_bits = np.unpackbits(packed, axis=1, count=N_BITS).astype(np.int32)
    n_true_bits = validation_bits.sum(axis=1)

    best_k = 0
    best_tanimoto = 0.0
    for k in range(5, 101, 5):
        predicted = bits_by_frequency[:k]
        intersection = validation_bits[:, predicted].sum(axis=1)
        union = n_true_bits + k - intersection
        mean_tanimoto = (intersection / union).mean()
        if mean_tanimoto > best_tanimoto:
            best_k = k
            best_tanimoto = mean_tanimoto
    return best_k, best_tanimoto


@torch.no_grad()
def evaluate(model: nn.Module, batcher: SpectrumBatcher, indices: np.ndarray, loss_function: nn.Module) -> tuple[float, dict[float, float]]:
    """Mean loss and mean Tanimoto (one per threshold in THRESHOLDS) over the given spectra."""
    model.eval()
    total_loss = 0.0
    total_tanimoto = {threshold: 0.0 for threshold in THRESHOLDS}
    for start in range(0, len(indices), BATCH_SIZE):
        batch_indices = indices[start:start + BATCH_SIZE]
        inputs, targets = batcher.get_batch(batch_indices)
        logits = model(inputs)
        probabilities = torch.sigmoid(logits)

        total_loss += loss_function(logits, targets).item() * len(batch_indices)
        for threshold in THRESHOLDS:
            predicted_bits = (probabilities > threshold).float()
            total_tanimoto[threshold] += tanimoto(predicted_bits, targets).sum().item()
    model.train()

    mean_tanimoto = {threshold: total / len(indices) for threshold, total in total_tanimoto.items()}
    return total_loss / len(indices), mean_tanimoto


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    # per-spectrum columns of train.parquet (no peak lists), in the same row order as the spectrum arrays
    spectrum_info = pl.scan_parquet(TRAIN_PATH).select(["ingest_lib"] + METADATA_COLUMNS).collect()
    metadata = metadata_features(spectrum_info)

    batcher = SpectrumBatcher(metadata)
    train_indices, validation_indices = split_spectra_by_molecule(batcher.fp_index, spectrum_info["ingest_lib"], rng)
    validation_indices, _ = split_selection_and_test(batcher.fp_index, validation_indices)  # test half: evaluate_ranking.py

    # drop the clearly wrong or useless spectra from training (validation is enveda-180: already clean)
    is_clean = clean_spectrum_mask(TRAIN_PATH)
    n_before = len(train_indices)
    train_indices = train_indices[is_clean[train_indices]]
    print(f"quality filter: kept {len(train_indices):,} of {n_before:,} train spectra")
    print(f"device {DEVICE} | {len(train_indices):,} train spectra, {len(validation_indices):,} validation spectra")

    train_model(build_model(), batcher, train_indices, validation_indices, rng, MODEL_PATH)


def train_model(model: nn.Module, batcher: SpectrumBatcher, train_indices: np.ndarray, validation_indices: np.ndarray,
                rng: np.random.Generator, model_path: Path):
    """Train with BCE and cosine decay, saving the model of the epoch with the best validation Tanimoto to model_path.

    batcher can be any object with get_batch, fp_index and packed_fingerprints, like SpectrumBatcher.
    """
    baseline_k, baseline_tanimoto = constant_baseline(batcher, train_indices, validation_indices)
    print(f"constant baseline: top {baseline_k} most frequent bits | validation Tanimoto {baseline_tanimoto:.3f}")

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.BCEWithLogitsLoss()

    # cosine decay of the learning rate, stepped after every batch
    steps_per_epoch = int(np.ceil(len(train_indices) / BATCH_SIZE))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS * steps_per_epoch)

    MODEL_DIR.mkdir(exist_ok=True)
    best_tanimoto = 0.0

    for epoch in range(1, N_EPOCHS + 1):
        shuffled = rng.permutation(train_indices)
        running_loss = 0.0
        n_steps = 0

        for start in range(0, len(shuffled), BATCH_SIZE):
            batch_indices = shuffled[start:start + BATCH_SIZE]
            inputs, targets = batcher.get_batch(batch_indices)

            logits = model(inputs)
            loss = loss_function(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            n_steps += 1
            if n_steps % 500 == 0:
                print(f"  epoch {epoch} step {n_steps:>5} | train loss {running_loss / 500:.4f}")
                running_loss = 0.0

        validation_loss, validation_tanimoto = evaluate(model, batcher, validation_indices, loss_function)
        best_threshold = max(validation_tanimoto, key=validation_tanimoto.get)
        epoch_tanimoto = validation_tanimoto[best_threshold]
        print(
            f"epoch {epoch} | lr {scheduler.get_last_lr()[0]:.1e} | validation loss {validation_loss:.4f} | "
            f"Tanimoto {epoch_tanimoto:.3f} at threshold {best_threshold} "
        )

        if epoch_tanimoto > best_tanimoto:
            best_tanimoto = epoch_tanimoto
            torch.save(model.state_dict(), model_path)
            print(f"  saved model to {model_path}")

    print(f"Best validation Tanimoto {best_tanimoto:.3f} (constant baseline {baseline_tanimoto:.3f})")


if __name__ == "__main__":
    main()
