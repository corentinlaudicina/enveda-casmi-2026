"""SHAP analysis of the fingerprint MLP: which inputs (m/z bins, metadata) drive its predictions.

Method: expected gradients, the SHAP estimator of shap.GradientExplainer, written out in torch:
    attribution_i(x) = E over (reference x' from BACKGROUND, alpha ~ U(0, 1)) of (x_i - x'_i) * df/dx_i at x' + alpha (x - x')
The attributions of a spectrum sum (up to sampling noise) to f(x) - mean f(background): they say how each input moves
f away from its value on an average training spectrum. An empty bin can get an attribution too: "no peak here,
where the background spectra had one".

Two quantities f are explained:
1. retrieval score: soft Tanimoto between the predicted probabilities and the spectrum's true fingerprint
   (the "tanimoto" scorer of rank_candidates.py). One number per spectrum, so one global picture of the model.
   Its f(background) is the score of the background predictions against the explained spectrum's fingerprint:
   attributions explain why this spectrum's prediction matches its molecule better than an average prediction does.
2. the logit of a few individual fingerprint bits (well predicted and frequent): which fragments the model reads
   for a given substructure.

Spectra explained: N_EXPLAINED held-out enveda-180 validation spectra and N_EXPLAINED enveda-np-examples spectra
(natural products, never trained on). Background: N_BACKGROUND random training spectra.

Outputs:
- figures/shap_mz_<model>.pdf: mean |attribution| vs fragment m/z and vs neutral loss (precursor m/z - fragment m/z)
- figures/shap_metadata_<model>.pdf: mean |attribution| of each metadata feature, and the metadata / peaks shares
- figures/shap_bits_<model>.pdf: signed mean attribution vs m/z for each explained bit, with its substructure
- results/shap/attributions_<model>.npz: all attributions, to explore further

Usage: python shap_analysis.py [model name in models/, default mlp_negatives]
"""

import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import roc_auc_score

from build_spectrum_arrays import BIN_WIDTH, N_BINS, PROJECT_DIR, TRAIN_PATH
from metadata_features import ADDUCT_NAMES, COLLISION_ENERGIES, METADATA_COLUMNS, metadata_features
from morgan_generator import MORGAN_GENERATOR, N_BITS
from spectrum_quality import clean_spectrum_mask
from train_MLP import (
    DEVICE, LIBRARY_DIR, MODEL_DIR, NATURAL_PRODUCT_TEST_LIBRARY, SEED,
    SpectrumBatcher, build_model, split_spectra_by_molecule,
)
from train_MLP_negatives import soft_tanimoto

N_EXPLAINED = 1000  # spectra explained per set (validation, natural products)
N_BACKGROUND = 500  # training spectra used as references
N_DRAWS = 64  # (reference, alpha) samples per explained spectrum
CHUNK_SIZE = 16  # explained spectra per forward/backward pass (CHUNK_SIZE x N_DRAWS rows)
N_BITS_EXPLAINED = 6
MIN_BIT_FREQUENCY = 0.1  # bits explained must be set in at least this fraction of validation molecules
MAX_NEUTRAL_LOSS = 500  # Da, range of the neutral-loss plot

FIGURE_DIR = PROJECT_DIR / "figures"
RESULT_DIR = PROJECT_DIR / "results" / "shap"

METADATA_NAMES = (
    ["precursor_mz", "neutral_mass", "is_positive"]
    + [f"energy {energy:g} eV" for energy in COLLISION_ENERGIES] + ["energy other"]
    + [f"adduct {name}" for name in ADDUCT_NAMES] + ["adduct other"]
)


def expected_gradients(function, inputs: torch.Tensor, targets: torch.Tensor, background: torch.Tensor,
                       generator: torch.Generator) -> np.ndarray:
    """Expected-gradients attributions (n, n_features) of function(inputs, targets) -> (n,), see module docstring."""
    attributions = []
    for start in range(0, len(inputs), CHUNK_SIZE):
        chunk_inputs = inputs[start:start + CHUNK_SIZE]
        chunk_targets = targets[start:start + CHUNK_SIZE]
        n_rows = len(chunk_inputs) * N_DRAWS

        # every explained spectrum repeated N_DRAWS times, each copy with its own reference and alpha
        repeated_inputs = chunk_inputs.repeat_interleave(N_DRAWS, dim=0)
        repeated_targets = chunk_targets.repeat_interleave(N_DRAWS, dim=0)
        reference_rows = torch.randint(len(background), (n_rows,), generator=generator).to(DEVICE)
        references = background[reference_rows]
        alphas = torch.rand(n_rows, 1, generator=generator).to(DEVICE)

        points = references + alphas * (repeated_inputs - references)
        points.requires_grad_(True)
        outputs = function(points, repeated_targets)
        gradients = torch.autograd.grad(outputs.sum(), points)[0]

        draws = (repeated_inputs - references) * gradients
        chunk_attributions = draws.reshape(len(chunk_inputs), N_DRAWS, -1).mean(dim=1)
        attributions.append(chunk_attributions.cpu().numpy())
    return np.concatenate(attributions)


def sum_into_bins(values: np.ndarray, positions: np.ndarray, n_bins: int) -> np.ndarray:
    """Sum values into integer positions 0..n_bins-1 (positions outside are dropped). Both arrays have the same shape."""
    inside = (positions >= 0) & (positions < n_bins)
    totals = np.zeros(n_bins)
    np.add.at(totals, positions[inside], values[inside])
    return totals


def mean_abs_by_mz(attributions: np.ndarray) -> np.ndarray:
    """Mean |attribution| per spectrum in 1-Da m/z bins (1000 values), from the 0.1-Da peak attributions."""
    per_bin = np.abs(attributions[:, :N_BINS]).mean(axis=0)
    return per_bin.reshape(-1, int(round(1 / BIN_WIDTH))).sum(axis=1)


def mean_abs_by_neutral_loss(attributions: np.ndarray, precursor_mzs: np.ndarray) -> np.ndarray:
    """Mean |attribution| per spectrum in 1-Da neutral-loss bins (precursor m/z - bin m/z), 0..MAX_NEUTRAL_LOSS."""
    bin_mzs = (np.arange(N_BINS) + 0.5) * BIN_WIDTH
    neutral_losses = precursor_mzs[:, None] - bin_mzs[None, :]
    loss_bins = np.floor(neutral_losses).astype(np.int64)
    totals = sum_into_bins(np.abs(attributions[:, :N_BINS]), loss_bins, MAX_NEUTRAL_LOSS)
    return totals / len(attributions)


def peak_shares(attributions: np.ndarray, precursor_mzs: np.ndarray) -> dict[str, float]:
    """Share of the total |attribution| going to metadata, the precursor region (+-1.5 Da) and fragments."""
    absolute = np.abs(attributions)
    bin_mzs = (np.arange(N_BINS) + 0.5) * BIN_WIDTH
    is_precursor_region = np.abs(bin_mzs[None, :] - precursor_mzs[:, None]) <= 1.5

    total = absolute.sum()
    metadata = absolute[:, N_BINS:].sum()
    precursor_region = absolute[:, :N_BINS][is_precursor_region].sum()
    return {
        "metadata": metadata / total,
        "precursor region": precursor_region / total,
        "fragments": (total - metadata - precursor_region) / total,
    }


def bit_substructure(bit: int, smiles_list: list[str]) -> str:
    """Most frequent radius-2 Morgan environment (as SMILES) setting this bit, over the given molecules."""
    environments = Counter()
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        additional_output = rdFingerprintGenerator.AdditionalOutput()
        additional_output.AllocateBitInfoMap()
        MORGAN_GENERATOR.GetFingerprint(mol, additionalOutput=additional_output)
        bit_info = additional_output.GetBitInfoMap()
        if bit not in bit_info:
            continue
        atom, radius = bit_info[bit][0]
        if radius == 0:
            environment = Chem.MolFragmentToSmiles(mol, atomsToUse=[atom])
        else:
            bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom)
            atoms = {atom}
            for bond_index in bonds:
                bond = mol.GetBondWithIdx(bond_index)
                atoms.add(bond.GetBeginAtomIdx())
                atoms.add(bond.GetEndAtomIdx())
            environment = Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms), bondsToUse=list(bonds), rootedAtAtom=atom)
        environments[environment] += 1
    return environments.most_common(1)[0][0]


def plot_bit_panel(ax, bit_result: dict):
    """Signed mean attribution (1-Da bins) of one bit's logit, over validation spectra having the bit."""
    mz_axis = np.arange(len(bit_result["mean_by_mz"]))
    ax.bar(mz_axis, bit_result["mean_by_mz"], width=1.0, color="tab:blue")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"bit {bit_result['bit']}: {bit_result['substructure']}\n"
                 f"frequency {bit_result['frequency']:.2f}, AUC {bit_result['auc']:.2f}", fontsize=9)
    ax.set_xlabel("fragment m/z")
    ax.set_ylabel("mean attribution to logit")
    ax.set_xlim(0, 600)


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "mlp_negatives"
    rng = np.random.default_rng(SEED)
    generator = torch.Generator().manual_seed(SEED)

    # same split as training: the validation molecules are never seen by the model
    spectrum_info = pl.scan_parquet(TRAIN_PATH).select(["ingest_lib"] + METADATA_COLUMNS).collect()
    metadata = metadata_features(spectrum_info)
    batcher = SpectrumBatcher(metadata)
    train_indices, validation_indices = split_spectra_by_molecule(batcher.fp_index, spectrum_info["ingest_lib"], rng)
    is_clean = clean_spectrum_mask(TRAIN_PATH)
    train_indices = train_indices[is_clean[train_indices]]
    natural_product_indices = np.flatnonzero((spectrum_info["ingest_lib"] == NATURAL_PRODUCT_TEST_LIBRARY).to_numpy())

    explained_sets = {
        "validation (enveda-180)": np.sort(rng.choice(validation_indices, size=N_EXPLAINED, replace=False)),
        "natural products": np.sort(rng.choice(natural_product_indices, size=min(N_EXPLAINED, len(natural_product_indices)), replace=False)),
    }
    background_indices = np.sort(rng.choice(train_indices, size=N_BACKGROUND, replace=False))

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / f"{model_name}.pt", map_location=DEVICE))
    model.eval()  # no dropout
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    background, _ = batcher.get_batch(background_indices)

    def retrieval_score(inputs, targets):
        probabilities = torch.sigmoid(model(inputs))
        return soft_tanimoto(probabilities, targets.unsqueeze(1)).squeeze(1)

    def background_scores(targets):
        """Per explained spectrum: mean score of the background predictions against its true fingerprint.

        This is the f(background) of the completeness check: along every path, f is scored against the explained
        spectrum's fingerprint, the reference end included.
        """
        with torch.no_grad():
            background_probabilities = torch.sigmoid(model(background))  # (n_background, N_BITS)
            intersection = targets @ background_probabilities.T  # (n, n_background)
            union = targets.sum(dim=1, keepdim=True) + background_probabilities.sum(dim=1)[None, :] - intersection
            return (intersection / union.clamp(min=1e-6)).mean(dim=1).cpu().numpy()

    print(f"model {model_name} | device {DEVICE} | {N_BACKGROUND} background spectra, {N_DRAWS} draws per spectrum")

    # 1. retrieval score, both sets
    results = {}
    for set_name, indices in explained_sets.items():
        inputs, targets = batcher.get_batch(indices)
        attributions = expected_gradients(retrieval_score, inputs, targets, background, generator)
        with torch.no_grad():
            scores = retrieval_score(inputs, targets).cpu().numpy()
        baselines = background_scores(targets)

        # completeness check: attributions should sum to f(x) - mean f(background), up to sampling noise
        gap = attributions.sum(axis=1) - (scores - baselines)
        precursor_mzs = spectrum_info["precursor_mz"].to_numpy()[indices]
        results[set_name] = {
            "indices": indices,
            "attributions": attributions,
            "scores": scores,
            "precursor_mzs": precursor_mzs,
            "shares": peak_shares(attributions, precursor_mzs),
        }
        print(f"\n{set_name}: {len(indices)} spectra | mean retrieval score {scores.mean():.3f}, "
              f"background {baselines.mean():.3f} | completeness gap {np.abs(gap).mean():.4f} "
              f"(mean |f(x) - f(background)| {np.abs(scores - baselines).mean():.3f})")
        shares = results[set_name]["shares"]
        print(f"  share of |attribution|: metadata {shares['metadata']:.1%}, precursor region "
              f"{shares['precursor region']:.1%}, fragments {shares['fragments']:.1%}")

        metadata_importance = np.abs(attributions[:, N_BINS:]).mean(axis=0)
        print("  metadata features by mean |attribution|:")
        for feature in np.argsort(metadata_importance)[::-1][:6]:
            print(f"    {METADATA_NAMES[feature]:<20} {metadata_importance[feature]:.4f}")

        by_mz = mean_abs_by_mz(attributions)
        print("  strongest fragment m/z (1-Da bins):", ", ".join(f"{mz}" for mz in np.argsort(by_mz)[::-1][:15]))
        by_loss = mean_abs_by_neutral_loss(attributions, precursor_mzs)
        print("  strongest neutral losses (1-Da bins, from 2 Da):",
              ", ".join(f"{loss}" for loss in np.argsort(by_loss[2:])[::-1][:15] + 2))

    # 2. individual bits: frequent, well predicted on validation
    validation_indices_explained = explained_sets["validation (enveda-180)"]
    inputs, targets = batcher.get_batch(validation_indices_explained)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(inputs)).cpu().numpy()
    true_bits = targets.cpu().numpy()
    frequency = true_bits.mean(axis=0)
    aucs = np.full(N_BITS, np.nan)
    for bit in np.flatnonzero((frequency >= MIN_BIT_FREQUENCY) & (frequency <= 1 - MIN_BIT_FREQUENCY)):
        aucs[bit] = roc_auc_score(true_bits[:, bit], probabilities[:, bit])
    explained_bits = np.argsort(np.nan_to_num(aucs, nan=0.0))[::-1][:N_BITS_EXPLAINED]

    molecules = pl.read_parquet(LIBRARY_DIR / "molecules.parquet")
    validation_smiles = molecules["normalized_smiles"].gather(np.unique(batcher.fp_index[validation_indices_explained])).to_list()

    print(f"\nbits explained (frequency >= {MIN_BIT_FREQUENCY}, best validation AUC):")
    bit_results = []
    for bit in explained_bits:
        def bit_logit(points, _targets, bit=bit):
            return model(points)[:, bit]

        attributions = expected_gradients(bit_logit, inputs, targets, background, generator)
        has_bit = true_bits[:, bit] == 1
        mean_by_mz = attributions[has_bit, :N_BINS].mean(axis=0).reshape(-1, int(round(1 / BIN_WIDTH))).sum(axis=1)
        bit_result = {
            "bit": int(bit),
            "frequency": frequency[bit],
            "auc": aucs[bit],
            "substructure": bit_substructure(int(bit), validation_smiles),
            "mean_by_mz": mean_by_mz,
            "attributions": attributions,
        }
        bit_results.append(bit_result)
        top_mzs = np.argsort(mean_by_mz)[::-1][:8]
        print(f"  bit {bit:>4} {bit_result['substructure']:<30} frequency {frequency[bit]:.2f} AUC {aucs[bit]:.3f} | "
              f"top m/z: {', '.join(str(mz) for mz in top_mzs)}")

    # figures
    FIGURE_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    validation = results["validation (enveda-180)"]
    natural = results["natural products"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    mz_axis = np.arange(N_BINS // int(round(1 / BIN_WIDTH)))
    axes[0].plot(mz_axis, mean_abs_by_mz(validation["attributions"]), color="tab:blue", linewidth=0.8, label="validation (enveda-180)")
    axes[0].plot(mz_axis, mean_abs_by_mz(natural["attributions"]), color="tab:orange", linewidth=0.8, label="natural products")
    axes[0].set_xlabel("fragment m/z (1-Da bins)")
    axes[0].set_ylabel("mean |attribution|")
    axes[0].set_title("Retrieval score: attribution by fragment m/z")
    axes[0].legend()

    loss_axis = np.arange(MAX_NEUTRAL_LOSS)
    axes[1].plot(loss_axis, mean_abs_by_neutral_loss(validation["attributions"], validation["precursor_mzs"]),
                 color="tab:blue", linewidth=0.8, label="validation (enveda-180)")
    axes[1].plot(loss_axis, mean_abs_by_neutral_loss(natural["attributions"], natural["precursor_mzs"]),
                 color="tab:orange", linewidth=0.8, label="natural products")
    axes[1].set_xlabel("neutral loss = precursor m/z - fragment m/z (1-Da bins)")
    axes[1].set_ylabel("mean |attribution|")
    axes[1].set_title("Retrieval score: attribution by neutral loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"shap_mz_{model_name}.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [3, 1]})
    validation_metadata = np.abs(validation["attributions"][:, N_BINS:]).mean(axis=0)
    natural_metadata = np.abs(natural["attributions"][:, N_BINS:]).mean(axis=0)
    positions = np.arange(len(METADATA_NAMES))
    axes[0].barh(positions - 0.2, validation_metadata, height=0.4, color="tab:blue", label="validation (enveda-180)")
    axes[0].barh(positions + 0.2, natural_metadata, height=0.4, color="tab:orange", label="natural products")
    axes[0].set_yticks(positions, METADATA_NAMES, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("mean |attribution| to the retrieval score")
    axes[0].set_title("Metadata features")
    axes[0].legend()

    share_names = ["metadata", "precursor region", "fragments"]
    validation_shares = [validation["shares"][name] for name in share_names]
    natural_shares = [natural["shares"][name] for name in share_names]
    share_positions = np.arange(len(share_names))
    axes[1].bar(share_positions - 0.2, validation_shares, width=0.4, color="tab:blue")
    axes[1].bar(share_positions + 0.2, natural_shares, width=0.4, color="tab:orange")
    axes[1].set_xticks(share_positions, share_names, rotation=30)
    axes[1].set_ylabel("share of total |attribution|")
    axes[1].set_title("Where the attribution goes")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"shap_metadata_{model_name}.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    plot_bit_panel(axes[0, 0], bit_results[0])
    plot_bit_panel(axes[0, 1], bit_results[1])
    plot_bit_panel(axes[1, 0], bit_results[2])
    plot_bit_panel(axes[1, 1], bit_results[3])
    plot_bit_panel(axes[2, 0], bit_results[4])
    plot_bit_panel(axes[2, 1], bit_results[5])
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"shap_bits_{model_name}.pdf")
    plt.close(fig)

    np.savez_compressed(
        RESULT_DIR / f"attributions_{model_name}.npz",
        validation_indices=validation["indices"],
        validation_attributions=validation["attributions"].astype(np.float32),
        natural_product_indices=natural["indices"],
        natural_product_attributions=natural["attributions"].astype(np.float32),
        background_indices=background_indices,
        explained_bits=explained_bits,
        bit_attributions=np.stack([bit_result["attributions"] for bit_result in bit_results]).astype(np.float32),
        feature_names=np.array([f"mz {i * BIN_WIDTH:.1f}" for i in range(N_BINS)] + METADATA_NAMES),
    )
    print(f"\nSaved figures to {FIGURE_DIR}/shap_*_{model_name}.pdf and attributions to {RESULT_DIR}")


if __name__ == "__main__":
    main()
