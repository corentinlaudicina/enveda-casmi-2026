"""Evaluate prvsiyan's public FPNet (spectrum -> fingerprint transformer) with the retrieval evaluation of
evaluate_ranking.py (scenario A), next to the MLP, on the same molecules and the same candidates.

- Query molecules: train_MLP.py's validation split, enveda-180 spectra only, one adduct per molecule
  (choose_query_spectra of evaluate_ranking.py), first N_MOLECULES of them by inchikey14.
- FPNet logits per molecule: pv_fp.molecule_logits on all its spectra, mean of the single model (per-spectrum
  mean) and the merged model (one merged spectrum), instrument INSTRUMENT.
- Candidates: train structures within PPM_TOLERANCE of the neutral mass (CandidateLibrary.candidates_in_mass_window),
  one entry per inchikey14 (the best-scoring SMILES).
- FPNet scores: "dot" = candidate fingerprint @ logits (the notebook's f.z), and "dot/sqrt" = dot / sqrt(bits set).
- MLP scores: likelihood and tanimoto, via evaluate_ranking.rank_queries on the same queries.

The FPNet fingerprints of all train structures are computed once and cached in external/fpnet_prvsiyan/.

Usage: python evaluate_fpnet.py [number of molecules, default 500; at most evaluate_ranking.N_EVAL_MOLECULES]
"""

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # the fingerprints_MLP modules
from build_spectrum_arrays import TRAIN_PATH  # noqa: E402
from evaluate_ranking import (  # noqa: E402
    RESULTS_DIR, TOP_K, build_pool, choose_query_spectra, predict_queries, random_ranking_mrr, rank_array,
    rank_queries, reciprocal_ranks,
)
from metadata_features import METADATA_COLUMNS  # noqa: E402
from predict import load_model  # noqa: E402
from rank_candidates import PPM_TOLERANCE, CandidateLibrary  # noqa: E402
from train_MLP import LIBRARY_DIR, MODEL_DIR, SEED, split_spectra_by_molecule  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
FPNET_DIR = PROJECT_DIR / "external" / "fpnet_prvsiyan"
sys.path.insert(0, str(FPNET_DIR))
import pv_fp  # noqa: E402  (prvsiyan's model code, verbatim from the notebook)
from fingerprint import N_BITS as FPNET_N_BITS, fpnet_fingerprint  # noqa: E402

N_MOLECULES = int(sys.argv[1]) if len(sys.argv) > 1 else 500
MLP_MODEL_PATH = MODEL_DIR / "mlp_all_libraries.pt"
FPNET_CHECKPOINTS = [FPNET_DIR / "fp_single_s2.pt", FPNET_DIR / "fp_merged_m1.pt"]
TRAIN_FINGERPRINT_PATH = FPNET_DIR / "train_fpnet_fingerprints.npy"  # cache, bit-packed, row i <-> fp_index i
INSTRUMENT = "timsTOF"


def dot_scores(logits: np.ndarray, candidate_bits: np.ndarray) -> np.ndarray:
    return candidate_bits @ logits


def dot_over_sqrt_scores(logits: np.ndarray, candidate_bits: np.ndarray) -> np.ndarray:
    n_bits_set = np.maximum(candidate_bits.sum(axis=1), 1.0)
    return dot_scores(logits, candidate_bits) / np.sqrt(n_bits_set)


FPNET_SCORERS = {"dot": dot_scores, "dot/sqrt": dot_over_sqrt_scores}


def load_fpnet() -> tuple:
    """(single models, merged models, device) as pv_fp.molecule_logits expects. MPS if it runs, else CPU."""
    devices = ["mps", "cpu"] if torch.backends.mps.is_available() else ["cpu"]
    for device in devices:
        single, merged, n_bits = pv_fp.load_fp_models([str(path) for path in FPNET_CHECKPOINTS], device)
        assert n_bits == FPNET_N_BITS, f"checkpoint predicts {n_bits} bits, fingerprint.py makes {FPNET_N_BITS}"
        models = (single, merged, device)

        # a tiny fake molecule, to check the device really supports the model
        try:
            test_spectrum = (np.array([100.0, 150.0]), np.array([1.0, 0.5]))
            logits = pv_fp.molecule_logits(models, [test_spectrum], [200.0], ["[M+H]+"], [INSTRUMENT], [20.0], [1.0])
            assert np.all(np.isfinite(logits))
            print(f"FPNet on {device}: {len(single)} single + {len(merged)} merged model(s), {n_bits} bits")
            return models
        except Exception as error:
            print(f"FPNet failed on {device} ({error!r}), trying the next device")
    raise RuntimeError("FPNet runs on no device")


def train_fpnet_fingerprints(train_molecules: pl.DataFrame) -> np.ndarray:
    """Bit-packed FPNet fingerprints of all train structures (fp_index order), computed once then read from the cache."""
    if TRAIN_FINGERPRINT_PATH.exists():
        return np.load(TRAIN_FINGERPRINT_PATH)

    smiles_list = train_molecules["normalized_smiles"].to_list()
    packed = np.zeros((len(smiles_list), (FPNET_N_BITS + 7) // 8), dtype=np.uint8)
    n_failed = 0
    for i, smiles in enumerate(smiles_list):
        bits = fpnet_fingerprint(smiles)
        if bits is None:
            n_failed += 1  # stays all zeros: scores 0
        else:
            packed[i] = np.packbits(bits)
        if (i + 1) % 25_000 == 0:
            print(f"  FPNet fingerprints {i + 1:>7,} / {len(smiles_list):,}")
    print(f"  {n_failed} unparsable SMILES")
    np.save(TRAIN_FINGERPRINT_PATH, packed)
    return packed


def load_peaks(spectrum_indices: np.ndarray) -> pl.DataFrame:
    """Peak lists of the given train.parquet rows: spectrum_index, ms2_mzs, ms2_normalized_intensities."""
    return (
        pl.scan_parquet(TRAIN_PATH)
        .with_row_index("spectrum_index")
        .filter(pl.col("spectrum_index").is_in(spectrum_indices.tolist()))
        .select(pl.col("spectrum_index").cast(pl.Int64), "ms2_mzs", "ms2_normalized_intensities")
        .collect()
    )


def fpnet_molecule_logits(models: tuple, spectra: pl.DataFrame) -> np.ndarray:
    """FPNet logits (FPNET_N_BITS,) of one molecule from all its spectra (rows of spectra)."""
    peak_lists = []
    for mzs, intensities in zip(spectra["ms2_mzs"].to_list(), spectra["ms2_normalized_intensities"].to_list()):
        peak_lists.append((np.array(mzs, dtype=np.float64), np.array(intensities, dtype=np.float64)))

    precursor_mzs = spectra["precursor_mz"].to_list()
    adducts = spectra["adduct"].to_list()
    instruments = [INSTRUMENT] * spectra.height
    # a merged "20,40,60" spectrum has several energies: use their mean, like the notebook
    energies = [float(np.mean(energy_list)) for energy_list in spectra["collision_energy_ev"].to_list()]
    modes = [1.0 if mode == "positive" else -1.0 for mode in spectra["ionization_mode"].to_list()]

    return pv_fp.molecule_logits(models, peak_lists, precursor_mzs, adducts, instruments, energies, modes)


def fpnet_rank(logits: np.ndarray, neutral_mass: float, pool: CandidateLibrary, true_inchikey14: str) -> dict:
    """Rank of the true inchikey14 among the mass-window candidates for each FPNet scorer (None if not a candidate)."""
    result = {}
    for scorer, score_function in FPNET_SCORERS.items():
        ranking = pool.rank(logits, neutral_mass, scorer=score_function, n_bits=FPNET_N_BITS)
        true_position = ranking["inchikey14"].index_of(true_inchikey14)
        result[f"rank_fpnet_{scorer}"] = None if true_position is None else true_position + 1
        result["n_candidates_fpnet"] = ranking.height
    return result


def summary_row(method: str, ranks: pl.Series) -> dict:
    integer_ranks = rank_array(ranks)
    return {
        "method": method,
        "MRR@25": reciprocal_ranks(ranks).mean(),
        "top-1": np.mean(integer_ranks <= 1),
        "top-25": np.mean(integer_ranks <= TOP_K),
    }


def main():
    rng = np.random.default_rng(SEED)  # same seed and first use as train_MLP.py: same validation molecules

    print("Loading spectra and split")
    spectrum_info = pl.scan_parquet(TRAIN_PATH).select(["ingest_lib", "inchikey14"] + METADATA_COLUMNS).collect()
    arrays = dict(np.load(LIBRARY_DIR / "spectrum_arrays.npz"))
    _, validation_indices = split_spectra_by_molecule(arrays["fp_index"], spectrum_info["ingest_lib"], rng)
    n_validation_molecules = spectrum_info[validation_indices]["inchikey14"].n_unique()

    # the MLP evaluation's query molecules (one adduct each), restricted to the first N_MOLECULES by inchikey14
    all_queries = choose_query_spectra(spectrum_info, validation_indices)
    chosen_inchikey14 = all_queries["inchikey14"].unique().sort().head(N_MOLECULES)
    queries = all_queries.filter(pl.col("inchikey14").is_in(chosen_inchikey14.implode())).sort("inchikey14", "spectrum_index")
    print(f"{chosen_inchikey14.len()} query molecules, {queries.height} spectra "
          f"(validation set: {n_validation_molecules:,} enveda-180 molecules)")

    # metadata and peaks of the query spectra
    spectrum_indices = queries["spectrum_index"].to_numpy()
    query_spectra = (
        queries
        .with_columns(spectrum_info[spectrum_indices].select(METADATA_COLUMNS))
        .join(load_peaks(spectrum_indices), on="spectrum_index", how="left")
    )

    # candidate pools: same structures, sorted by mass, with Morgan (MLP) or FPNet fingerprints
    train_molecules = pl.read_parquet(LIBRARY_DIR / "molecules.parquet")
    print("FPNet fingerprints of the train structures")
    fpnet_pool = CandidateLibrary(train_molecules, train_fpnet_fingerprints(train_molecules))
    mlp_pool = build_pool([(train_molecules, np.load(LIBRARY_DIR / "morgan_fingerprints.npy"))])

    # FPNet
    models = load_fpnet()
    start_time = time.time()
    fpnet_results = []
    for (true_inchikey14,), spectra in query_spectra.group_by("inchikey14", maintain_order=True):
        logits = fpnet_molecule_logits(models, spectra)
        result = {"inchikey14": true_inchikey14}
        result.update(fpnet_rank(logits, spectra["neutral_mass"].mean(), fpnet_pool, true_inchikey14))
        fpnet_results.append(result)
        if len(fpnet_results) % 100 == 0:
            print(f"  FPNet {len(fpnet_results)} molecules, {(time.time() - start_time) / len(fpnet_results):.2f} s/molecule")
    fpnet_seconds = time.time() - start_time
    fpnet_results = pl.DataFrame(fpnet_results)

    # MLP on the same queries
    start_time = time.time()
    mlp = load_model(MLP_MODEL_PATH)
    probabilities = predict_queries(mlp, queries, spectrum_info, arrays)
    mlp_results = rank_queries(queries, probabilities, mlp_pool)
    mlp_seconds = time.time() - start_time

    results = mlp_results.join(fpnet_results, on="inchikey14", how="inner")
    n_molecules = results.height
    n_mismatch = (results["n_candidates"] != results["n_candidates_fpnet"]).sum()
    if n_mismatch > 0:
        print(f"WARNING: {n_mismatch} molecules have different candidate counts for FPNet and MLP")

    RESULTS_DIR.mkdir(exist_ok=True)
    results_path = RESULTS_DIR / f"fpnet_vs_mlp_{n_molecules}.csv"
    results.write_csv(results_path)
    print(f"Per-molecule results in {results_path}")

    random_mrr = np.mean([
        random_ranking_mrr(n, found) for n, found in zip(results["n_candidates"], results["true_in_window"])
    ])
    summary = pl.DataFrame([
        summary_row("FPNet, dot", results["rank_fpnet_dot"]),
        summary_row("FPNet, dot / sqrt(bits)", results["rank_fpnet_dot/sqrt"]),
        summary_row(f"MLP {MLP_MODEL_PATH.stem}, likelihood", results["rank_likelihood"]),
        summary_row(f"MLP {MLP_MODEL_PATH.stem}, tanimoto", results["rank_tanimoto"]),
        {"method": "random", "MRR@25": random_mrr, "top-1": None, "top-25": None},
    ])

    print(f"\n=== enveda validation, train pool, {PPM_TOLERANCE:g} ppm ({n_molecules} molecules) ===")
    print(f"True structure in the window: {results['true_in_window'].mean():.1%} | "
          f"candidates per molecule: median {results['n_candidates'].median():.0f}, mean {results['n_candidates'].mean():.0f}")
    with pl.Config(tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True, float_precision=3):
        print(summary)

    print(f"\nTiming: FPNet {fpnet_seconds:.0f} s ({fpnet_seconds / n_molecules:.2f} s/molecule), MLP {mlp_seconds:.0f} s")
    print(f"Estimated FPNet time for all {n_validation_molecules:,} validation molecules: "
          f"{fpnet_seconds / n_molecules * n_validation_molecules / 60:.0f} min")


if __name__ == "__main__":
    main()
