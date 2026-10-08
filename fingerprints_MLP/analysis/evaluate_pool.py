"""Step 5: recall, crowding and MLP retrieval (MRR@25) with the train + PubChem pool of build_pool.py.

Query molecules: the validation molecules of evaluate_ranking.py (train_MLP.py's split, enveda-180 spectra,
one adduct each), the first N_MOLECULES by inchikey14. Three pools, all read from external/pool/:
- "train only":                     the train structures (the answer is always there: ranking alone)
- "(train - queries) + PubChem":    the query molecules removed from the train side, so a query is found only if
                                    PubChem has it: the realistic case for novel test molecules
- "train + PubChem":                everything (the answer is always there: cost of the crowding)

Per query molecule: candidates = pool rows within PPM_TOLERANCE of its neutral mass; their Morgan fingerprints are
computed on the fly (morgan_generator.py, as for the MLP's targets) and scored by the MLP's scorers.
A candidate is the truth if its METRIC key (tautomer-canonical, metric_key.py) equals the truth's. Tautomers only move
hydrogens and bond orders, so they share the heavy-atom skeleton (skeleton_hash): the slow metric key is computed
only for candidates with the truth's skeleton (PubChem has thousands of same-formula isomers, but few same-skeleton ones).
Candidates are deduplicated on the pool key, keeping each key's best score.

Results are saved per model (results/pool_evaluation_<n>_<model name>.csv). For another model than BASELINE_MODEL,
the MRRs are compared with BASELINE_MODEL's results on the same molecules, if they exist (paired difference).

Usage: python evaluate_pool.py [number of molecules, default 100] [model file in fingerprints_MLP/models/,
                                default: train_MLP.MODEL_PATH]
"""

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem import rdMolHash
from rdkit.Chem.Descriptors import ExactMolWt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # the fingerprints_MLP modules
from build_spectrum_arrays import TRAIN_PATH  # noqa: E402
from evaluate_ranking import (  # noqa: E402
    RESULTS_DIR, choose_query_spectra, predict_queries, random_ranking_mrr, rank_array, reciprocal_ranks,
)
from metadata_features import METADATA_COLUMNS  # noqa: E402
from morgan_generator import N_BITS, fingerprint_from_mol  # noqa: E402
from predict import load_model  # noqa: E402
from rank_candidates import PPM_TOLERANCE, SCORERS  # noqa: E402
from train_MLP import LIBRARY_DIR, MODEL_DIR, MODEL_PATH, SEED, split_spectra_by_molecule  # noqa: E402

from build_pool import POOL_DIR  # noqa: E402
from metric_key import metric_key, standard_key  # noqa: E402
from pubchem_tier import PubChemTier  # noqa: E402

N_MOLECULES = int(sys.argv[1]) if len(sys.argv) > 1 else 100
EVALUATED_MODEL_PATH = MODEL_DIR / sys.argv[2] if len(sys.argv) > 2 else MODEL_PATH
BASELINE_MODEL = "mlp_all_libraries.pt"  # pool MRR@25 0.133 (tanimoto, (train - queries) + PubChem, 1000 molecules)
SAME_MASS_PPM = 0.2  # tier masses agree with RDKit's ExactMolWt within 0.05 ppm
N_WORKERS = 8
POOL_NAMES = ["train only", "(train - queries) + PubChem", "train + PubChem"]


def pool_masks(window: pl.DataFrame, query_keys: pl.Series, query_keys_in_pubchem: pl.Series) -> dict[str, np.ndarray]:
    """For each pool, which rows of the window belong to it.

    A query molecule stays in the "(train - queries) + PubChem" pool only if PubChem has it (query_keys_in_pubchem).
    """
    is_train = (window["source"] == "train").to_numpy()
    is_query = window["key"].is_in(query_keys.implode()).to_numpy()
    in_pubchem = window["key"].is_in(query_keys_in_pubchem.implode()).to_numpy()
    return {
        "train only": is_train,
        "(train - queries) + PubChem": ~is_train | ~is_query | in_pubchem,
        "train + PubChem": np.ones(window.height, dtype=bool),
    }


def skeleton_hash(mol: Chem.Mol) -> str:
    """Elements and connectivity of the heavy atoms, ignoring bond orders and hydrogens: identical for tautomers."""
    return rdMolHash.MolHash(mol, rdMolHash.HashFunction.ElementGraph)


def featurize(window: pl.DataFrame, truth_skeleton: str, truth_metric_key: str) -> tuple[np.ndarray, np.ndarray]:
    """(Morgan bits as float32 (n, N_BITS), is_truth (n,)) for the window's SMILES. Unparsable SMILES get no bits."""
    bits = np.zeros((window.height, N_BITS), dtype=np.float32)
    is_truth = np.zeros(window.height, dtype=bool)
    for i, smiles in enumerate(window["smiles"].to_list()):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        bits[i] = fingerprint_from_mol(mol)
        if skeleton_hash(mol) == truth_skeleton:
            is_truth[i] = metric_key(smiles) == truth_metric_key
    return bits, is_truth


def truth_rank(scores: np.ndarray, keys: pl.Series, is_truth: np.ndarray) -> tuple[int | None, int]:
    """(rank of the first truth candidate or None, number of distinct candidates), one entry per key (best score)."""
    ranking = (
        pl.DataFrame({"key": keys, "score": scores, "is_truth": is_truth})
        .sort("score", descending=True, maintain_order=True)
        .unique("key", keep="first", maintain_order=True)
    )
    truth_positions = np.flatnonzero(ranking["is_truth"].to_numpy())
    rank = int(truth_positions[0]) + 1 if len(truth_positions) > 0 else None
    return rank, ranking.height


def query_keys_found_in_pubchem(truths: pl.DataFrame) -> pl.Series:
    """Standard keys of the query molecules that are in the PubChem tier.

    The pool's in_pubchem column is unknown below 471 Da (see build_pool.py), so it is looked up here: the tier rows
    with the molecule's exact mass (same formula) are keyed and compared to the molecule's key.
    """
    tier = PubChemTier()
    found = []
    with Pool(N_WORKERS) as workers:
        for truth in truths.iter_rows(named=True):
            mass = ExactMolWt(Chem.MolFromSmiles(truth["truth_smiles"]))
            start, end = tier.mass_window(mass, SAME_MASS_PPM)
            if end == start:
                continue
            keys = workers.map(standard_key, tier.smiles(start, end), chunksize=500)
            if truth["truth_key"] in keys:
                found.append(truth["truth_key"])
    return pl.Series("key", found, dtype=pl.String)


def truth_table(query_inchikey14: pl.Series) -> pl.DataFrame:
    """Per query molecule: its SMILES (train's normalized_smiles), standard key, metric key and skeleton hash."""
    molecules = (
        pl.read_parquet(LIBRARY_DIR / "molecules.parquet")
        .filter(pl.col("inchikey14").is_in(query_inchikey14.implode()))
        .unique("inchikey14", keep="first", maintain_order=True)
    )
    smiles = molecules["normalized_smiles"].to_list()
    return molecules.select("inchikey14", pl.col("normalized_smiles").alias("truth_smiles")).with_columns(
        pl.Series("truth_key", [standard_key(s) for s in smiles], dtype=pl.String),
        pl.Series("truth_metric_key", [metric_key(s) for s in smiles], dtype=pl.String),
        pl.Series("truth_skeleton", [skeleton_hash(Chem.MolFromSmiles(s)) for s in smiles], dtype=pl.String),
    )


def summarize(results: pl.DataFrame):
    print(f"\n=== {results.select('inchikey14').n_unique()} validation molecules, ±{PPM_TOLERANCE:g} ppm ===")
    rows = []
    for pool_name in POOL_NAMES:
        pool_results = results.filter(pl.col("pool") == pool_name)
        n_candidates = pool_results["n_candidates"]
        random_mrr = np.mean([
            random_ranking_mrr(n, found) for n, found in zip(n_candidates, pool_results["found"])
        ])
        row = {
            "pool": pool_name,
            "recall": pool_results["found"].mean(),
            "median cand.": n_candidates.median(),
            "p90 cand.": n_candidates.quantile(0.9),
            "max cand.": n_candidates.max(),
            "random MRR": random_mrr,
        }
        for scorer in SCORERS:
            ranks = rank_array(pool_results[f"rank_{scorer}"])
            row[f"MRR {scorer}"] = reciprocal_ranks(pool_results[f"rank_{scorer}"]).mean()
            row[f"top-1 {scorer}"] = np.mean(ranks <= 1)
        rows.append(row)
    with pl.Config(tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True, float_precision=3, tbl_cols=-1, tbl_width_chars=200):
        print(pl.DataFrame(rows))


def results_path(n_molecules: int, model_file_name: str) -> Path:
    return RESULTS_DIR / f"pool_evaluation_{n_molecules}_{Path(model_file_name).stem}.csv"


def compare_with_baseline(results: pl.DataFrame, baseline_results: pl.DataFrame):
    """MRR@25 of the evaluated model vs the baseline model, on the same molecules and pools, per scorer.

    The difference is paired (same molecule, same candidates), with its standard error over molecules.
    """
    print(f"\n=== {EVALUATED_MODEL_PATH.name} vs {BASELINE_MODEL} (same molecules) ===")
    paired = results.join(baseline_results, on=["inchikey14", "pool"], how="inner", suffix="_baseline")
    rows = []
    for pool_name in POOL_NAMES:
        pool_paired = paired.filter(pl.col("pool") == pool_name)
        for scorer in SCORERS:
            new = reciprocal_ranks(pool_paired[f"rank_{scorer}"])
            baseline = reciprocal_ranks(pool_paired[f"rank_{scorer}_baseline"])
            difference = new - baseline
            rows.append({
                "pool": pool_name,
                "scorer": scorer,
                "baseline MRR": baseline.mean(),
                "new MRR": new.mean(),
                "difference": difference.mean(),
                "std. error": difference.std(ddof=1) / np.sqrt(len(difference)),
                "better": int(np.sum(difference > 0)),
                "worse": int(np.sum(difference < 0)),
            })
    with pl.Config(tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True, float_precision=3, tbl_cols=-1, tbl_width_chars=200):
        print(pl.DataFrame(rows))


def main():
    rng = np.random.default_rng(SEED)  # same seed and first use as train_MLP.py: same validation molecules

    print("Loading spectra, split and model")
    spectrum_info = pl.scan_parquet(TRAIN_PATH).select(["ingest_lib", "inchikey14"] + METADATA_COLUMNS).collect()
    arrays = dict(np.load(LIBRARY_DIR / "spectrum_arrays.npz"))
    _, validation_indices = split_spectra_by_molecule(arrays["fp_index"], spectrum_info["ingest_lib"], rng)

    all_queries = choose_query_spectra(spectrum_info, validation_indices)
    chosen_inchikey14 = all_queries["inchikey14"].unique().sort().head(N_MOLECULES)
    queries = all_queries.filter(pl.col("inchikey14").is_in(chosen_inchikey14.implode())).sort("inchikey14", "spectrum_index")

    model = load_model(EVALUATED_MODEL_PATH)
    probabilities = predict_queries(model, queries, spectrum_info, arrays)
    queries = queries.with_row_index("prediction_row")

    truths = truth_table(chosen_inchikey14)
    query_keys = truths["truth_key"]
    lookup_start = time.time()
    query_keys_in_pubchem = query_keys_found_in_pubchem(truths)
    print(f"query molecules in PubChem: {query_keys_in_pubchem.len()} of {truths.height} "
          f"(lookup {time.time() - lookup_start:.0f} s)")
    pool = pl.scan_parquet(POOL_DIR / "pool_*.parquet")
    print(f"{chosen_inchikey14.len()} query molecules | model {EVALUATED_MODEL_PATH.name}")

    results = []
    start_time = time.time()
    for molecule_number, ((inchikey14,), spectra) in enumerate(queries.group_by("inchikey14", maintain_order=True), start=1):
        truth = truths.filter(pl.col("inchikey14") == inchikey14).row(0, named=True)
        molecule_probabilities = probabilities[spectra["prediction_row"].to_numpy()].mean(axis=0)
        neutral_mass = spectra["neutral_mass"].mean()

        tolerance = neutral_mass * PPM_TOLERANCE * 1e-6
        window = pool.filter(pl.col("mass").is_between(neutral_mass - tolerance, neutral_mass + tolerance)).collect()
        bits, is_truth = featurize(window, truth["truth_skeleton"], truth["truth_metric_key"])
        scores = {scorer: SCORERS[scorer](molecule_probabilities, bits) for scorer in SCORERS}

        for pool_name, mask in pool_masks(window, query_keys, query_keys_in_pubchem).items():
            result = {"inchikey14": inchikey14, "pool": pool_name, "found": bool(is_truth[mask].any())}
            for scorer in SCORERS:
                rank, n_candidates = truth_rank(scores[scorer][mask], window["key"].filter(mask), is_truth[mask])
                result[f"rank_{scorer}"] = rank
                result["n_candidates"] = n_candidates
            results.append(result)

        if molecule_number % 25 == 0:
            seconds_per_molecule = (time.time() - start_time) / molecule_number
            print(f"  {molecule_number} molecules, {seconds_per_molecule:.1f} s/molecule, last window {window.height:,} rows", flush=True)

    seconds = time.time() - start_time
    results = pl.DataFrame(results)
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = results_path(chosen_inchikey14.len(), EVALUATED_MODEL_PATH.name)
    results.write_csv(output_path)
    print(f"Per-molecule results in {output_path}")

    summarize(results)
    baseline_path = results_path(chosen_inchikey14.len(), BASELINE_MODEL)
    if EVALUATED_MODEL_PATH.name != BASELINE_MODEL and baseline_path.exists():
        compare_with_baseline(results, pl.read_csv(baseline_path))
    print(f"\nTiming: {seconds / 60:.1f} min ({seconds / chosen_inchikey14.len():.1f} s/molecule)")


if __name__ == "__main__":
    main()
