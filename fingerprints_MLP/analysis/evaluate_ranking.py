"""Evaluate the MLP + candidate ranking with the competition metric (MRR@25) on molecules the MLP has not seen.

Each query molecule gets the spectra of ONE adduct (all its collision energies), like a test molecule_id.
Its prediction is the MLP's bit probabilities averaged over those spectra; candidates are the structures
of a pool within PPM_TOLERANCE of its neutral mass; metric = 1 / rank of the true inchikey14 if in the top 25, else 0.

Scenarios:
- A. enveda validation, train pool: N_EVAL_MOLECULES validation molecules of train_MLP.py's split, from the test
     half (split_selection_and_test: the training scripts choose their best epoch on the other half),
     candidates = train structures. The answer is always in the pool: measures the ranking alone.
- B. enveda validation, train + COCONUT pool: same molecules, ~3x more candidates: measures the cost of a bigger pool.
- C. natural products, (train - queries) + COCONUT pool: the 250 enveda-np-examples molecules (timsTOF natural
     products, never seen by the MLP, which trains on enveda-180 only). They are removed from the train part of
     the pool, so they are found only if COCONUT has them: a realistic "novel natural product" test.

Compared to random ranking of the same candidates and to library search on novel molecules (0.017). Per-molecule results of all scenarios are saved to results/fingerprint_ranking_<model name>.csv;
compare_models.py summarises them over training seeds.

A combined model (train_MLP_combined.py) also gets the DreaMS embedding of each query spectrum (from dreaMS/embed_train.py: query spectra are train rows), and the flag 0 for spectra of rare adducts that have none.

Usage: python evaluate_ranking.py [model file in fingerprints_MLP/models/, default: train_MLP.MODEL_PATH]
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # the fingerprints_MLP modules
from build_spectrum_arrays import TRAIN_PATH  # noqa: E402
from metadata_features import METADATA_COLUMNS, metadata_features, neutral_mass_expression  # noqa: E402
from predict import load_model, predict_probabilities, uses_dreams_embedding  # noqa: E402
from rank_candidates import PPM_TOLERANCE, SCORERS, CandidateLibrary  # noqa: E402
from train_MLP import (  # noqa: E402
    LIBRARY_DIR, MODEL_DIR, MODEL_PATH, SEED, split_selection_and_test, split_spectra_by_molecule,
)
from train_MLP_dreams import embeddings_of_rows  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

N_EVAL_MOLECULES = 1000
TOP_K = 25
NATURAL_PRODUCT_LIBRARY = "enveda-np-examples"
POOL_COLUMNS = ["normalized_smiles", "inchikey14", "exact_mass"]


def choose_query_spectra(spectrum_info: pl.DataFrame, spectrum_indices: np.ndarray) -> pl.DataFrame:
    """One row per query spectrum: up to N_EVAL_MOLECULES molecules among the given spectra, each with one adduct.

    Columns: spectrum_index (row in train.parquet / the spectrum arrays), inchikey14, adduct, neutral_mass.
    """
    spectra = (
        spectrum_info[spectrum_indices]
        .select("inchikey14", "adduct", neutral_mass_expression().alias("neutral_mass"))
        .with_columns(pl.Series("spectrum_index", spectrum_indices))
    )

    # one random adduct per molecule, then a random subset of molecules
    molecule_adducts = (
        spectra.select("inchikey14", "adduct")
        .unique()
        .sort("inchikey14", "adduct")  # fixed order before shuffling, so the sample is reproducible
        .sample(fraction=1.0, shuffle=True, seed=SEED)
        .unique(subset="inchikey14", keep="first", maintain_order=True)
        .head(N_EVAL_MOLECULES)
    )
    return spectra.join(molecule_adducts, on=["inchikey14", "adduct"], how="inner")


def predict_queries(
    model: nn.Module, queries: pl.DataFrame, spectrum_info: pl.DataFrame, arrays: dict[str, np.ndarray]
) -> np.ndarray:
    """Bit probabilities of every query spectrum, row i <-> row i of queries."""
    spectrum_indices = queries["spectrum_index"].to_numpy()
    metadata = metadata_features(spectrum_info[spectrum_indices])

    embeddings = has_embedding = None
    if uses_dreams_embedding(model):
        embeddings, has_embedding = embeddings_of_rows(spectrum_indices)
        print(f"DreaMS embedding for {has_embedding.sum():,} of {len(spectrum_indices):,} query spectra")

    return predict_probabilities(
        model, arrays["peak_bins"][spectrum_indices], arrays["peak_intensities"][spectrum_indices], metadata,
        embeddings, has_embedding,
    )


def build_pool(sources: list[tuple[pl.DataFrame, np.ndarray]]) -> CandidateLibrary:
    """Concatenate (molecules, packed fingerprints) sources into one CandidateLibrary."""
    molecules = pl.concat([source_molecules.select(POOL_COLUMNS) for source_molecules, _ in sources])
    packed_fingerprints = np.concatenate([source_fingerprints for _, source_fingerprints in sources])
    return CandidateLibrary(molecules, packed_fingerprints)


def remove_molecules(
    molecules: pl.DataFrame, packed_fingerprints: np.ndarray, excluded_inchikey14: pl.Series
) -> tuple[pl.DataFrame, np.ndarray]:
    """The source without the structures whose inchikey14 is in excluded_inchikey14."""
    keep = ~molecules["inchikey14"].is_in(excluded_inchikey14.implode()).to_numpy()
    return molecules.filter(keep), packed_fingerprints[keep]


def rank_queries(queries: pl.DataFrame, probabilities: np.ndarray, library: CandidateLibrary) -> pl.DataFrame:
    """One row per query molecule: rank of the true inchikey14 for each scorer (null if not a candidate)."""
    queries = queries.with_row_index("prediction_row")
    results = []
    for (true_inchikey14,), spectra in queries.group_by("inchikey14", maintain_order=True):
        molecule_probabilities = probabilities[spectra["prediction_row"].to_numpy()].mean(axis=0)
        neutral_mass = spectra["neutral_mass"].mean()

        result = {"inchikey14": true_inchikey14, "adduct": spectra["adduct"][0], "n_spectra": spectra.height}
        for scorer in SCORERS:
            ranking = library.rank(molecule_probabilities, neutral_mass, scorer=scorer)
            true_position = ranking["inchikey14"].index_of(true_inchikey14)  # None if not a candidate
            result[f"rank_{scorer}"] = None if true_position is None else true_position + 1
            result["n_candidates"] = ranking.height
        result["true_in_window"] = result["rank_likelihood"] is not None
        results.append(result)
    return pl.DataFrame(results)


def random_ranking_mrr(n_candidates: int, true_in_window: bool) -> float:
    """Expected reciprocal rank (top TOP_K) if the true structure is placed uniformly at random among n candidates."""
    if not true_in_window or n_candidates == 0:
        return 0.0
    ranks = np.arange(1, min(TOP_K, n_candidates) + 1)
    return float(np.sum(1.0 / ranks) / n_candidates)


def rank_array(ranks: pl.Series) -> np.ndarray:
    """Ranks as integers; a truth that is not a candidate (null) gets a rank beyond any TOP_K."""
    return ranks.fill_null(np.iinfo(np.int32).max).to_numpy()


def reciprocal_ranks(ranks: pl.Series) -> np.ndarray:
    """1 / rank if the truth is in the top TOP_K, else 0."""
    integer_ranks = rank_array(ranks)
    return np.where(integer_ranks <= TOP_K, 1.0 / integer_ranks, 0.0)


def print_summary(scenario: str, results: pl.DataFrame):
    print(f"\n=== {scenario} ({results.height} molecules) ===")
    print(f"True structure in the pool and inside the {PPM_TOLERANCE:g} ppm window: {results['true_in_window'].mean():.1%}")
    print(f"Candidates per molecule (distinct inchikey14): median {results['n_candidates'].median():.0f}, "
          f"mean {results['n_candidates'].mean():.0f}, max {results['n_candidates'].max()}")

    random_mrr = np.mean([
        random_ranking_mrr(n, found) for n, found in zip(results["n_candidates"], results["true_in_window"])
    ])
    print(f"{'scorer':<12} {'MRR@25':>7} {'top-1':>7} {'top-5':>7} {'top-25':>7} {'median rank if found':>21}")
    for scorer in SCORERS:
        ranks = rank_array(results[f"rank_{scorer}"])
        median_rank = results[f"rank_{scorer}"].median()  # None if never found
        median_text = "-" if median_rank is None else f"{median_rank:.0f}"
        print(f"{scorer:<12} {reciprocal_ranks(results[f'rank_{scorer}']).mean():>7.3f} {np.mean(ranks <= 1):>7.1%} "
              f"{np.mean(ranks <= 5):>7.1%} {np.mean(ranks <= TOP_K):>7.1%} {median_text:>21}")
    print(f"{'random':<12} {random_mrr:>7.3f}")


def main():
    rng = np.random.default_rng(SEED)  # same seed and first use as train_MLP.py: same validation molecules

    print("Loading spectra, libraries and model")
    spectrum_info = pl.scan_parquet(TRAIN_PATH).select(["ingest_lib", "inchikey14"] + METADATA_COLUMNS).collect()
    arrays = dict(np.load(LIBRARY_DIR / "spectrum_arrays.npz"))
    _, validation_indices = split_spectra_by_molecule(arrays["fp_index"], spectrum_info["ingest_lib"], rng)
    _, test_indices = split_selection_and_test(arrays["fp_index"], validation_indices)  # not used to pick the epoch

    train_molecules = pl.read_parquet(LIBRARY_DIR / "molecules.parquet")
    train_fingerprints = np.load(LIBRARY_DIR / "morgan_fingerprints.npy")
    coconut_molecules = pl.read_parquet(LIBRARY_DIR / "coconut_molecules.parquet")
    coconut_fingerprints = np.load(LIBRARY_DIR / "coconut_fingerprints.npy")
    model_path = MODEL_DIR / sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH
    print(f"Model: {model_path.name}")
    model = load_model(model_path)

    # query molecules and their predictions
    enveda_queries = choose_query_spectra(spectrum_info, test_indices)
    enveda_probabilities = predict_queries(model, enveda_queries, spectrum_info, arrays)

    natural_product_indices = np.flatnonzero((spectrum_info["ingest_lib"] == NATURAL_PRODUCT_LIBRARY).to_numpy())
    natural_product_queries = choose_query_spectra(spectrum_info, natural_product_indices)
    natural_product_probabilities = predict_queries(model, natural_product_queries, spectrum_info, arrays)

    # candidate pools
    train_pool = build_pool([(train_molecules, train_fingerprints)])
    train_coconut_pool = build_pool([(train_molecules, train_fingerprints), (coconut_molecules, coconut_fingerprints)])
    # the natural products are removed from the train part only: they can still be found through COCONUT
    train_without_natural_products = remove_molecules(
        train_molecules, train_fingerprints, natural_product_queries["inchikey14"].unique()
    )
    novel_natural_product_pool = build_pool([train_without_natural_products, (coconut_molecules, coconut_fingerprints)])

    scenarios = {
        "A. enveda validation, train pool": (enveda_queries, enveda_probabilities, train_pool),
        "B. enveda validation, train + COCONUT pool": (enveda_queries, enveda_probabilities, train_coconut_pool),
        "C. natural products, (train - queries) + COCONUT pool": (
            natural_product_queries, natural_product_probabilities, novel_natural_product_pool
        ),
    }

    all_results = []
    for scenario, (queries, probabilities, pool) in scenarios.items():
        results = rank_queries(queries, probabilities, pool)
        print_summary(scenario, results)
        all_results.append(results.with_columns(pl.lit(scenario).alias("scenario")))

    results_path = RESULTS_DIR / f"fingerprint_ranking_{model_path.stem}.csv"
    RESULTS_DIR.mkdir(exist_ok=True)
    pl.concat(all_results).write_csv(results_path)
    print(f"\nPer-molecule results in {results_path}")


if __name__ == "__main__":
    main()
