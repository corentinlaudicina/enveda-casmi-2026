"""Library search with DreaMS embeddings vs the cosine baseline, on a split of train.parquet.

The provided test.parquet is a copy of train spectra, so it cannot be used to judge a method.
This script makes its own split instead:
- test: N held-out enveda-180 molecules (the test-like library), a few spectra each as the query;
- library: every other train spectrum.

Each test molecule is ranked against the train structures within a mass window, by comparing
its spectra with the candidates' library spectra. Two scenarios:
- known: the test molecule's other spectra stay in the library;
- novel: all its spectra are removed (only its structure remains a candidate).

Compared on the same queries and candidates:
- scorers: cosine on peaks (the baseline), DreaMS embeddings, re-centred DreaMS embeddings,
  and a hybrid (mean of cosine and DreaMS);
- comparison rules: same adduct and energy first, then fall back to either any spectrum of the
  same polarity (the baseline) or any spectrum of the same adduct.

Runs in the DreaMS environment (Python 3.11):
    .venv-dreams/bin/python library_search.py
    .venv-dreams/bin/python library_search.py --n-test 500 --seed 1
    .venv-dreams/bin/python library_search.py --cosine-only

DreaMS embeddings are cached by train row in dreams_cache/embedding_cache.npz, so only new
spectra are embedded (~40-70 spectra/s on a Mac).
"""

import argparse
import os
import time
from dataclasses import dataclass

# The Hugging Face login token on this machine has expired; the DreaMS weights are public
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

DATA_DIR = "data/enveda-CASMI26-molecule-id-mass-spectra"
TRAIN_PATH = os.path.join(DATA_DIR, "train.parquet")
CACHE_PATH = "dreams_cache/embedding_cache.npz"
OLD_EMBEDDINGS_PATH = "dreams_cache/embeddings.npz"  # from embed_dreams.py, reused if present
RESULTS_PATH = "results/library_search.csv"

TOP_K = 25
PPM = 10.0  # initial mass window
MAX_PPM = 50.0  # the window doubles until it holds TOP_K candidates or reaches this
MIN_SPECTRA_PER_TEST_MOLECULE = 4

# Mass added to the neutral molecule by each adduct (electron mass included)
ADDUCT_SHIFT = {
    "[M+H]+": 1.007276,
    "[M+Na]+": 22.989218,
    "[M+NH4]+": 18.033823,
    "[M+K]+": 38.963158,
    "[M-H]-": -1.007276,
    "[M+Cl]-": 34.969402,
    "[M+CH2O2-H]-": 44.998201,
}
ADDUCTS = list(ADDUCT_SHIFT.keys())

def elapsed(start):
    return f"{time.time() - start:.0f}s"


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------

@dataclass
class Spectrum:
    """One cleaned MS/MS spectrum (unit-norm intensities) and how it was measured."""

    mz: np.ndarray
    intensity: np.ndarray
    adduct: str
    energy: str
    polarity: str
    row_id: int

    @staticmethod
    def from_row(row, min_intensity=0.01):
        mz = np.asarray(row["ms2_mzs"])
        intensity = np.asarray(row["ms2_normalized_intensities"])

        # Drop weak peaks and the precursor region: every candidate shares the precursor mass
        keep = (intensity >= min_intensity) & (mz < row["precursor_mz"] - 1.5)
        mz = mz[keep]
        intensity = np.sqrt(intensity[keep])  # square root so the base peak does not dominate
        norm = np.linalg.norm(intensity)
        if norm > 0:
            intensity = intensity / norm

        energy = row["energy_label"]
        if energy is None:
            energy = "unknown"
        return Spectrum(mz, intensity, row["adduct"], energy, row["ionization_mode"], row["row_id"])


@dataclass
class Query:
    """A molecule to identify: its true key, neutral mass and spectra."""

    key: str
    neutral_mass: float
    spectra: list


def energy_label_expression():
    """Cleaned energies as text: "20", "40", "20,40,60" (null when unknown)."""
    values = pl.col("collision_energy_ev").list.eval(pl.element().cast(pl.Int64).cast(pl.String))
    return values.list.join(",").alias("energy_label")


def load_peaks(row_ids):
    """Peaks and measurement info of the given train rows, in one pass over the file."""
    return (
        pl.scan_parquet(TRAIN_PATH)
        .with_row_index("row_id")
        .filter(pl.col("row_id").is_in(sorted(row_ids)))
        .select("row_id", "inchikey14", "ms2_mzs", "ms2_normalized_intensities", "precursor_mz",
                "adduct", "ionization_mode", energy_label_expression())
        .collect()
    )


# ---------------------------------------------------------------------------
# Train metadata, structures and the split
# ---------------------------------------------------------------------------

def load_metadata():
    """One row per train spectrum, without the peaks."""
    return (
        pl.scan_parquet(TRAIN_PATH)
        .with_row_index("row_id")
        .select("row_id", "inchikey14", "normalized_smiles", "ingest_lib", "adduct", "precursor_mz")
        .collect()
    )


def load_structures(metadata):
    """One row per train structure, sorted by exact mass (RDKit's monoisotopic mass of the SMILES)."""
    structures = metadata.unique("inchikey14", keep="first").select("inchikey14", smiles="normalized_smiles")
    masses = [ExactMolWt(Chem.MolFromSmiles(smiles)) for smiles in structures["smiles"]]
    return structures.with_columns(exact_mass=pl.Series(masses, dtype=pl.Float64)).sort("exact_mass")


def make_split(metadata, structures, n_test, spectra_per_query, seed):
    """Pick test molecules from enveda-180 and the spectra that form their queries."""
    enveda = metadata.filter(pl.col("ingest_lib") == "enveda-180", pl.col("adduct").is_in(ADDUCTS))
    spectra_count = enveda.group_by("inchikey14").len()
    eligible = (
        spectra_count
        .filter(pl.col("len") >= MIN_SPECTRA_PER_TEST_MOLECULE)
        .join(structures.select("inchikey14"), on="inchikey14")  # must be a candidate structure
    )
    test_keys = eligible["inchikey14"].sort().sample(n_test, seed=seed).to_list()

    query_row_ids = (
        enveda
        .filter(pl.col("inchikey14").is_in(test_keys))
        .sample(fraction=1.0, shuffle=True, seed=seed)
        .group_by("inchikey14")
        .head(spectra_per_query)
        ["row_id"]
        .to_list()
    )
    return test_keys, query_row_ids


def build_queries(query_rows):
    queries = []
    for (key,), group in query_rows.group_by("inchikey14", maintain_order=True):
        masses = []
        spectra = []
        for row in group.iter_rows(named=True):
            masses.append(row["precursor_mz"] - ADDUCT_SHIFT[row["adduct"]])
            spectra.append(Spectrum.from_row(row))
        queries.append(Query(key, float(np.mean(masses)), spectra))
    return queries


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

class SpectralLibrary:
    """Train structures sorted by exact mass, and the spectra of the structures we need."""

    def __init__(self, structures, spectra_by_key):
        self.keys = structures["inchikey14"].to_list()
        self.smiles = structures["smiles"].to_list()
        self.masses = structures["exact_mass"].to_numpy()
        self.structures = structures
        self.spectra_by_key = spectra_by_key

    def candidate_indices(self, neutral_mass):
        """Structures within the mass window, widening it until there are TOP_K or MAX_PPM is reached."""
        ppm = PPM
        indices = self.indices_within(neutral_mass, ppm)
        while len(indices) < TOP_K and ppm < MAX_PPM:
            ppm = ppm * 2
            indices = self.indices_within(neutral_mass, ppm)
        return indices

    def indices_within(self, neutral_mass, ppm):
        tolerance = neutral_mass * ppm * 1e-6
        start = np.searchsorted(self.masses, neutral_mass - tolerance, side="left")
        end = np.searchsorted(self.masses, neutral_mass + tolerance, side="right")
        return range(start, end)

    def spectra_for(self, key):
        return self.spectra_by_key.get(key, [])

    def without_spectra_for(self, keys):
        """A copy where the given structures have no spectra (to simulate novel molecules)."""
        spectra_by_key = {key: spectra for key, spectra in self.spectra_by_key.items() if key not in keys}
        return SpectralLibrary(self.structures, spectra_by_key)


def group_spectra_by_key(rows):
    spectra_by_key = {}
    for row in rows.iter_rows(named=True):
        spectrum = Spectrum.from_row(row)
        if row["inchikey14"] not in spectra_by_key:
            spectra_by_key[row["inchikey14"]] = []
        spectra_by_key[row["inchikey14"]].append(spectrum)
    return spectra_by_key


# ---------------------------------------------------------------------------
# DreaMS embeddings, cached by train row
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """DreaMS embeddings (unit length) of train rows, stored on disk between runs."""

    def __init__(self, path):
        self.path = path
        self.vectors = {}
        if os.path.exists(path):
            self.add_file(path)
        elif os.path.exists(OLD_EMBEDDINGS_PATH):
            self.add_file(OLD_EMBEDDINGS_PATH)  # embeddings from embed_dreams.py

    def add_file(self, path):
        data = np.load(path)
        self.add(data["row_ids"], data["embeddings"])

    def add(self, row_ids, embeddings):
        for row_id, embedding in zip(row_ids, embeddings):
            self.vectors[int(row_id)] = embedding.astype(np.float32)

    def missing(self, row_ids):
        return [row_id for row_id in row_ids if row_id not in self.vectors]

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        row_ids = np.array(list(self.vectors.keys()))
        embeddings = np.stack([self.vectors[row_id] for row_id in row_ids])
        np.savez(self.path, row_ids=row_ids, embeddings=embeddings)


def embed_rows(embedder, rows):
    """Unit-length DreaMS embeddings of raw spectra rows (peaks as stored in train.parquet)."""
    prepared = [
        embedder.prepare(row["ms2_mzs"], row["ms2_normalized_intensities"], row["precursor_mz"])
        for row in rows.iter_rows(named=True)
    ]
    return embedder.embed(prepared)


# ---------------------------------------------------------------------------
# Scorers: similarity of a query spectrum and a library spectrum, in [0, 1]
# ---------------------------------------------------------------------------

class CosineScorer:
    """Cosine similarity of two peak lists, matching peaks within an m/z tolerance."""

    name = "cosine"

    def __init__(self, tolerance=0.01):
        self.tolerance = tolerance

    def score(self, a, b):
        if len(a.mz) == 0 or len(b.mz) == 0:
            return 0.0
        # For each peak of a, the closest peak of b (b.mz is sorted)
        right = np.clip(np.searchsorted(b.mz, a.mz), 0, len(b.mz) - 1)
        left = np.clip(right - 1, 0, len(b.mz) - 1)
        distance_right = np.abs(b.mz[right] - a.mz)
        distance_left = np.abs(b.mz[left] - a.mz)
        closest = np.where(distance_left < distance_right, left, right)
        matched = np.minimum(distance_left, distance_right) <= self.tolerance
        # Intensities are unit-norm, so the cosine is the sum of products of matched peaks
        return float(min(np.sum(a.intensity[matched] * b.intensity[closest[matched]]), 1.0))


class EmbeddingScorer:
    """Cosine similarity of two DreaMS embeddings, optionally after subtracting the mean embedding."""

    def __init__(self, vectors, name, center=None):
        self.name = name
        self.vectors = {}
        for row_id, vector in vectors.items():
            if center is not None:
                vector = vector - center
                vector = vector / np.linalg.norm(vector)
            self.vectors[row_id] = vector

    def score(self, a, b):
        if a.row_id not in self.vectors or b.row_id not in self.vectors:
            return 0.0
        return max(float(np.dot(self.vectors[a.row_id], self.vectors[b.row_id])), 0.0)


class HybridScorer:
    """Mean of two scorers."""

    def __init__(self, first, second, name):
        self.first = first
        self.second = second
        self.name = name

    def score(self, a, b):
        return 0.5 * (self.first.score(a, b) + self.second.score(a, b))


# ---------------------------------------------------------------------------
# Which library spectra a query spectrum is compared with
# ---------------------------------------------------------------------------

def same_setup(query_spectrum, spectrum):
    return spectrum.adduct == query_spectrum.adduct and spectrum.energy == query_spectrum.energy


def comparable_with_fallback(fallback_field):
    """Same adduct and energy if any; otherwise any spectrum sharing the query's fallback_field."""

    def comparable(query_spectrum, library_spectra):
        exact = [s for s in library_spectra if same_setup(query_spectrum, s)]
        if len(exact) > 0:
            return exact
        return [s for s in library_spectra if getattr(s, fallback_field) == getattr(query_spectrum, fallback_field)]

    return comparable


COMPARISON_RULES = {
    "polarity fallback": comparable_with_fallback("polarity"),  # the baseline
    "adduct fallback": comparable_with_fallback("adduct"),
}


# ---------------------------------------------------------------------------
# Search and metrics
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    key: str
    smiles: str
    score: float
    mass_error_ppm: float


class LibrarySearch:
    """Rank candidate structures for a query: mass window first, then spectral similarity."""

    def __init__(self, library, scorer, comparable):
        self.library = library
        self.scorer = scorer
        self.comparable = comparable

    def score_candidate(self, query, key):
        """Mean over the query spectra of the best similarity with the candidate's comparable spectra."""
        library_spectra = self.library.spectra_for(key)
        if len(library_spectra) == 0:
            return 0.0
        best_scores = []
        for query_spectrum in query.spectra:
            best = 0.0
            for spectrum in self.comparable(query_spectrum, library_spectra):
                best = max(best, self.scorer.score(query_spectrum, spectrum))
            best_scores.append(best)
        return float(np.mean(best_scores))

    def rank(self, query):
        candidates = []
        for index in self.library.candidate_indices(query.neutral_mass):
            key = self.library.keys[index]
            mass_error = abs(self.library.masses[index] - query.neutral_mass) / query.neutral_mass * 1e6
            candidates.append(Candidate(key, self.library.smiles[index], self.score_candidate(query, key), mass_error))
        # Highest score first; ties (e.g. no spectra) broken by the smallest mass error
        candidates.sort(key=lambda candidate: (-candidate.score, candidate.mass_error_ppm))
        return candidates[:TOP_K]


def evaluate(search, queries):
    """MRR@25 and top-k hit rates; guesses are compared by inchikey14."""
    reciprocal_ranks = []
    for query in queries:
        reciprocal_rank = 0.0
        for rank, candidate in enumerate(search.rank(query), start=1):
            if candidate.key == query.key:
                reciprocal_rank = 1.0 / rank
                break
        reciprocal_ranks.append(reciprocal_rank)
    reciprocal_ranks = np.array(reciprocal_ranks)
    return {
        "mrr@25": float(reciprocal_ranks.mean()),
        "top1": float(np.mean(reciprocal_ranks == 1.0)),
        "top5": float(np.mean(reciprocal_ranks >= 1 / 5)),
        "top25": float(np.mean(reciprocal_ranks > 0)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-test", type=int, default=200, help="number of held-out test molecules")
    parser.add_argument("--spectra-per-query", type=int, default=3, help="spectra per test molecule")
    parser.add_argument("--seed", type=int, default=0, help="random seed of the split")
    parser.add_argument("--cosine-only", action="store_true", help="skip DreaMS (no embedding needed)")
    parser.add_argument("--output", default=RESULTS_PATH, help="CSV file for the results table")
    return parser.parse_args()


def main():
    args = parse_arguments()
    started = time.time()

    print("1. Train metadata and structures")
    metadata = load_metadata()
    structures = load_structures(metadata)
    print(f"   {metadata.height:,} spectra, {structures.height:,} candidate structures ({elapsed(started)})")

    print("2. Split")
    test_keys, query_row_ids = make_split(metadata, structures, args.n_test, args.spectra_per_query, args.seed)
    query_rows = load_peaks(query_row_ids)
    queries = build_queries(query_rows)
    print(f"   {len(queries)} test molecules, {len(query_row_ids)} query spectra ({elapsed(started)})")

    print("3. Library spectra of every candidate")
    empty_library = SpectralLibrary(structures, {})
    candidate_keys = set()
    for query in queries:
        for index in empty_library.candidate_indices(query.neutral_mass):
            candidate_keys.add(empty_library.keys[index])
    query_row_set = set(query_row_ids)
    library_row_ids = (
        metadata
        .filter(pl.col("inchikey14").is_in(list(candidate_keys)), pl.col("adduct").is_in(ADDUCTS))
        ["row_id"]
        .to_list()
    )
    library_row_ids = [row_id for row_id in library_row_ids if row_id not in query_row_set]
    library_rows = load_peaks(library_row_ids)
    library = SpectralLibrary(structures, group_spectra_by_key(library_rows))
    novel_library = library.without_spectra_for(set(test_keys))
    print(f"   {len(candidate_keys):,} candidate structures, {library_rows.height:,} library spectra ({elapsed(started)})")

    scorers = [CosineScorer()]
    if not args.cosine_only:
        print("4. DreaMS embeddings")
        cache = EmbeddingCache(CACHE_PATH)
        needed_row_ids = library_row_ids + query_row_ids
        missing = cache.missing(needed_row_ids)
        print(f"   {len(needed_row_ids) - len(missing):,} cached, {len(missing):,} to embed")
        if len(missing) > 0:
            from embed_dreams import DreamsEmbedder  # imported here: it needs torch and dreams

            embedder = DreamsEmbedder()
            all_rows = pl.concat([library_rows, query_rows])
            missing_rows = all_rows.filter(pl.col("row_id").is_in(missing))
            cache.add(missing_rows["row_id"].to_list(), embed_rows(embedder, missing_rows))
            cache.save()
        vectors = {row_id: cache.vectors[row_id] for row_id in needed_row_ids}
        library_mean = np.mean([vectors[row_id] for row_id in library_row_ids], axis=0)

        dreams = EmbeddingScorer(vectors, "DreaMS")
        dreams_centered = EmbeddingScorer(vectors, "DreaMS centred", center=library_mean)
        scorers.append(dreams)
        scorers.append(dreams_centered)
        scorers.append(HybridScorer(CosineScorer(), dreams, "cosine + DreaMS"))
        print(f"   done ({elapsed(started)})")

    print("5. Evaluation")
    results = []
    for scenario, scenario_library in [("known", library), ("novel", novel_library)]:
        for rule_name, comparable in COMPARISON_RULES.items():
            for scorer in scorers:
                metrics = evaluate(LibrarySearch(scenario_library, scorer, comparable), queries)
                results.append({"scenario": scenario, "scorer": scorer.name, "comparison": rule_name, **metrics})
                print(f"   {scenario:<6} {scorer.name:<16} {rule_name:<18} "
                      f"MRR@25 {metrics['mrr@25']:.3f} | top-1 {metrics['top1']:.3f} | "
                      f"top-5 {metrics['top5']:.3f} | top-25 {metrics['top25']:.3f}", flush=True)

    table = pl.DataFrame(results).with_columns(
        n_test=pl.lit(len(queries)), seed=pl.lit(args.seed)
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    table.write_csv(args.output)
    print(f"\nSaved {args.output} ({elapsed(started)} in total)")


if __name__ == "__main__":
    main()
