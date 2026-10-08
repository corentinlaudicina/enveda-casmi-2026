"""Rank candidate structures for one query molecule from its predicted fingerprint probabilities.

1. Candidates: every structure of the library whose exact mass is within PPM_TOLERANCE of the query's neutral mass.
2. Score: how well each candidate's true fingerprint agrees with the predicted probabilities (two scorers below).
3. Rank by score, keeping one entry per inchikey14 (the metric compares inchikey14, so a second SMILES with the same inchikey14, e.g. a stereoisomer, would waste a slot of the top 25).
"""

import numpy as np
import polars as pl

from morgan_generator import N_BITS

PPM_TOLERANCE = 10.0
EPSILON = 1e-6  # keeps log(p) and log(1 - p) finite


def likelihood_scores(probabilities: np.ndarray, candidate_bits: np.ndarray) -> np.ndarray:
    """Log-likelihood of each candidate's fingerprint under the predicted bit probabilities.

    log L = sum over bits of  bit * log(p) + (1 - bit) * log(1 - p)
          = sum over bits of  bit * [log(p) - log(1 - p)]  +  sum over bits of log(1 - p)
    The second sum is the same for every candidate, so it doesn't change the ranking and is dropped: the score is a weighted count of the candidate's bits, each bit weighted by its log-odds.
    """
    p = np.clip(probabilities, EPSILON, 1 - EPSILON)
    log_odds = np.log(p) - np.log(1 - p)
    return candidate_bits @ log_odds


def tanimoto_scores(probabilities: np.ndarray, candidate_bits: np.ndarray) -> np.ndarray:
    """Tanimoto similarity between the probabilities (as soft bits) and each candidate's fingerprint."""
    intersection = candidate_bits @ probabilities
    union = probabilities.sum() + candidate_bits.sum(axis=1) - intersection
    return intersection / union


SCORERS = {"likelihood": likelihood_scores, "tanimoto": tanimoto_scores}


def mass_windows(sorted_masses: np.ndarray, neutral_masses, ppm: float = PPM_TOLERANCE):
    """(first, last) row positions, last excluded, of the sorted_masses within ppm of each neutral mass (scalar or array)."""
    neutral_masses = np.asarray(neutral_masses)
    tolerance = neutral_masses * ppm * 1e-6
    first = np.searchsorted(sorted_masses, neutral_masses - tolerance, side="left")
    last = np.searchsorted(sorted_masses, neutral_masses + tolerance, side="right")
    return first, last


class CandidateLibrary:
    """Structures with their exact masses and fingerprints, sorted by mass for fast mass-window lookups."""

    def __init__(self, molecules: pl.DataFrame, packed_fingerprints: np.ndarray):
        """molecules: one row per structure with normalized_smiles, inchikey14, exact_mass;
        packed_fingerprints: its bit-packed fingerprints, same row order."""
        order = np.argsort(molecules["exact_mass"].to_numpy())
        self.molecules = molecules[order]
        self.masses = self.molecules["exact_mass"].to_numpy()
        self.packed_fingerprints = packed_fingerprints[order]

    def candidates_in_mass_window(self, neutral_mass: float) -> np.ndarray:
        """Row positions of the structures within PPM_TOLERANCE of neutral_mass."""
        first, last = mass_windows(self.masses, neutral_mass)
        return np.arange(first, last)

    def rank(
        self, probabilities: np.ndarray, neutral_mass: float, scorer="likelihood", n_bits: int = N_BITS
    ) -> pl.DataFrame:
        """All candidates of the mass window, best first, one row per inchikey14.

        scorer: a name of SCORERS, or a function (probabilities, candidate_bits) -> scores.
        n_bits: width of the stored fingerprints.
        Columns: normalized_smiles, inchikey14, score. Empty frame if no structure has that mass.
        """
        positions = self.candidates_in_mass_window(neutral_mass)
        candidate_bits = np.unpackbits(self.packed_fingerprints[positions],
                                       axis=1, count=n_bits).astype(np.float32)
        score_function = SCORERS[scorer] if isinstance(scorer, str) else scorer
        scores = score_function(probabilities, candidate_bits)

        candidates = self.molecules[positions].select("normalized_smiles", "inchikey14")
        candidates = candidates.with_columns(pl.Series("score", scores))
        ranking = candidates.sort("score", descending=True)

        # The ranking is sorted, so the first row of each inchikey14 is its best-scoring one.
        ranking = ranking.unique(subset="inchikey14", keep="first", maintain_order=True)
        return ranking
