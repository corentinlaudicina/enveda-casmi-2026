"""Compare model families over their training seeds, from the per-molecule results of evaluate_ranking.py.

A family is a model name without its seed: mlp_resampled covers models/mlp_resampled_seed0.pt, ..._seed1.pt, ...
For each scenario and scorer: MRR@25 of every seed, then their mean and standard deviation. A difference between two
families smaller than about two standard deviations is within the run-to-run noise.

Needs results/fingerprint_ranking_<family>_seed<N>.csv: run python evaluate_ranking.py <family>_seed<N>.pt first.

Usage: python compare_models.py [family ...]  (default: mlp_negatives mlp_resampled mlp_combined)
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # the fingerprints_MLP modules
from evaluate_ranking import RESULTS_DIR, reciprocal_ranks  # noqa: E402
from rank_candidates import SCORERS  # noqa: E402

DEFAULT_FAMILIES = ["mlp_negatives", "mlp_resampled", "mlp_combined"]


def seed_mrrs(family: str) -> pl.DataFrame:
    """One row per (seed, scenario): MRR@25 of each scorer."""
    rows = []
    for results_path in sorted(RESULTS_DIR.glob(f"fingerprint_ranking_{family}_seed*.csv")):
        seed = int(results_path.stem.split("_seed")[-1])
        results = pl.read_csv(results_path)
        for (scenario,), scenario_results in results.group_by("scenario", maintain_order=True):
            row = {"seed": seed, "scenario": scenario}
            for scorer in SCORERS:
                row[scorer] = reciprocal_ranks(scenario_results[f"rank_{scorer}"]).mean()
            rows.append(row)
    return pl.DataFrame(rows)


def main():
    families = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_FAMILIES

    all_mrrs = []
    for family in families:
        mrrs = seed_mrrs(family)
        if mrrs.is_empty():
            print(f"no results for {family} in {RESULTS_DIR}: run evaluate_ranking.py {family}_seed<N>.pt first")
            continue
        all_mrrs.append(mrrs.with_columns(pl.lit(family).alias("family")))
    if not all_mrrs:
        raise SystemExit("nothing to compare")
    all_mrrs = pl.concat(all_mrrs)

    for (scenario,), scenario_mrrs in all_mrrs.group_by("scenario", maintain_order=True):
        print(f"\n=== {scenario} ===")
        for scorer in SCORERS:
            print(f"MRR@25, {scorer} scorer:")
            for (family,), family_mrrs in scenario_mrrs.group_by("family", maintain_order=True):
                values = family_mrrs.sort("seed")[scorer].to_numpy()
                standard_deviation = values.std(ddof=1) if len(values) > 1 else np.nan
                per_seed = " ".join(f"{value:.3f}" for value in values)
                print(f"  {family:<24} {values.mean():.3f} ± {standard_deviation:.3f}   "
                      f"({len(values)} seeds: {per_seed})")


if __name__ == "__main__":
    main()
