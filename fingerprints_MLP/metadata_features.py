"""Numeric features describing how a spectrum was measured, appended to the binned spectrum as MLP input.

All features are scaled with fixed constants (not statistics fitted on train), so train and test
spectra are transformed identically, and every feature lies roughly in [0, 1] like the sqrt intensities.

Features, in order:
- precursor_mz / 1000
- neutral_mass / 1000: the molecule's mass, recovered from precursor_mz and the adduct
  (for [2M+Na]+, precursor_mz is about twice the molecule's mass)
- is_positive: 1 for positive ionization mode, 0 for negative
- collision energy multi-hot: one column per energy in COLLISION_ENERGIES, 1 if the spectrum was measured at it
  (a merged 20,40,60 spectrum has all three set), plus one "other energy" column
- adduct one-hot: one column per adduct in ADDUCTS, plus one "other adduct" column
"""

import numpy as np
import polars as pl

# Columns of train.parquet / test.parquet needed to compute the features
METADATA_COLUMNS = ["precursor_mz", "adduct", "ionization_mode", "collision_energy_ev"]

MASS_SCALE = 1000.0       # Da

# The energies of enveda-180 and of the test spectra, alone or merged
COLLISION_ENERGIES = [20.0, 40.0, 60.0]  # eV

# adduct -> (number of molecules M in the ion, mass added to n * M), all singly charged:
# precursor_mz = n * M + shift, so M = (precursor_mz - shift) / n
# Covers every adduct of enveda-180, which includes the 7 adducts of the test spectra.
PROTON = 1.007276
ADDUCTS = {
    "[M+H]+": (1, PROTON),
    "[M+Na]+": (1, 22.989218),
    "[M+NH4]+": (1, 18.033823),
    "[M+K]+": (1, 38.963158),
    "[2M+H]+": (2, PROTON),
    "[2M+Na]+": (2, 22.989218),
    "[M-H]-": (1, -PROTON),
    "[M+Cl]-": (1, 34.969402),
    "[M+Br]-": (1, 78.918885),
    "[M+CH2O2-H]-": (1, 46.005479 - PROTON),   # formate adduct
    "[M+C2H4O2-H]-": (1, 60.021129 - PROTON),  # acetate adduct
    "[2M-H]-": (2, -PROTON),
    "[2M+CH2O2-H]-": (2, 46.005479 - PROTON),
    "[2M+C2H4O2-H]-": (2, 60.021129 - PROTON),
}
ADDUCT_NAMES = list(ADDUCTS)

N_METADATA_FEATURES = 3 + len(COLLISION_ENERGIES) + 1 + len(ADDUCT_NAMES) + 1


def neutral_mass_expression() -> pl.Expr:
    """Polars expression: the molecule's neutral mass M = (precursor_mz - shift) / n, from its adduct.

    An adduct outside ADDUCTS falls back to M = precursor_mz.
    """
    n_molecules_in_ion = {name: n for name, (n, shift) in ADDUCTS.items()}
    adduct_shift = {name: shift for name, (n, shift) in ADDUCTS.items()}

    multiplier = pl.col("adduct").replace_strict(n_molecules_in_ion, default=1, return_dtype=pl.Float64)
    shift = pl.col("adduct").replace_strict(adduct_shift, default=0.0, return_dtype=pl.Float64)
    return (pl.col("precursor_mz") - shift) / multiplier


def metadata_features(spectra: pl.DataFrame) -> np.ndarray:
    """Return a float32 matrix (n_spectra, N_METADATA_FEATURES) from a frame holding METADATA_COLUMNS.

    An adduct outside ADDUCTS sets the "other adduct" column, and its neutral mass falls back to precursor_mz.
    An energy outside COLLISION_ENERGIES sets the "other energy" column; a missing one leaves all energy columns at 0.
    """
    neutral_mass = neutral_mass_expression()
    energies = pl.col("collision_energy_ev")  # list of energies in eV

    feature_columns = [
        (pl.col("precursor_mz") / MASS_SCALE).alias("precursor_mz"),
        (neutral_mass / MASS_SCALE).alias("neutral_mass"),
        (pl.col("ionization_mode") == "positive").alias("is_positive"),
    ]
    for energy in COLLISION_ENERGIES:
        feature_columns.append(energies.list.contains(energy).alias(f"energy {energy:g} eV"))
    has_other_energy = energies.list.eval(~pl.element().is_in(COLLISION_ENERGIES)).list.any()
    feature_columns.append(has_other_energy.alias("energy other"))

    for name in ADDUCT_NAMES:
        feature_columns.append((pl.col("adduct") == name).alias(f"adduct {name}"))
    feature_columns.append((~pl.col("adduct").is_in(ADDUCT_NAMES)).alias("adduct other"))

    features = spectra.select(feature_columns).cast(pl.Float32).fill_null(0.0)
    return features.to_numpy()
