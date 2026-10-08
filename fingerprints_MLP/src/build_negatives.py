"""Negatives for the ranking loss of train_MLP_negatives.py: for every molecule of the fingerprint library,
N_NEGATIVES structures of the candidate pool (train + PubChem, external/pool/) with the same mass (±PPM_TOLERANCE).

At retrieval time the predicted fingerprint has to beat ~10,000 same-mass structures; these are a random sample of them.
They are precomputed because training reads millions of batches: fingerprinting SMILES on the fly, or reading the
USB pool at random, would be far too slow.

Per molecule (all fp_index rows of molecules.parquet, so the output is indexed like morgan_fingerprints.npy;
the negatives of the validation molecules only serve the validation ranking metric):
- window: the pool structures within PPM_TOLERANCE of the molecule's exact mass
- the molecules the MLP never trains on (validation split, natural-product test set, libraries outside
  TRAIN_LIBRARIES) are removed from the pool first, matched on the pool key: they never appear as negatives
- N_DRAWN random structures of the window are fingerprinted; the first N_NEGATIVES that parse and whose fingerprint
  differs from the molecule's own are kept. This drops the molecule itself, its stereoisomers and fingerprint
  collisions: the loss could not separate them anyway.
A window with too few structures gives fewer negatives: the rest of the row is zeros, and n_negatives says how many
are real.

The pool is read once, part by part in mass order (sequential reads on the USB disk), and the molecules are handled
in mass order alongside. The end of each part is carried over to the next, for windows that span two parts.

Output: library/data/negatives.npz
- fingerprints: uint8 (n_molecules, N_NEGATIVES, N_BITS / 8), bit-packed like morgan_fingerprints.npy
- n_negatives: int16 (n_molecules,), number of real negatives in each row
- window_size: int32 (n_molecules,), pool structures in the window (the molecule itself included)

Usage: python build_negatives.py
"""

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl
from build_spectrum_arrays import TRAIN_PATH
from morgan_generator import N_BITS, fingerprint
from rank_candidates import PPM_TOLERANCE, mass_windows
from train_MLP import LIBRARY_DIR, SEED, split_spectra_by_molecule

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "library" / "pubchem_pool"))
from build_pool import POOL_DIR  # noqa: E402
from metric_key import standard_key  # noqa: E402

OUTPUT_PATH = LIBRARY_DIR / "negatives.npz"
N_NEGATIVES = 32
N_DRAWN = 48  # structures fingerprinted per molecule, to have N_NEGATIVES left after dropping failures and twins
N_BYTES = N_BITS // 8
N_WORKERS = 8


def held_out_keys(molecules: pl.DataFrame, workers: Pool) -> pl.Series:
    """Pool keys (standard InChIKey first block) of the molecules that have no training spectrum in train_MLP.py.

    Same split as train_MLP.py: same seed, and the split is the generator's first use.
    """
    rng = np.random.default_rng(SEED)
    fp_index = np.load(LIBRARY_DIR / "spectrum_arrays.npz")["fp_index"]
    ingest_lib = pl.scan_parquet(TRAIN_PATH).select("ingest_lib").collect()["ingest_lib"]
    train_indices, _ = split_spectra_by_molecule(fp_index, ingest_lib, rng)

    is_trained_on = np.zeros(molecules.height, dtype=bool)
    is_trained_on[fp_index[train_indices]] = True
    smiles = molecules.filter(~pl.Series(is_trained_on))["normalized_smiles"].to_list()
    keys = workers.map(standard_key, smiles, chunksize=1_000)
    return pl.Series("key", keys, dtype=pl.String).drop_nulls()


def select_negatives(drawn_smiles: list[str], own_packed: np.ndarray) -> np.ndarray:
    """Packed fingerprints (k, N_BYTES) of the first N_NEGATIVES drawn structures that parse and differ from the molecule."""
    kept = []
    for smiles in drawn_smiles:
        bits = fingerprint(smiles)
        if bits is None:
            continue
        packed = np.packbits(bits)
        if np.array_equal(packed, own_packed):
            continue
        kept.append(packed)
        if len(kept) == N_NEGATIVES:
            break
    if len(kept) == 0:
        return np.zeros((0, N_BYTES), dtype=np.uint8)
    return np.stack(kept)


def draw_windows(
    molecule_masses: np.ndarray, loaded_masses: np.ndarray, loaded_smiles: list[str], rng: np.random.Generator
) -> tuple[list[list[str]], np.ndarray]:
    """For each molecule: (SMILES of up to N_DRAWN random structures of its mass window, window size)."""
    drawn_smiles = []
    window_sizes = np.zeros(len(molecule_masses), dtype=np.int32)
    starts, ends = mass_windows(loaded_masses, molecule_masses)
    for i in range(len(molecule_masses)):
        start = int(starts[i])
        end = int(ends[i])
        window_sizes[i] = end - start

        n_drawn = min(N_DRAWN, end - start)
        positions = start + rng.choice(end - start, size=n_drawn, replace=False)
        drawn_smiles.append([loaded_smiles[position] for position in positions])
    return drawn_smiles, window_sizes


def main():
    molecules = pl.read_parquet(LIBRARY_DIR / "molecules.parquet")
    own_fingerprints = np.load(LIBRARY_DIR / "morgan_fingerprints.npy")
    masses = molecules["exact_mass"].to_numpy()
    mass_order = np.argsort(masses)  # fp_index of the molecules, lightest first
    sorted_masses = masses[mass_order]

    negatives = np.zeros((molecules.height, N_NEGATIVES, N_BYTES), dtype=np.uint8)
    n_negatives = np.zeros(molecules.height, dtype=np.int16)
    window_size = np.zeros(molecules.height, dtype=np.int32)

    part_paths = sorted(POOL_DIR.glob("pool_*.parquet"))
    rng = np.random.default_rng(SEED)
    start_time = time.time()

    with Pool(N_WORKERS) as workers:
        excluded_keys = held_out_keys(molecules, workers)
        print(f"{molecules.height:,} molecules | {excluded_keys.len():,} never trained on, removed from the pool")

        carried = pl.DataFrame(schema={"smiles": pl.String, "mass": pl.Float64})
        next_molecule = 0  # position in mass_order of the first molecule not yet handled
        for part_number, part_path in enumerate(part_paths):
            part = (
                pl.read_parquet(part_path, columns=["key", "smiles", "mass"])
                .filter(~pl.col("key").is_in(excluded_keys.implode()))
                .select("smiles", "mass")
            )
            loaded = pl.concat([carried, part])  # still sorted by mass: the carried rows are lighter

            # the molecules whose whole window is loaded: its upper end is below the next part's lightest structure
            if part_number + 1 < len(part_paths):
                next_part_lightest = pl.scan_parquet(part_paths[part_number + 1]).select(pl.col("mass").min()).collect().item()
            else:
                next_part_lightest = np.inf
            window_upper_ends = sorted_masses * (1 + PPM_TOLERANCE * 1e-6)
            end_molecule = int(np.searchsorted(window_upper_ends, next_part_lightest, side="left"))
            fp_indices = mass_order[next_molecule:end_molecule]

            drawn_smiles, window_size[fp_indices] = draw_windows(
                masses[fp_indices], loaded["mass"].to_numpy(), loaded["smiles"].to_list(), rng
            )
            tasks = [(smiles, own_fingerprints[fp_index]) for smiles, fp_index in zip(drawn_smiles, fp_indices)]
            selected = workers.starmap(select_negatives, tasks, chunksize=100)
            for fp_index, packed in zip(fp_indices, selected):
                negatives[fp_index, :len(packed)] = packed
                n_negatives[fp_index] = len(packed)
            next_molecule = end_molecule

            # keep the structures that the windows of the remaining molecules can still reach (2 x tolerance + margin)
            carried = loaded.filter(pl.col("mass") >= next_part_lightest * (1 - 3 * PPM_TOLERANCE * 1e-6))

            elapsed_minutes = (time.time() - start_time) / 60
            print(f"  part {part_number + 1}/{len(part_paths)}: {len(fp_indices):,} molecules, "
                  f"{next_molecule:,} done | {elapsed_minutes:.0f} min elapsed", flush=True)

    assert next_molecule == molecules.height, "some molecules were never handled"
    np.savez(OUTPUT_PATH, fingerprints=negatives, n_negatives=n_negatives, window_size=window_size)

    print(f"Saved {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1e9:.2f} GB)")
    print(f"window size: median {np.median(window_size):,.0f}, 10th percentile {np.percentile(window_size, 10):,.0f}")
    print(f"negatives per molecule: mean {n_negatives.mean():.1f} | "
          f"fewer than {N_NEGATIVES}: {np.mean(n_negatives < N_NEGATIVES):.1%} | none: {np.mean(n_negatives == 0):.1%}")


if __name__ == "__main__":
    main()
