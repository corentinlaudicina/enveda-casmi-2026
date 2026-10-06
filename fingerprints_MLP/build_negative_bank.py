"""Bank of decoys for train_MLP_resampled.py: a random BANK_FRACTION of the candidate pool (train + PubChem,
external/pool/), fingerprinted once and sorted by mass.

train_MLP_negatives.py uses 32 fixed negatives per molecule (negatives.npz): seen every epoch, the model can learn
those 32 by heart. With the bank, every batch draws fresh decoys from the molecule's mass window in the bank instead
(a median window holds ~9,300 pool structures, so ~400 bank structures).

- the molecules the MLP never trains on (validation split, natural-product test set, libraries outside
  TRAIN_LIBRARIES) are removed from the pool first, matched on the pool key, as in build_negatives.py
- the molecule itself, its stereoisomers and fingerprint collisions are NOT removed here: training masks every decoy
  whose fingerprint equals the true one
- the pool parts are sorted by mass and read in order, so the bank is sorted by mass

Output: library/negative_bank.npz (~1 GB, under the 4 GB limit of the USB disk)
- fingerprints: uint8 (n_bank, N_BITS / 8), bit-packed like morgan_fingerprints.npy
- masses: float64 (n_bank,), exact masses, ascending

Usage: python build_negative_bank.py
"""

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl

from build_negatives import N_WORKERS, held_out_keys
from morgan_generator import fingerprint
from train_MLP import LIBRARY_DIR, SEED

sys.path.insert(0, str(Path(__file__).resolve().parent / "pubchem_pool"))
from build_pool import POOL_DIR  # noqa: E402

OUTPUT_PATH = LIBRARY_DIR / "negative_bank.npz"
BANK_FRACTION = 0.045  # of ~90M pool structures: ~4M in the bank, 1 GB of fingerprints


def packed_fingerprint(smiles: str) -> np.ndarray | None:
    """Bit-packed Morgan fingerprint (N_BYTES,) uint8, or None if RDKit can't parse the SMILES."""
    bits = fingerprint(smiles)
    if bits is None:
        return None
    return np.packbits(bits)


def main():
    molecules = pl.read_parquet(LIBRARY_DIR / "molecules.parquet")
    part_paths = sorted(POOL_DIR.glob("pool_*.parquet"))
    rng = np.random.default_rng(SEED)
    start_time = time.time()

    bank_fingerprints = []
    bank_masses = []
    with Pool(N_WORKERS) as workers:
        excluded_keys = held_out_keys(molecules, workers)
        print(f"{excluded_keys.len():,} molecules never trained on, removed from the pool")

        for part_number, part_path in enumerate(part_paths):
            part = (
                pl.read_parquet(part_path, columns=["key", "smiles", "mass"])
                .filter(~pl.col("key").is_in(excluded_keys.implode()))
            )
            is_drawn = rng.random(part.height) < BANK_FRACTION
            drawn = part.filter(pl.Series(is_drawn))  # keeps the mass order

            packed = workers.map(packed_fingerprint, drawn["smiles"].to_list(), chunksize=2_000)
            parsed = np.array([fingerprint_bytes is not None for fingerprint_bytes in packed])
            bank_fingerprints.append(np.stack([fingerprint_bytes for fingerprint_bytes in packed if fingerprint_bytes is not None]))
            bank_masses.append(drawn["mass"].to_numpy()[parsed])

            elapsed_minutes = (time.time() - start_time) / 60
            print(f"  part {part_number + 1}/{len(part_paths)}: {parsed.sum():,} structures | "
                  f"{elapsed_minutes:.0f} min elapsed", flush=True)

    fingerprints = np.concatenate(bank_fingerprints)
    masses = np.concatenate(bank_masses)
    assert np.all(np.diff(masses) >= 0), "the pool parts are not sorted by mass"
    np.savez(OUTPUT_PATH, fingerprints=fingerprints, masses=masses)
    print(f"Saved {len(masses):,} structures to {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
