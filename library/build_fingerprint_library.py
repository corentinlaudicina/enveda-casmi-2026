"""Build the fingerprint library: one Morgan fingerprint per distinct molecule of train.parquet.

train.parquet is read batch by batch (only the two label columns), so the 3 GB file never sits in memory.
Molecules appear many times (one row per spectrum), so each SMILES is fingerprinted only the first time it is seen.

Outputs (in library/data/):
- molecules.parquet: one row per molecule: fp_index, normalized_smiles, inchikey14, exact_mass (monoisotopic, Da)
- morgan_fingerprints.npy: uint8 matrix (n_molecules, 2048 / 8), bit-packed; row i <-> fp_index i
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fingerprints_MLP" / "src"))
from morgan_generator import fingerprint_from_mol  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_DIR / "data" / "enveda-CASMI26-molecule-id-mass-spectra" / "train.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"

BATCH_SIZE = 100_000  # rows read at a time; only 2 string columns, so this is a few MB

def build_fingerprint_library(train_path: Path) -> tuple[pl.DataFrame, np.ndarray]:
    """Read train.parquet sequentially and fingerprint each distinct SMILES once.

    Returns the molecule table (fp_index, normalized_smiles, inchikey14, exact_mass) and the
    bit-packed fingerprint matrix, whose row i belongs to the molecule with fp_index i.
    """
    parquet_file = pq.ParquetFile(train_path)

    seen_smiles = set()          # SMILES already handled (fingerprinted or failed)
    kept_smiles = []             # SMILES of the molecules we keep, in order
    kept_inchikey14 = []
    kept_exact_mass = []
    packed_fingerprints = []     # one packed array (256 bytes) per kept molecule
    n_rows_read = 0
    n_failed = 0

    batches = parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["normalized_smiles", "inchikey14"],
    )
    for batch in batches:
        smiles_column = batch.column("normalized_smiles").to_pylist()
        inchikey14_column = batch.column("inchikey14").to_pylist()

        for smiles, inchikey14 in zip(smiles_column, inchikey14_column):
            if smiles in seen_smiles:
                continue
            seen_smiles.add(smiles)

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                n_failed += 1
                continue
            bits = fingerprint_from_mol(mol)

            kept_smiles.append(smiles)
            kept_inchikey14.append(inchikey14)
            kept_exact_mass.append(ExactMolWt(mol))
            packed_fingerprints.append(np.packbits(bits))

        n_rows_read += batch.num_rows
        print(f"{n_rows_read:>9,} rows read, {len(kept_smiles):>7,} molecules, {n_failed} unparsable")

    molecules = pl.DataFrame({
        "fp_index": np.arange(len(kept_smiles)),
        "normalized_smiles": kept_smiles,
        "inchikey14": kept_inchikey14,
        "exact_mass": kept_exact_mass,
    })
    fingerprints = np.stack(packed_fingerprints)
    return molecules, fingerprints

if __name__ == "__main__":
    molecules, fingerprints = build_fingerprint_library(TRAIN_PATH)

    OUTPUT_DIR.mkdir(exist_ok=True)
    molecules.write_parquet(OUTPUT_DIR / "molecules.parquet")
    np.save(OUTPUT_DIR / "morgan_fingerprints.npy", fingerprints)

    print(f"Saved {molecules.height:,} molecules, fingerprint matrix {fingerprints.shape} to {OUTPUT_DIR}")
