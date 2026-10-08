"""Candidate library from COCONUT (natural products database), in the same format as the train library.

- One structure per inchikey14: COCONUT lists stereoisomers separately (739k SMILES, 480k inchikey14),
  but the metric ignores stereo, so extra stereoisomers would only duplicate candidates.
- Isotope labels: COCONUT also lists labelled variants (glycine-d5, [13C] compounds: 217 SMILES, 205 inchikey14).
  They share the inchikey14 of the unlabelled molecule but not its mass, so a kept labelled variant would sit in the
  wrong mass window. The unlabelled entry of an inchikey14 is preferred; the 28 inchikey14 that COCONUT has only in
  labelled form get their labels removed (mass, fingerprint and SMILES of the unlabelled molecule, RDKit SMILES).
- inchikey14 = first block of COCONUT's standard InChIKey (identical to RDKit's on a 2,000-structure check).
- Fingerprint and exact mass are computed with RDKit, exactly as for the train molecules.

Outputs (in library/data/):
- coconut_molecules.parquet: coconut_id, normalized_smiles (COCONUT's canonical SMILES), inchikey14, exact_mass
- coconut_fingerprints.npy: uint8 (n_molecules, 2048 / 8), bit-packed; row i <-> row i of the parquet
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fingerprints_MLP" / "src"))
from morgan_generator import fingerprint_from_mol  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
COCONUT_PATH = PROJECT_DIR / "data" / "enveda-CASMI26-molecule-id-mass-spectra" / "coconut_csv-10-2026.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
ISOTOPE_LABEL = r"\[\d+[A-Z]"  # a SMILES bracket atom with a mass number: [2H], [13C], [13CH3]


def without_isotope_labels(mol: Chem.Mol) -> Chem.Mol:
    """The same molecule with every atom at natural isotopic composition (labelled hydrogens become implicit)."""
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    return Chem.RemoveHs(mol)


def build_coconut_library(coconut_path: Path) -> tuple[pl.DataFrame, np.ndarray]:
    coconut = (
        pl.read_csv(coconut_path, columns=["identifier", "canonical_smiles", "standard_inchi_key"])
        .with_columns(
            pl.col("standard_inchi_key").str.slice(0, 14).alias("inchikey14"),
            pl.col("canonical_smiles").str.contains(ISOTOPE_LABEL).alias("is_labelled"),
        )
        # unlabelled entries first, then a fixed order, so the kept stereoisomer is reproducible
        .sort("is_labelled", "identifier")
        .unique(subset="inchikey14", keep="first", maintain_order=True)
    )
    print(f"{coconut.height:,} distinct inchikey14 in COCONUT, {coconut['is_labelled'].sum()} only as labelled variants")

    kept_ids = []
    kept_smiles = []
    kept_inchikey14 = []
    kept_exact_mass = []
    packed_fingerprints = []
    n_failed = 0

    rows = coconut.select("identifier", "canonical_smiles", "inchikey14", "is_labelled").iter_rows()
    for n_done, (coconut_id, smiles, inchikey14, is_labelled) in enumerate(rows, start=1):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            n_failed += 1
            continue
        if is_labelled:
            mol = without_isotope_labels(mol)
            smiles = Chem.MolToSmiles(mol)

        kept_ids.append(coconut_id)
        kept_smiles.append(smiles)
        kept_inchikey14.append(inchikey14)
        kept_exact_mass.append(ExactMolWt(mol))
        packed_fingerprints.append(np.packbits(fingerprint_from_mol(mol)))

        if n_done % 100_000 == 0:
            print(f"{n_done:>8,} structures, {n_failed} unparsable")

    molecules = pl.DataFrame({
        "coconut_id": kept_ids,
        "normalized_smiles": kept_smiles,
        "inchikey14": kept_inchikey14,
        "exact_mass": kept_exact_mass,
    })
    return molecules, np.stack(packed_fingerprints)


if __name__ == "__main__":
    molecules, fingerprints = build_coconut_library(COCONUT_PATH)
    molecules.write_parquet(OUTPUT_DIR / "coconut_molecules.parquet")
    np.save(OUTPUT_DIR / "coconut_fingerprints.npy", fingerprints)
    print(f"Saved {molecules.height:,} COCONUT molecules, fingerprint matrix {fingerprints.shape} to {OUTPUT_DIR}")
