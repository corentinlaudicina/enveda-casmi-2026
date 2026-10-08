"""Candidate library from ChEBI and LIPID MAPS, in the same format as the train and COCONUT libraries.

ChEBI (biologically relevant small molecules) and LIPID MAPS (lipids) add metabolites and lipids that COCONUT, a natural
products database, may lack. Downloads, each in its own folder of data/:
- data/chebi/chebi.sdf.gz: https://ftp.ebi.ac.uk/pub/databases/chebi/SDF/chebi.sdf.gz (all stars)
- data/lipidmaps/LMSD.sdf.zip: https://www.lipidmaps.org/files/?file=LMSD&ext=sdf.zip

ChEBI is an ontology, not a clean structure list, so every structure is standardised before use:
- largest fragment only: drops counter-ions of salts and water of hydrates
- neutralised: ChEBI often stores the charged form a molecule takes at pH 7.3 (carboxylate, ammonium). Its inchikey14
  is the neutral molecule's, but its exact mass is a proton off, which would put it in the wrong mass window.
- dropped: unparsable records, generic structures with R groups (* atoms: "a fatty acid", polymers), isotope-labelled
  structures (same inchikey14 as the unlabelled molecule, different mass) and exact masses above MAX_EXACT_MASS.
LIPID MAPS goes through the same steps (its structures are already clean: they rarely change anything).

- One structure per inchikey14 (first block of RDKit's standard InChIKey of the standardised molecule, as COCONUT's),
  ChEBI first, then LIPID MAPS, each in file order.
- Structures already in train or COCONUT are kept, flagged with in_train / in_coconut: the natural-product scenario of
  evaluate_ranking.py removes its queries from the train part of the pool, and must still find them here if ChEBI or
  LIPID MAPS has them. Ranking keeps one entry per inchikey14, so the duplicates cost nothing else.
- Fingerprint and exact mass are computed with RDKit from the standardised molecule, as for the train molecules.

Outputs (in library/data/):
- bio_molecules.parquet: source ("chebi" or "lipidmaps"), source_id, normalized_smiles (RDKit SMILES of the
  standardised molecule), inchikey14, exact_mass, in_train, in_coconut
- bio_fingerprints.npy: uint8 (n_molecules, 2048 / 8), bit-packed; row i <-> row i of the parquet

Usage: python build_bio_library.py
"""

import gzip
import sys
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.MolStandardize import rdMolStandardize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fingerprints_MLP" / "src"))
from morgan_generator import fingerprint_from_mol  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHEBI_PATH = PROJECT_DIR / "data" / "chebi" / "chebi.sdf.gz"
LIPIDMAPS_PATH = PROJECT_DIR / "data" / "lipidmaps" / "LMSD.sdf.zip"
LIBRARY_DIR = Path(__file__).resolve().parent / "data"

MAX_EXACT_MASS = 2000.0  # proteins, polymers and other giants are not test molecules

RDLogger.DisableLog("rdApp.*")  # ChEBI has many records RDKit warns about
LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()
UNCHARGER = rdMolStandardize.Uncharger()


def standardise(mol: Chem.Mol) -> tuple[Chem.Mol | None, str]:
    """(standardised molecule, "kept") or (None, reason it is dropped)."""
    if mol is None or mol.GetNumAtoms() == 0:
        return None, "unparsable or no structure"
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        return None, "generic structure (* atoms)"
    if any(atom.GetIsotope() != 0 for atom in mol.GetAtoms()):
        return None, "isotope-labelled"

    try:
        mol = LARGEST_FRAGMENT.choose(mol)
        mol = UNCHARGER.uncharge(mol)
    except Exception:
        return None, "standardisation failed"

    if ExactMolWt(mol) > MAX_EXACT_MASS:
        return None, f"exact mass above {MAX_EXACT_MASS:g}"
    return mol, "kept"


def read_structures(source: str, supplier, id_property: str) -> tuple[pl.DataFrame, list[np.ndarray]]:
    """Standardised structures of one SDF supplier: (one row per kept record, their packed fingerprints)."""
    rows = []
    packed_fingerprints = []
    reasons = {}
    for n_done, mol in enumerate(supplier, start=1):
        source_id = mol.GetProp(id_property) if mol is not None and mol.HasProp(id_property) else None
        mol, reason = standardise(mol)
        if mol is not None:
            inchikey14 = Chem.MolToInchiKey(mol)[:14]
            if not inchikey14:
                mol, reason = None, "no InChIKey"
        reasons[reason] = reasons.get(reason, 0) + 1
        if mol is None:
            continue

        rows.append({
            "source": source,
            "source_id": source_id,
            "normalized_smiles": Chem.MolToSmiles(mol),
            "inchikey14": inchikey14,
            "exact_mass": ExactMolWt(mol),
        })
        packed_fingerprints.append(np.packbits(fingerprint_from_mol(mol)))

        if n_done % 50_000 == 0:
            print(f"  {source}: {n_done:>7,} records")

    print(f"{source}: " + ", ".join(f"{reason} {count:,}" for reason, count in sorted(reasons.items())))
    return pl.DataFrame(rows), packed_fingerprints


def build_bio_library() -> tuple[pl.DataFrame, np.ndarray]:
    with gzip.open(CHEBI_PATH, "rb") as chebi_file:
        chebi, chebi_fingerprints = read_structures(
            "chebi", Chem.ForwardSDMolSupplier(chebi_file, sanitize=True), "ChEBI ID"
        )
    with zipfile.ZipFile(LIPIDMAPS_PATH) as archive, archive.open("structures.sdf") as lipidmaps_file:
        lipidmaps, lipidmaps_fingerprints = read_structures(
            "lipidmaps", Chem.ForwardSDMolSupplier(lipidmaps_file, sanitize=True), "LM_ID"
        )

    # one structure per inchikey14: the first one, ChEBI before LIPID MAPS
    molecules = pl.concat([chebi, lipidmaps]).with_row_index("row")
    molecules = molecules.unique(subset="inchikey14", keep="first", maintain_order=True)
    packed_fingerprints = np.stack(chebi_fingerprints + lipidmaps_fingerprints)[molecules["row"].to_numpy()]

    train_inchikey14 = pl.read_parquet(LIBRARY_DIR / "molecules.parquet")["inchikey14"]
    coconut_inchikey14 = pl.read_parquet(LIBRARY_DIR / "coconut_molecules.parquet")["inchikey14"]
    molecules = molecules.drop("row").with_columns(
        pl.col("inchikey14").is_in(train_inchikey14.implode()).alias("in_train"),
        pl.col("inchikey14").is_in(coconut_inchikey14.implode()).alias("in_coconut"),
    )
    return molecules, packed_fingerprints


def print_summary(molecules: pl.DataFrame):
    print(f"\n{molecules.height:,} distinct inchikey14 in ChEBI + LIPID MAPS")
    summary = (
        molecules.group_by("source")
        .agg(
            pl.len().alias("structures"),
            pl.col("in_train").sum().alias("in train"),
            pl.col("in_coconut").sum().alias("in COCONUT"),
            (~pl.col("in_train") & ~pl.col("in_coconut")).sum().alias("new to the pool"),
        )
        .sort("source")
    )
    print(summary)


if __name__ == "__main__":
    molecules, fingerprints = build_bio_library()
    print_summary(molecules)
    molecules.write_parquet(LIBRARY_DIR / "bio_molecules.parquet")
    np.save(LIBRARY_DIR / "bio_fingerprints.npy", fingerprints)
    print(f"Saved {molecules.height:,} molecules, fingerprint matrix {fingerprints.shape} to {LIBRARY_DIR}")
