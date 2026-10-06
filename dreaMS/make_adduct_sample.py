"""Pick train spectra covering every common adduct, for the adduct-pair study (adduct_pairs.ipynb).

Writes the rows that still need a DreaMS embedding; embed them with:
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 ../.venv-dreams/bin/python embed_dreams.py \
        dreams_cache/adduct_rows_to_embed.parquet dreams_cache/adduct_embeddings.npz
"""

import numpy as np
import polars as pl
import os

HD_path = "/Volumes/HDLAUDICINA/enveda-CASMI26-molecule-id-mass-spectra"

TRAIN_PATH = os.path.join(HD_path,"data/enveda-CASMI26-molecule-id-mass-spectra/train.parquet")
CACHE_PATH = os.path.join(HD_path,"dreaMS/dreams_cache/embedding_cache.npz")
SAMPLE_PATH = os.path.join(HD_path,"dreaMS/dreams_cache/adduct_sample.parquet") # every row of the sample
TO_EMBED_PATH = os.path.join(HD_path,"dreaMS/dreams_cache/adduct_rows_to_embed.parquet") # the rows not embedded yet

MIN_SPECTRA_PER_ADDUCT = 1_000  # rarer adducts are left out
SPECTRA_PER_MOLECULE_ADDUCT = 3
MOLECULES_PER_ADDUCT = 2_000
SEED = 0

metadata = (
    pl.scan_parquet(TRAIN_PATH)
    .with_row_index("row_id")
    .select("row_id", "inchikey14", "adduct", "ionization_mode")
    .collect()
)

# Drop labels whose charge sign contradicts the polarity, e.g. "[M-H]-" in positive mode
sign_matches_polarity = (
    (pl.col("adduct").str.ends_with("+") & (pl.col("ionization_mode") == "positive"))
    | (pl.col("adduct").str.ends_with("-") & (pl.col("ionization_mode") == "negative"))
)
metadata = metadata.filter(sign_matches_polarity)

common_adducts = (
    metadata
    .group_by("adduct")
    .len()
    .filter(pl.col("len") >= MIN_SPECTRA_PER_ADDUCT)
    ["adduct"]
)
metadata = metadata.filter(pl.col("adduct").is_in(common_adducts.implode()))
print(f"{len(common_adducts)} adducts with at least {MIN_SPECTRA_PER_ADDUCT} spectra")

# Molecules measured with at least two of these adducts: only they give cross-adduct pairs
multi_adduct_molecules = (
    metadata
    .group_by("inchikey14")
    .agg(n_adducts=pl.col("adduct").n_unique())
    .filter(pl.col("n_adducts") >= 2)
    ["inchikey14"]
)
metadata = metadata.filter(pl.col("inchikey14").is_in(multi_adduct_molecules.implode()))

# Up to MOLECULES_PER_ADDUCT random molecules per adduct
chosen_molecule_adducts = (
    metadata
    .select("adduct", "inchikey14")
    .unique()
    .sort("adduct", "inchikey14")
    .with_columns(rank=pl.int_range(pl.len()).shuffle(seed=SEED).over("adduct"))
    .filter(pl.col("rank") < MOLECULES_PER_ADDUCT)
    .drop("rank")
)

# Up to SPECTRA_PER_MOLECULE_ADDUCT random spectra per (molecule, adduct)
sample = (
    metadata
    .join(chosen_molecule_adducts, on=["adduct", "inchikey14"])
    .sort("row_id")
    .with_columns(rank=pl.int_range(pl.len()).shuffle(seed=SEED).over("inchikey14", "adduct"))
    .filter(pl.col("rank") < SPECTRA_PER_MOLECULE_ADDUCT)
    .drop("rank")
)

already_embedded = np.load(CACHE_PATH)["row_ids"]
to_embed = sample.filter(~pl.col("row_id").is_in(already_embedded.tolist()))

pl.Config.set_tbl_rows(40)
print(
    sample
    .group_by("adduct")
    .agg(spectra=pl.len(), molecules=pl.col("inchikey14").n_unique())
    .sort("spectra", descending=True)
)
print(f"{sample.height:,} spectra in the sample, {to_embed.height:,} still to embed")

sample.select("row_id").write_parquet(SAMPLE_PATH)
to_embed.select("row_id").write_parquet(TO_EMBED_PATH)
