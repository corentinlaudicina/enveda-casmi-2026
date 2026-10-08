"""Step 4: the candidate pool = train structures + the PubChem tier, one row per structure key.

Key: standard InChIKey first block (metric_key.standard_key). The metric's tautomer-canonical key costs ~13 h on
the full tier, so it is computed later, only for the candidates that matter (see evaluate_pool.py).

Pass 1, the tier in chunks of ~CHUNK_SIZE structures (it is sorted by mass; a chunk ends where the mass changes,
so all copies of a structure, which have the same mass, fall in the same chunk):
  standard key of every SMILES (N_WORKERS processes) -> drop failures -> keep the first row per key
  -> drop the structures that are in train (they come from the train side, with in_pubchem = True)
  -> temporary part, sorted by mass.
Pass 2, each part: add the train structures of its mass range, sort by mass, write the final part.

Output (external/pool/, parts because the external disk is FAT32: no file above 4 GB):
  pool_NNN.parquet with columns
  - key: standard InChIKey first block
  - smiles: train's normalized_smiles, or the tier's SMILES
  - mass: monoisotopic mass (Da)
  - source: "train" or "pubchem"
  - in_pubchem: True if the structure is in the PubChem tier (always True for source "pubchem").
    Null for the train structures below 471.19 Da: the first run, interrupted after 17 chunks, did not save
    which train structures those chunks held (recomputing it costs ~100 min). evaluate_pool.py computes it
    for its query molecules.
Parts are in mass order: part i's masses are all below part i + 1's.

Usage: python build_pool.py
"""

import time
from multiprocessing import Pool

import numpy as np
import polars as pl

from metric_key import standard_key
from pubchem_tier import PROJECT_DIR, PubChemTier

LIBRARY_DIR = PROJECT_DIR / "library" / "data"
POOL_DIR = PROJECT_DIR / "external" / "pool"
TEMPORARY_DIR = POOL_DIR / "pass1"
CHUNK_SIZE = 5_000_000
N_WORKERS = 8


def compute_keys(smiles: list[str], workers: Pool) -> list[str | None]:
    return workers.map(standard_key, smiles, chunksize=5_000)


def load_train_structures(workers: Pool) -> pl.DataFrame:
    """One row per train structure: key, smiles, mass, source = "train"."""
    molecules = pl.read_parquet(LIBRARY_DIR / "molecules.parquet")
    keys = compute_keys(molecules["normalized_smiles"].to_list(), workers)
    return (
        molecules
        .select(pl.col("normalized_smiles").alias("smiles"), pl.col("exact_mass").alias("mass"))
        .with_columns(pl.Series("key", keys, dtype=pl.String), pl.lit("train").alias("source"))
        .drop_nulls("key")
        .unique("key", keep="first", maintain_order=True)
        .select("key", "smiles", "mass", "source")
    )


def chunk_bounds(tier: PubChemTier) -> list[tuple[int, int]]:
    """(start, end) tier positions of the chunks: about CHUNK_SIZE each, cut where the mass changes."""
    bounds = []
    start = 0
    while start < len(tier):
        end = min(start + CHUNK_SIZE, len(tier))
        if end < len(tier):
            # move the cut back to the first structure of this mass, so equal masses stay together
            end = int(np.searchsorted(tier.masses[start:end + 1], tier.masses[end], side="left")) + start
        assert end > start, "a whole chunk has a single mass: increase CHUNK_SIZE"
        bounds.append((start, end))
        start = end
    return bounds


def pass1_chunk(tier: PubChemTier, start: int, end: int, train_keys: pl.Series, workers: Pool) -> tuple[pl.DataFrame, pl.Series, int]:
    """PubChem structures of one chunk, deduplicated and without train structures.

    Returns (chunk, keys of the train structures found in this chunk, number of SMILES that failed).
    """
    smiles = tier.smiles(start, end)
    keys = compute_keys(smiles, workers)
    chunk = pl.DataFrame({
        "key": pl.Series(keys, dtype=pl.String),
        "smiles": smiles,
        "mass": np.asarray(tier.masses[start:end]),
    })
    n_failed = chunk["key"].null_count()
    chunk = chunk.drop_nulls("key").unique("key", keep="first", maintain_order=True)

    is_train = chunk["key"].is_in(train_keys.implode())
    train_keys_found = chunk.filter(is_train)["key"]
    chunk = chunk.filter(~is_train).with_columns(pl.lit("pubchem").alias("source"))
    return chunk, train_keys_found, n_failed


def train_found_path(part_number: int):
    """Keys of the train structures found in a chunk during pass 1 (needed by pass 2 for in_pubchem)."""
    return TEMPORARY_DIR / f"train_found_{part_number:03d}.parquet"


def main():
    POOL_DIR.mkdir(exist_ok=True)
    TEMPORARY_DIR.mkdir(exist_ok=True)
    tier = PubChemTier()
    bounds = chunk_bounds(tier)
    print(f"{len(tier):,} tier structures in {len(bounds)} chunks")

    with Pool(N_WORKERS) as workers:
        train = load_train_structures(workers)
        print(f"{train.height:,} train structures")

        # pass 1 (resumable: a chunk whose temporary part exists is skipped)
        n_failed_total = 0
        n_kept_total = 0
        start_time = time.time()
        for part_number, (start, end) in enumerate(bounds):
            temporary_path = TEMPORARY_DIR / f"pool_{part_number:03d}.parquet"
            if temporary_path.exists():
                print(f"  part {part_number + 1}/{len(bounds)}: already done, skipped")
                continue
            chunk, train_keys_found, n_failed = pass1_chunk(tier, start, end, train["key"], workers)
            train_keys_found.to_frame("key").write_parquet(train_found_path(part_number))
            chunk.write_parquet(temporary_path)  # written last: its existence marks the chunk as done
            n_failed_total += n_failed
            n_kept_total += chunk.height

            elapsed_minutes = (time.time() - start_time) / 60
            print(f"  part {part_number + 1}/{len(bounds)}: {end - start:,} -> {chunk.height:,} kept | "
                  f"{elapsed_minutes:.0f} min elapsed", flush=True)

    print(f"pass 1 done (this run: {n_kept_total:,} PubChem-only structures kept, {n_failed_total:,} failed SMILES)")

    # pass 2: train structures into the part of their mass range
    for part_number, (start, end) in enumerate(bounds):
        lowest = -np.inf if part_number == 0 else float(tier.masses[start])
        highest = np.inf if part_number == len(bounds) - 1 else float(tier.masses[end])
        train_rows = train.filter((pl.col("mass") >= lowest) & (pl.col("mass") < highest))
        if train_found_path(part_number).exists():
            found_keys = pl.read_parquet(train_found_path(part_number))["key"]
            train_rows = train_rows.with_columns(pl.col("key").is_in(found_keys.implode()).alias("in_pubchem"))
        else:
            # chunk of the first, interrupted run: which train structures it held was not saved
            train_rows = train_rows.with_columns(pl.lit(None, dtype=pl.Boolean).alias("in_pubchem"))

        temporary_path = TEMPORARY_DIR / f"pool_{part_number:03d}.parquet"
        pubchem_rows = pl.read_parquet(temporary_path).with_columns(pl.lit(True).alias("in_pubchem"))
        part = pl.concat([train_rows, pubchem_rows.select(train_rows.columns)]).sort("mass")
        part.write_parquet(POOL_DIR / f"pool_{part_number:03d}.parquet", compression="zstd")
        temporary_path.unlink()
        train_found_path(part_number).unlink(missing_ok=True)
    TEMPORARY_DIR.rmdir()

    pool = pl.scan_parquet(POOL_DIR / "pool_*.parquet")
    summary = pool.group_by("source").agg(pl.len().alias("structures")).collect()
    print(summary)
    size_gb = sum(path.stat().st_size for path in POOL_DIR.glob("pool_*.parquet")) / 1e9
    print(f"pool: {pool.select(pl.len()).collect().item():,} structures, {size_gb:.2f} GB in {len(bounds)} parts")


if __name__ == "__main__":
    main()
