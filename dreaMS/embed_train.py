"""DreaMS embeddings of every train spectrum whose adduct is common (at least 1,000 spectra).

Runs in the DreaMS environment (Python 3.11), from the dreaMS folder:
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 caffeinate -i .venv-dreams/bin/python embed_train.py

About 2.53M spectra: 10 to 28 hours on the M3 Mac. The work is split into parts of PART_SIZE
train rows, each saved to its own file as soon as it is done, so:
- no file comes near the 4 GB limit of the FAT32 disk (one part is about 270 MB);
- the run can be stopped (Ctrl-C) and started again: finished parts are skipped.

Output, in OUTPUT_DIR:
- selected_rows.parquet: row_id and adduct of every spectrum to embed;
- part_0000000.npz, part_0131072.npz, ...: `row_ids` and unit-length `embeddings` (float16).
Embeddings are stored as float16 to halve the size (5 GB instead of 10 GB in total); the model
itself runs in float32. Cast back with `.astype(np.float32)` after loading.
"""

import os
import time

import numpy as np
import polars as pl

from embed_dreams import DreamsEmbedder

HD_PATH = "/Volumes/HDLAUDICINA/enveda-CASMI26-molecule-id-mass-spectra"
TRAIN_PATH = os.path.join(HD_PATH, "data/enveda-CASMI26-molecule-id-mass-spectra/train.parquet")
OUTPUT_DIR = os.path.join(HD_PATH, "dreaMS/dreams_train_embeddings")

MIN_SPECTRA_PER_ADDUCT = 1_000  # rarer adducts are left out
PART_SIZE = 131_072  # train rows per part: one parquet row group, so each part is read in one go
PROGRESS_EVERY = 5_000  # spectra between progress messages


def select_rows():
    """Row ids of the spectra to embed: common adduct, and a charge sign matching the polarity."""
    metadata = (
        pl.scan_parquet(TRAIN_PATH)
        .with_row_index("row_id")
        .select("row_id", "adduct", "ionization_mode")
        .collect()
    )

    # Drop labels whose charge sign contradicts the polarity, e.g. "[M-H]-" in positive mode
    sign_matches_polarity = (
        (pl.col("adduct").str.ends_with("+") & (pl.col("ionization_mode") == "positive"))
        | (pl.col("adduct").str.ends_with("-") & (pl.col("ionization_mode") == "negative"))
    )
    metadata = metadata.filter(sign_matches_polarity)

    adduct_counts = metadata.group_by("adduct").len()
    common_adducts = adduct_counts.filter(pl.col("len") >= MIN_SPECTRA_PER_ADDUCT)["adduct"]
    selected = metadata.filter(pl.col("adduct").is_in(common_adducts.implode()))

    print(f"{len(common_adducts)} adducts with at least {MIN_SPECTRA_PER_ADDUCT:,} spectra:")
    print(", ".join(sorted(common_adducts.to_list())))
    print(f"{selected.height:,} spectra to embed, out of {metadata.height:,}")
    return selected.select("row_id", "adduct")


def embed_part(embedder, start, selected_row_ids):
    """Embed the selected spectra among train rows start to start + PART_SIZE."""
    rows = (
        pl.scan_parquet(TRAIN_PATH)
        .with_row_index("row_id")
        .slice(start, PART_SIZE)
        .select("row_id", "ms2_mzs", "ms2_normalized_intensities", "precursor_mz")
        .collect()
    )
    rows = rows.filter(pl.col("row_id").is_in(selected_row_ids.implode()))

    prepared = []
    for row in rows.iter_rows(named=True):
        prepared.append(embedder.prepare(row["ms2_mzs"], row["ms2_normalized_intensities"], row["precursor_mz"]))

    # Embed in pieces, only to print progress: a whole part takes about an hour
    embeddings = []
    started = time.time()
    for piece_start in range(0, len(prepared), PROGRESS_EVERY):
        embeddings.append(embedder.embed(prepared[piece_start:piece_start + PROGRESS_EVERY]))
        done = min(piece_start + PROGRESS_EVERY, len(prepared))
        rate = done / (time.time() - started)
        print(f"   {done:,}/{len(prepared):,} spectra, {rate:.0f}/s", flush=True)

    return rows["row_id"].to_numpy(), np.concatenate(embeddings)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    selected_path = os.path.join(OUTPUT_DIR, "selected_rows.parquet")
    if os.path.exists(selected_path):
        selected = pl.read_parquet(selected_path)
        print(f"{selected.height:,} spectra to embed (from {selected_path})")
    else:
        selected = select_rows()
        selected.write_parquet(selected_path)

    n_train_rows = pl.scan_parquet(TRAIN_PATH).select(pl.len()).collect().item()
    part_starts = list(range(0, n_train_rows, PART_SIZE))

    embedder = DreamsEmbedder()
    print(f"model loaded on {embedder.device}")

    run_started = time.time()
    for part_number, start in enumerate(part_starts, start=1):
        part_path = os.path.join(OUTPUT_DIR, f"part_{start:07d}.npz")
        if os.path.exists(part_path):
            print(f"part {part_number}/{len(part_starts)}: already done, skipped")
            continue

        print(f"part {part_number}/{len(part_starts)}: train rows {start:,} to {min(start + PART_SIZE, n_train_rows):,}")
        row_ids, embeddings = embed_part(embedder, start, selected["row_id"])

        # Write under a temporary name, then rename: a part file only exists once it is complete
        temporary_path = os.path.join(OUTPUT_DIR, f"part_{start:07d}.tmp.npz")
        np.savez(temporary_path, row_ids=row_ids, embeddings=embeddings.astype(np.float16))
        os.replace(temporary_path, part_path)
        print(f"   saved {part_path} ({(time.time() - run_started) / 3600:.1f} h since start)")

    print("all parts done")


if __name__ == "__main__":
    main()
