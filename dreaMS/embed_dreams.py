"""Compute DreaMS embeddings for a list of train spectra.

Runs in the DreaMS environment (Python 3.11):
    .venv-dreams/bin/python embed_dreams.py dreams_cache/rows_to_embed.parquet dreams_cache/embeddings.npz

Input: a parquet file with a `row_id` column (positions in train.parquet).
Output: an .npz file with `row_ids` and unit-length `embeddings` (one 1024-d row per spectrum).
"""

import os
import sys
import time

import numpy as np
import polars as pl
import torch

import dreams.utils.data as dreams_data
import dreams.utils.dformats as dreams_formats
from dreams.api import PreTrainedModel
from dreams.definitions import DREAMS_EMBEDDING

TRAIN_PATH = "/Volumes/HDLAUDICINA/enveda-CASMI26-molecule-id-mass-spectra/data/enveda-CASMI26-molecule-id-mass-spectra" #"data/enveda-CASMI26-molecule-id-mass-spectra/train.parquet"
BATCH_SIZE = 64  # larger batches run out of memory on a 16 GB Mac
CHUNK_SIZE = 10_000  # spectra read from disk at a time

DEFAULT_ROWS_PATH = "dreams_cache/rows_to_embed.parquet"
DEFAULT_OUTPUT_PATH = "dreams_cache/embeddings.npz"

class DreamsEmbedder:
    """Wraps the pre-trained DreaMS embedding model."""

    def __init__(self):
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model = PreTrainedModel.from_name(DREAMS_EMBEDDING).model.eval().to(self.device)
        # Keeps the 100 strongest peaks and prepends the precursor, as during DreaMS training 
        # (230 is the median, but transformer was trained on the top 100)
        self.preprocessor = dreams_data.SpectrumPreprocessor(dformat=dreams_formats.DataFormatA(), n_highest_peaks=100)

    def prepare(self, mzs, intensities, precursor_mz):
        peaks = np.array([mzs, intensities])
        return self.preprocessor(peaks, prec_mz=precursor_mz, high_form=False)

    def embed(self, prepared_spectra):
        """Embed a list of prepared spectra, returning unit-length vectors."""
        outputs = []
        for start in range(0, len(prepared_spectra), BATCH_SIZE):
            batch = np.stack(prepared_spectra[start:start + BATCH_SIZE])
            batch = torch.tensor(batch).to(self.device)
            with torch.inference_mode():
                outputs.append(self.model(batch).cpu().numpy())
        embeddings = np.concatenate(outputs)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms


def main(rows_path, output_path):
    row_ids = pl.read_parquet(rows_path)["row_id"].to_list()
    print(f"{len(row_ids)} spectra to embed")

    embedder = DreamsEmbedder()
    print(f"model loaded on {embedder.device}")

    all_row_ids = []
    all_embeddings = []
    started = time.time()
    for start in range(0, len(row_ids), CHUNK_SIZE):
        chunk_ids = row_ids[start:start + CHUNK_SIZE]
        rows = (
            pl.scan_parquet(TRAIN_PATH)
            .with_row_index("row_id")
            .filter(pl.col("row_id").is_in(chunk_ids))
            .select("row_id", "ms2_mzs", "ms2_normalized_intensities", "precursor_mz")
            .collect()
        )

        prepared = []
        for row in rows.iter_rows(named=True):
            prepared.append(embedder.prepare(row["ms2_mzs"], row["ms2_normalized_intensities"], row["precursor_mz"]))

        all_embeddings.append(embedder.embed(prepared))
        all_row_ids.extend(rows["row_id"].to_list())

        done = len(all_row_ids)
        rate = done / (time.time() - started)
        print(f"{done}/{len(row_ids)} spectra, {rate:.0f}/s, {(len(row_ids) - done) / rate / 60:.0f} min left", flush=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, row_ids=np.array(all_row_ids), embeddings=np.concatenate(all_embeddings).astype(np.float32))
    print(f"saved {output_path}")


if __name__ == "__main__":

    if len(sys.argv) == 3:
        rows_path = sys.argv[1]
        output_path = sys.argv[2]
    else:
        rows_path = DEFAULT_ROWS_PATH
        output_path = DEFAULT_OUTPUT_PATH
    main(rows_path, output_path)
