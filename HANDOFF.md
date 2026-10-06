# Project handoff: Enveda CASMI 2026

State of the project as of 2026-10-02, for the next Claude session. Read this first; everything below was measured unless marked as a hypothesis.

## The competition
- Kaggle "Enveda CASMI 2026 – Molecule ID from Mass Spectra". Predict 2D structures (SMILES) from LC-MS/MS spectra.
- **Metric:** MRR@25. Up to 25 SMILES per `molecule_id`, best first, `;`-separated. A guess is correct if its InChIKey14 (first block, after RDKit tautomer canonicalization, RDKit pinned at 2026.03.3) matches the answer's. Stereo and tautomers don't matter.
- **Code competition:** submit a notebook, ≤ 9 h, **no internet**. Freely available external data and pretrained models are allowed (must be uploaded as Kaggle datasets beforehand). Final deadline 2026-12-14.
- The official metric notebook is linked on the competition page. It has **not** been copied yet; local scoring compares the train-provided `inchikey14`, which is an approximation.

## Data (`data/enveda-CASMI26-molecule-id-mass-spectra/`)
- `train.parquet` (3 GB): 2,539,608 spectra, 275,810 distinct `inchikey14`. Columns include `normalized_smiles`, `inchikey14`, `molecular_formula`, `adduct`, `ionization_mode`, `precursor_mz`, `ms2_mzs`, `ms2_normalized_intensities` (lists, base peak = 1), `collision_energy_orig`, `collision_energy_ev`, `ingest_lib`.
  - Loading every peak list needs ~9 GB of RAM (16 GB Mac): use `pl.scan_parquet` and load peaks only for the rows needed.
  - `enveda-180` (1.15M spectra, 182,941 molecules, timsTOF) is clean and test-like. The public libraries (pluskal, riken, gnps, massbank, mona, ...) contain bad rows: ppm errors > 20, fragments above the precursor, < 3 peaks.
  - `collision_energy_orig` is raw text, negative in negative mode on Enveda's instrument (sign convention). Use `collision_energy_ev` (cleaned, never negative).
- `test.parquet`: 1,213 spectra, 400 molecules, all timsTOF, energies 20/40/60 eV and merged `20,40,60`.
  - **It is a placeholder:** every one of its spectra appears verbatim in train (`enveda-180`). Kaggle swaps in the hidden test when the notebook reruns. **Never evaluate methods on it.** Evaluate on held-out train molecules, and on the leaderboard.
- Each test molecule's neutral mass (precursor − adduct shift) agrees across its spectra within 5.5 ppm: the mass filter is reliable.

## Environments
- `.venv` (Python 3.14): polars, numpy, scikit-learn, umap-learn, matplotlib, duckdb, uv. **No RDKit.**
- `.venv-dreams` (Python 3.11, created with uv): DreaMS and its pinned deps (torch 2.2.1, numpy 1.25, rdkit 2023.09.6, not the metric's pinned version), plus polars.
  - DreaMS weights are downloaded inside `.venv-dreams/lib/python3.11/site-packages/dreams/models/pretrained/`.
  - The machine's Hugging Face token has expired: set `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, or downloads fail with 401.
  - DreaMS on the M3 Mac: 25–67 spectra/s on MPS, batch ≤ 64 (larger runs out of memory), float16 gives NaN.

## Files
| File | What it is |
|---|---|
| `exploration.ipynb` | User's duckdb cells on train, then a polars exploration of the test set |
| `baseline.ipynb` | Cosine library-search baseline: validation (known/novel), DreaMS comparison section, `submission.csv` writer, test-leak check. Local paths, not yet adapted to Kaggle |
| `submission.csv` | Output of `baseline.ipynb` on the placeholder test. Valid format, meaningless score |
| `library_search.py` | Main experiment script (run in `.venv-dreams`). Own train split, scorers × comparison rules, both scenarios. Results in `results/library_search.csv` |
| `embed_dreams.py` | Standalone DreaMS embedding of a list of train rows |
| `exploration_embeddings.ipynb` | Study of the DreaMS embedding space: load, tools, distances, PCA vs UMAP, maps, distances after reduction, isomer test |
| `dreams_cache/embedding_cache.npz` | ~127k DreaMS embeddings keyed by train `row_id` (used by `library_search.py`, seed 0 split) |
| `dreams_cache/embeddings.npz` | 121k embeddings from the first DreaMS run (used by the notebooks) |
| `figures/` | Saved PCA/UMAP maps |
| `kaggle_submission/` | Kaggle-ready baseline: `baseline_kaggle.ipynb` (finds data under `/kaggle/input`, writes `/kaggle/working/submission.csv`, falls back safely on unknown adducts and empty candidate lists) and `kernel-metadata.json` (kernel `cocolau/casmi-baseline`, internet off, competition data attached) |

## Approach so far: library search
For a query molecule: (1) candidates = train structures whose exact mass (from the formula) is within 10 ppm of its neutral mass, window widened up to 50 ppm until ≥ 25; (2) compare each query spectrum with the candidate's library spectra of the same adduct and energy (fallback: same polarity); (3) candidate score = mean over query spectra of the best similarity; (4) rank, top 25.

Evaluation (`library_search.py`, 200 held-out `enveda-180` molecules, 3 spectra each, seed 0). **known** = the molecule's other spectra stay in the library; **novel** = removed.

| Scorer | known MRR@25 (top-1) | novel MRR@25 |
|---|---|---|
| cosine on peaks (0.01 m/z, sqrt intensities, peaks < 1% and precursor region dropped) | 0.867 (0.780) | 0.017 |
| DreaMS embedding dot product | 0.811 (0.710) | 0.002 |
| DreaMS, mean-centred | 0.816 (0.715) | 0.002 |
| mean of cosine and DreaMS | 0.872 (0.795) | 0.002 |

- Falling back to the same adduct instead of the same polarity changes nothing.
- The hybrid's +0.005 is within noise at n = 200.
- Conclusion: **DreaMS as a plug-in similarity does not beat cosine.** No library search can find novel molecules (no reference spectrum to match).

## DreaMS embedding findings (`exploration_embeddings.ipynb`)
- 1,024-d unit vectors. 1,024 is the model width (`d_model`); the embedding is the transformed precursor token (`embs[:, 0]`), not a compression of the peaks. DreaMS keeps the 100 strongest peaks; test spectra have a median of 230.
- PCA: 20 components = 50% of variance, 211 = 90%. **PCA-256 is lossless for search:** same-molecule nearest neighbour 0.959 vs 0.960 (full), isomer top-1 0.889 vs 0.890. Fit the PCA once on a representative library sample, save it, re-normalize after projecting.
- The space is a cone (mean embedding length 0.44). Random pairs sit at angular distance ~0.44, not 0.5.
- Same molecule, same polarity: close across collision energies (median angular 0.23 vs 0.44 random). **Opposite polarity, or a different adduct: about as far as unrelated molecules.**
- Isomer test (same formula and polarity): same molecule vs isomer AUC 0.90 with the same adduct and energy, 0.83 across energies, 0.31 across adducts. The closest spectrum among all isomers is the molecule's own 88.6% of the time (random 7.6%), 87.7% with replicates excluded.
- UMAP keeps neighbourhoods but not global distances (Spearman ~0.55). Use it for pictures, and PCA for compression.

## Open questions and untested hypotheses
- **Why DreaMS loses to cosine (hypotheses):** the known scenario is near-replicate matching, where exact peaks are ideal; DreaMS keeps only 100 peaks; its training objective rewards the overall pattern, not exact peak positions. Cheap checks: run cosine through the isomer test; cap cosine to 100 peaks; build a "structural analogue" scenario (molecule removed, close analogues kept; needs RDKit).
## First leaderboard result (2026-10-02)
- The cosine baseline (`kaggle_submission/`) scored **0.098** on the public leaderboard. Submitted with `kaggle competitions submit enveda-CASMI26-molecule-id-mass-spectra -k cocolau/casmi-baseline -v 1 -f submission.csv -m "..."`.
- Reading: local known = 0.867, novel = 0.017, so 0.098 ≈ 10% known + 90% novel. **About 9 in 10 hidden-test molecules have no spectrum in train.** Rough estimate (local metric approximates the official one; the public leaderboard may be a subset of the test).
- Consequence: better library search can add at most ~0.02. The score is in the novel molecules.

## Suggested next steps
1. **Candidate recall:** for held-out molecules, how often is the true structure within 10 ppm in train ∪ COCONUT 2.0 (695k natural products)? That caps any ranking and tells whether COCONUT is the right pool.
2. **Score candidates without spectra:** DreaMS fine-tuned to predict a molecular fingerprint, candidates ranked by Tanimoto similarity (or simulated spectra for candidates). Give polarity and adduct to the model explicitly. Keep library search for the ~10% known molecules.
3. Copy the official metric (RDKit 2026.03.3) and recompute canonical keys for train.
4. Everything external (COCONUT, model weights, precomputed fingerprints/embeddings) must be uploaded as Kaggle datasets and listed in `dataset_sources` of `kernel-metadata.json`, since the notebook runs offline.

## Working with this user
- Uses **polars**, not pandas.
- Wants **simple, readable code**: explicit steps, no clever shortcuts; unfold loops over plot panels; OOP where it keeps things clean; "real Python scripts" for experiments, notebooks for exploration.
- Sometimes wants to write the code themself and asks for guidance: give steps and hints, not solutions, when they say so.
- Prefers true metrics (Euclidean, angular) over cosine distance, and standard matplotlib colormaps (`tab10`, `viridis`).
