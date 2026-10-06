# Enveda CASMI 2026: molecule identification from MS/MS spectra

Code for the Kaggle competition [Enveda CASMI 2026 – Molecule ID from Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra).
Given LC-MS/MS spectra, predict the 2D structure (SMILES) of the molecule. Up to 25 guesses per `molecule_id`, best first.

- **Metric:** MRR@25. A guess is correct if its InChIKey14 (after RDKit tautomer canonicalisation, RDKit 2026.03.3) matches the answer's. Stereochemistry and tautomers do not matter.
- **Code competition:** the notebook runs offline (≤ 9 h). External data and pretrained models must be uploaded as Kaggle datasets beforehand.

The data (`data/`, `external/`), trained models and generated libraries are not in this repository (see [Not in the repo](#not-in-the-repo)).

## Current approach: spectrum → fingerprint → candidate ranking

Generating a structure from a spectrum is hard, so the problem is split in two:

1. **Predict a molecular fingerprint from the spectrum.** An MLP reads one MS/MS spectrum and predicts the 2048-bit Morgan fingerprint of the molecule. Each bit asks "is this substructure present?", so the model is trained with binary cross-entropy over independent bits.
2. **Rank candidate structures against the predicted fingerprint.** Candidates are structures from a pool whose exact mass is within 10 ppm of the query's neutral mass. Each candidate is scored by how well its true fingerprint agrees with the predicted probabilities (soft Tanimoto similarity). The top 25, one per InChIKey14, are submitted.

```
spectra of one molecule_id
   │  bin peaks (0.1 Da, sqrt intensity) + metadata (adduct, energy, mass, polarity)
   ▼
MLP → bit probabilities (2048), averaged over the molecule's spectra of one adduct
   │
   ▼
candidates: pool structures within ±10 ppm of the neutral mass
   │  score = soft Tanimoto(candidate fingerprint, predicted probabilities)
   ▼
top 25 SMILES, one per InChIKey14
```

### Model input
- The spectrum's 100 strongest peaks, binned into 10,000 bins of 0.1 Da (max sqrt intensity per bin) ([build_spectrum_arrays.py](fingerprints_MLP/build_spectrum_arrays.py)).
- Measurement metadata ([metadata_features.py](fingerprints_MLP/metadata_features.py)): precursor m/z, neutral mass, polarity, collision energy (multi-hot), adduct (one-hot).
- Training spectra are filtered by [spectrum_quality.py](fingerprints_MLP/spectrum_quality.py): precursor mass error, minimum number of peaks, no fragment heavier than the precursor.

### Training variants
| Script | What changes |
|---|---|
| [train_MLP.py](fingerprints_MLP/train_MLP.py) | Baseline: BCE on fingerprint bits. Split by molecule, validation on `enveda-180` only |
| [train_MLP_negatives.py](fingerprints_MLP/train_MLP_negatives.py) | Adds a ranking loss: the truth must beat 32 random same-mass PubChem decoys (softmax cross-entropy on soft Tanimoto). This trains the model for the actual task: separating isomers |
| [train_MLP_resampled.py](fingerprints_MLP/train_MLP_resampled.py) | Same loss, but fresh decoys at every batch, so the model cannot memorise a fixed set |
| [train_MLP_dreams.py](fingerprints_MLP/train_MLP_dreams.py) | Uses DreaMS embeddings as input instead of binned peaks |

The best model so far is the one trained with the ranking loss (`mlp_negatives`).

### Candidate pools
- **train**: the structures of `train.parquet` (275,810 molecules).
- **train + COCONUT**: adds 480k natural products ([build_coconut_library.py](fingerprints_MLP/build_coconut_library.py)). This is the current pool for submissions.
- **train + PubChem** ([pubchem_pool/](fingerprints_MLP/pubchem_pool/)): 90M structures. Tested, but it did not help (see below).

## Results

Local evaluation uses held-out molecules the model never saw ([evaluate_ranking.py](fingerprints_MLP/evaluate_ranking.py)), because the visible `test.parquet` is a placeholder copied from train.

| Step | Result |
|---|---|
| Cosine library search (first submission) | local MRR 0.867 on molecules already in train, 0.017 on novel ones. **Public LB 0.098**: about 9 in 10 hidden-test molecules have no spectrum in train |
| MLP + ranking, validation molecules, train pool | MRR@25 0.46 (baseline MLP), 0.456 with the ranking loss |
| Baseline MLP on natural products (`enveda-np-examples`, train+COCONUT pool) | 0.138 vs 0.107 random: a model trained on `enveda-180` alone barely transfers to natural products |
| Ranking loss, validation molecules, (train − queries) + PubChem pool | 0.133 → 0.175 (+0.043 ± 0.006) |
| Public LB, MLP with ranking loss, train + COCONUT pool | **0.112** (best) |
| Public LB, same model, train + PubChem pool | 0.079 |

- **Library search cannot find novel molecules.** Better spectrum similarity adds at most ~0.02 to the score. Replacing cosine with DreaMS embeddings did not help either (0.811 vs 0.867 on known molecules).
- **`enveda-180` is not natural-product-like** (0.03% of its molecules are in COCONUT, vs 99.6% of `enveda-np-examples`), so it is a poor proxy for the hidden test alone. Training on the public libraries too, and evaluating on natural products separately, matters.
- **A bigger pool is not better.** PubChem raises recall but adds many decoys: on the placeholder output, 68% of top-1 guesses were PubChem-only structures, and the leaderboard score dropped. The hidden-test answers are mostly not PubChem-only.
- **SHAP analysis** ([shap_analysis.py](fingerprints_MLP/shap_analysis.py)): ~80% of the attribution comes from fragment peaks (mostly m/z 60–250), 16–18% from metadata. Neutral losses of 162 (hexose) and 308 (rutinoside) matter for natural products.

## Repository layout

```
fingerprints_MLP/
  build_fingerprint_library.py   Morgan fingerprint of every train molecule
  build_coconut_library.py       COCONUT candidates, same format
  build_spectrum_arrays.py       train spectra → fixed-size peak arrays
  build_negatives.py             32 fixed same-mass decoys per molecule
  build_negative_bank.py         decoy bank for the resampled variant
  train_MLP*.py                  the training variants above
  predict.py                     spectra → fingerprint probabilities
  rank_candidates.py             candidates in a mass window, scoring, top 25
  evaluate_ranking.py            MRR@25 in three scenarios (see its docstring)
  evaluate_fpnet.py              comparison with a public fingerprint model
  shap_analysis.py               which inputs drive the predictions
  pubchem_pool/                  PubChem candidate pool: build and evaluate
exploration/                     data exploration notebook
figures/                         saved figures
HANDOFF.md                       detailed project log: data traps, environments, earlier results
```

## Not in the repo
- `data/` and `external/` (competition data, COCONUT, PubChem tier): symlinks to an external disk.
- Trained models (`*.pt`), generated libraries (`*.npy`, `*.parquet`, `*.npz`) and result files: rebuild them with the `build_*.py` and `train_*.py` scripts, in that order.
- `dreaMS/` (DreaMS embedding experiments, separate Python 3.11 environment).
- `kaggle_submission/`: the Kaggle notebooks and datasets used for submissions.

## Environment
Python 3.14 with polars, numpy, torch and RDKit 2026.3.x (the metric pins 2026.03.3). DreaMS needs its own Python 3.11 environment.

## Notes
- The test set visible locally is a placeholder: every spectrum in it is also in train. Never evaluate on it; use held-out train molecules.
- Local MRR approximates the official metric; the official metric notebook is the reference.
