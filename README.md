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
- The spectrum's 100 strongest peaks, binned into 10,000 bins of 0.1 Da (max sqrt intensity per bin) ([build_spectrum_arrays.py](fingerprints_MLP/src/build_spectrum_arrays.py)).
- Measurement metadata ([metadata_features.py](fingerprints_MLP/src/metadata_features.py)): precursor m/z, neutral mass, polarity, collision energy (multi-hot), adduct (one-hot).
- Combined model only: the spectrum's DreaMS embedding (full, or its first k PCA coordinates) and a has-embedding flag ([dreams_inputs.py](fingerprints_MLP/src/dreams_inputs.py), shared by training and prediction). Spectra of rare adducts have no embedding: zeros and flag 0.
- Training spectra are filtered by [spectrum_quality.py](fingerprints_MLP/src/spectrum_quality.py): precursor mass error, minimum number of peaks, no fragment heavier than the precursor.

### Training variants
| Script | What changes |
|---|---|
| [train_MLP.py](fingerprints_MLP/src/train_MLP.py) | Baseline: BCE on fingerprint bits. Split by molecule, validation on `enveda-180` only |
| [train_MLP_negatives.py](fingerprints_MLP/src/train_MLP_negatives.py) | Adds a ranking loss: the truth must beat 32 random same-mass PubChem decoys (softmax cross-entropy on soft Tanimoto). This trains the model for the actual task: separating isomers |
| [train_MLP_resampled.py](fingerprints_MLP/src/train_MLP_resampled.py) | Same loss, but fresh decoys at every batch, so the model cannot memorise a fixed set |
| [train_MLP_dreams.py](fingerprints_MLP/src/train_MLP_dreams.py) | Uses DreaMS embeddings as input instead of binned peaks |
| [train_MLP_combined.py](fingerprints_MLP/src/train_MLP_combined.py) | `train_MLP_resampled.py` with binned peaks **and** the DreaMS embedding as input. Embedding dropout (0.15) during training, so the model also works without it. Optional PCA reduction: `python train_MLP_combined.py <seed> <k>` |

### Candidate pools
- **train**: the structures of `train.parquet` (275,810 molecules).
- **train + COCONUT**: adds 480k natural products ([build_coconut_library.py](library/build_coconut_library.py)). This is the current pool for submissions. Isotope-labelled COCONUT entries (e.g. `[13C]`, deuterated) are replaced by their unlabelled form, so they sit in the right mass window.
- **ChEBI + LIPID MAPS** ([build_bio_library.py](library/build_bio_library.py)): 158,790 standardised metabolites and lipids, 70,606 of them new to train + COCONUT. Built, not yet used in evaluation or submissions.
- **train + PubChem** ([pubchem_pool/](library/pubchem_pool/)): 90M structures. Tested, but it did not help (see below).

## Repository layout

```
library/                           candidate structures, shared by every approach
  build_fingerprint_library.py     Morgan fingerprint of every train molecule
  build_coconut_library.py         COCONUT candidates, same format
  build_bio_library.py             ChEBI + LIPID MAPS candidates, same format
  pubchem_pool/                    PubChem candidate pool: build_pool, pubchem_tier, metric_key
  data -> external disk            the generated files (fingerprints, molecules, negatives, spectrum arrays, PCA)
fingerprints_MLP/
  src/                             the model: everything training and prediction need
    morgan_generator.py            the fingerprint definition (also used by library/)
    build_spectrum_arrays.py       train spectra → fixed-size peak arrays
    metadata_features.py           measurement metadata → input features
    spectrum_quality.py            which train spectra to drop
    build_negatives.py             32 fixed same-mass decoys per molecule
    build_negative_bank.py         decoy bank for the resampled variant
    train_MLP*.py                  the training variants above
    dreams_inputs.py               DreaMS embedding → input columns of the combined model
    predict.py                     spectra → fingerprint probabilities (any variant, width read from the weights)
    rank_candidates.py             candidates in a mass window, scoring, top 25
  analysis/                        evaluation and inspection of trained models
    evaluate_ranking.py            MRR@25 in three scenarios (see its docstring)
    compare_models.py              mean ± sd over training seeds
    evaluate_pool.py               MRR@25 with the PubChem pool
    evaluate_fpnet.py              comparison with a public fingerprint model
    shap_analysis.py               which inputs drive the predictions
    demo_*.ipynb                   walkthroughs of the decoys and of the ranking
  models/                          trained weights (not in the repo)
dreaMS/                            DreaMS embeddings (own Python 3.11 environment)
  embed_train.py                   embeds every common-adduct train spectrum, in resumable parts
  embed_dreams.py                  embeds a list of train rows
  library_search.py                DreaMS vs cosine library search
  fit_dreams_pca.py                PCA of the embeddings for the combined model
  dreams_lite.py                   torch + numpy re-implementation of the embedding model, for Kaggle
  extract_dreams_weights.py        official checkpoint → plain weights for dreams_lite
  verify_dreams_lite.py            dreams_lite vs the official embeddings
  make_test_reference.py           official embeddings of test.parquet, to check dreams_lite on Kaggle
  *.ipynb                          embedding exploration, adduct pairs
exploration/                       data exploration notebook
figures/                           saved figures
HANDOFF.md                         detailed project log: data traps, environments, earlier results
```

Paths are relative to each script's own file, so scripts run from any folder: `python fingerprints_MLP/src/train_MLP.py`. Modules of one folder import each other directly (`from train_MLP import ...`), so the Kaggle dataset can keep them as a flat copy of `fingerprints_MLP/src/`; scripts that reach into another folder add it to `sys.path` first.

## Not in the repo
- `data/` and `external/` (competition data, COCONUT, PubChem tier): symlinks to an external disk.
- Trained models (`*.pt`), generated libraries (`*.npy`, `*.parquet`, `*.npz`, in `library/data/`, a symlink to the external disk) and result files: rebuild them with the `build_*.py` and `train_*.py` scripts, in that order.
- DreaMS weights, embeddings and caches (`dreaMS/dreams_cache/`, embeddings on the external disk).
- Kaggle kernels and dataset folders (`kaggle_submission/`).

## Environment
Python 3.14 with polars, numpy, torch and RDKit 2026.3.x (the metric pins 2026.03.3). The `dreams` package needs its own Python 3.11 environment; `dreams_lite.py` runs in the main one.

## Notes
- The test set visible locally is a placeholder: every spectrum in it is also in train. Never evaluate on it; use held-out train molecules.
- Local MRR approximates the official metric; the official metric notebook is the reference.
- `dreams_lite.py` matches the official embeddings exactly on the Mac, but not yet on Kaggle: ties in intensity at the 100-peak cut are ordered differently by `np.argsort` on x86.
