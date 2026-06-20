# mini-alphafold

A small-scale, educational re-implementation of AlphaFold-style protein structure
prediction components in PyTorch.

The goal is to make the major ideas behind modern protein folding systems easier
to study by building a functional, intentionally simplified pipeline from first
principles. This project is not intended to match AlphaFold2 accuracy; it is a
learning-oriented scaffold for data parsing, sequence models, contact maps, MSA
features, Evoformer-style attention, iterative recycling, and coarse 3D structure
prediction.

## Current Status

The project has working implementations across Phases 0–3, with a full test suite
and experiment configs:

| Phase | Area | Status |
|---|---|---|
| Phase 0 | Data pipeline and tokenization | Complete |
| Phase 1 | Sequence and contact prediction | Complete |
| Phase 2 | Coarse 3D structure prediction | Complete |
| Phase 3 | MSA features, Evoformer, and recycling | Complete |

**Test coverage:** 110 tests (91 unit + 19 integration), all passing. Coverage
spans data parsing, collation, all loss functions, model shape contracts, training
loops, checkpoint lifecycle, and the full data-pipeline-to-model path.

## Project Structure

```text
mini-alphafold/
├── src/
│   ├── data/        # PDB parsing, tokenization, dataset loading, MSA features
│   ├── models/      # Baselines, contact predictor, Evoformer, structure module, recycling
│   ├── training/    # Contact and structure losses, FAPE utilities
│   └── utils/       # Shared utilities
├── configs/         # Experiment YAML configs for all training scripts
├── docs/            # Architecture notes, papers, workflow docs
├── notebooks/       # Exploration and analysis notebooks
├── scripts/         # CLI entry points for training workflows
├── tests/
│   ├── data/        # Unit tests: PDB parser, protein dataset, MSA encoder, loaders
│   ├── models/      # Unit tests: baseline, contact predictor, Evoformer, structure module
│   ├── training/    # Unit tests: all loss functions
│   └── integration/ # Integration tests: training loops, checkpoints, data pipeline
├── pytest.ini
├── requirements.txt
└── CONTRIBUTING.md
```

## Implemented Components

### Data Pipeline

- Tokenizes amino acid sequences into a 20-residue vocabulary plus `PAD` and
  `UNK` tokens.
- Parses PDB and mmCIF files with Biopython.
- Extracts amino acid sequences and backbone atom coordinates in `[N, CA, C, O]`
  order.
- Loads secondary-structure labels from the Hugging Face
  `proteinea/secondary_structure_prediction` dataset.
- Builds pseudo-MSAs by mutating query sequences when real homologs are not
  available. MSA depth (N_seq) is caller-controlled — the encoder does not fix it.

### Sequence Models

- Bidirectional LSTM baseline for per-residue secondary-structure prediction.
- Transformer encoder baseline for the same sequence-labeling task.
- Shared padding-aware data collation for variable-length protein sequences.

### Contact Prediction

- Symmetric residue-residue contact-map head.
- Binary cross-entropy and focal contact losses.
- Synthetic contact-map data for quick pipeline checks.
- Optional real PDB-derived contact maps using CA-CA distance thresholds.

### Structure Prediction

- Coarse C-alpha coordinate prediction from sequence encoders.
- Per-residue rotation-frame prediction for FAPE.
- Backbone frame construction from true `[N, CA, C]` atoms.
- Frame-Aligned Point Error (FAPE) loss and backbone RMSD evaluation.

### MSA, Evoformer, and Recycling

- MSA feature tensors: one-hot residues (22-dim), deletion features (1-dim), and
  profile frequencies (22-dim) = 45-dim per position.
- Simplified Evoformer stack with row/column MSA attention, outer-product mean
  pair updates, and optional triangle multiplicative updates.
- **Recycling module** (`src/models/recycling.py`): `RecyclingEmbedder` bins
  previous-pass Cα distances and LayerNorms the previous MSA row; `RecycledEvoformer`
  wraps the Evoformer + structure module and runs N recycles, with gradients
  flowing only through the final pass.
- End-to-end MSA-aware structure prediction script using pseudo-MSAs.
- Interactive 3D visualization of predicted vs. ground-truth Cα backbone via
  py3Dmol (`scripts/visualize_prediction.py`).

## Quick Start

```bash
git clone https://github.com/mattackerman1/mini-alphafold.git
cd mini-alphafold
pip install -r requirements.txt
pytest tests/
```

## Training Scripts

All scripts accept a `--config` flag pointing to a YAML file in `configs/`.

```bash
# Secondary structure (synthetic data)
python scripts/train_baseline.py --config configs/baseline.yaml

# Secondary structure (real Hugging Face data)
python scripts/train_baseline.py --config configs/baseline.yaml --real-data --ss-type ss3

# Contact-map prediction
python scripts/train_contact.py --config configs/contact.yaml

# Contact-map prediction from PDB structures
python scripts/train_contact.py --config configs/contact.yaml --pdb-ids 1UBQ,1CRN,4HHB

# Coarse 3D structure prediction
python scripts/train_structure.py --config configs/structure.yaml

# MSA-aware end-to-end structure prediction
python scripts/train_end_to_end.py --config configs/end_to_end.yaml

# Visualize predicted vs. ground-truth backbone
python scripts/visualize_prediction.py --checkpoint ckpt_phase3.pt --pdb-id 1UBQ
```

## Next Steps: Improving Model Accuracy

The current pipeline trains on synthetic data and produces near-random predictions
(~35% SS3 accuracy vs. 33% chance). The following steps, roughly in order of
impact, would move it toward meaningful accuracy:

### 1. Train on real labeled data (highest impact)
The existing `--real-data` flag in `train_baseline.py` connects to the Hugging Face
`proteinea/secondary_structure_prediction` dataset. Enabling it should push SS3
accuracy from ~35% toward 70–80% with no model changes — the architecture is sound,
the bottleneck is data.

### 2. Use real MSAs instead of pseudo-MSAs
Pseudo-MSAs (random mutations of the query) carry no evolutionary signal. Real
homologs from databases like UniRef90 or BFD are what give AlphaFold2 its power.
The `MSAEncoder` already supports real sequences — the gap is a data download and
preprocessing step.

### 3. Train on real PDB structures for structure prediction
`train_end_to_end.py` supports `--pdb-ids` for real structure data. A set of
~1,000 diverse, high-resolution PDB chains would give the Evoformer real
sequence-structure signal to learn from.

### 4. Increase model capacity
Current defaults are intentionally small for speed (2 Evoformer blocks, c_m=64,
c_z=32). Scaling to 4–8 blocks with c_m=128, c_z=64 and training longer would
improve representation quality once real data is in place.

### 5. Add more recycling passes
`RecycledEvoformer` defaults to 3 recycles. AlphaFold2 uses up to 48. Even
increasing to 5–10 on real data should improve structure convergence.

### 6. Add template features
AlphaFold2 conditions predictions on known homologous structures (templates).
Adding a template embedding — even a simplified version — would provide strong
structural priors, especially for proteins with known relatives in the PDB.

### 7. Side-chain prediction
The current model predicts Cα positions only. Extending the structure module to
predict χ-angles or all-atom coordinates is the next natural step toward full
structure utility.

## Tech Stack

| Layer | Library |
|---|---|
| Deep learning | PyTorch |
| Biology / structure | Biopython |
| Datasets | Hugging Face Datasets |
| Visualization | Matplotlib, seaborn, py3Dmol |
| Testing | pytest |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the multi-agent development workflow,
issue conventions, and coding standards.

## License

[MIT](LICENSE)
