# mini-alphafold

A small-scale, educational re-implementation of AlphaFold-style protein structure
prediction components in PyTorch.

The goal is to make the major ideas behind modern protein folding systems easier
to study by building a functional, intentionally simplified pipeline from first
principles. This project is not intended to match AlphaFold2 accuracy; it is a
learning-oriented scaffold for data parsing, sequence models, contact maps, MSA
features, Evoformer-style attention, and coarse 3D structure prediction.

## Current Status

The project has working prototypes across Phases 0-3:

| Phase | Area | Status |
|---|---|---|
| Phase 0 | Data pipeline and tokenization | Implemented prototype |
| Phase 1 | Sequence and contact prediction | Implemented prototype |
| Phase 2 | Coarse 3D structure prediction | Implemented prototype |
| Phase 3 | MSA features and simplified Evoformer | Implemented prototype |

Most modules include smoke tests in `if __name__ == "__main__"` blocks, and the
training scripts can run against synthetic data without downloads. Formal unit
and integration test coverage is still sparse and should be a priority before
expanding the modeling surface further.

## Project Structure

```text
mini-alphafold/
├── src/
│   ├── data/        # PDB parsing, tokenization, dataset loading, MSA features
│   ├── models/      # Baselines, contact predictor, Evoformer, structure module
│   ├── training/    # Contact and structure losses, FAPE utilities
│   └── utils/       # Shared utilities
├── configs/         # Experiment configs
├── docs/            # Architecture notes, papers, workflow docs
├── notebooks/       # Exploration and analysis notebooks
├── scripts/         # CLI entry points for training workflows
├── tests/           # Test package placeholder
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
  available.

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
- Kabsch-aligned RMSD, TM-score, and GDT-TS evaluation helpers.

### MSA and Evoformer Prototype

- MSA feature tensors with one-hot residues, gap/deletion features, and profile
  frequencies.
- Simplified Evoformer stack with row/column attention over MSA features.
- Outer-product mean to update pair representations.
- Optional triangle multiplicative pair updates.
- End-to-end MSA-aware structure prediction script using pseudo-MSAs.

## Training Scripts

```bash
# Secondary structure baseline on synthetic data
python scripts/train_baseline.py

# Secondary structure baseline on real Hugging Face data
python scripts/train_baseline.py --real-data --ss-type ss3

# Contact-map prediction on synthetic data
python scripts/train_contact.py

# Contact-map prediction from PDB structures
python scripts/train_contact.py --pdb-ids 1UBQ,1CRN,4HHB

# Coarse 3D structure prediction
python scripts/train_structure.py

# MSA-aware end-to-end structure prediction
python scripts/train_end_to_end.py
```

## Quick Start

```bash
git clone https://github.com/mattackerman1/mini-alphafold.git
cd mini-alphafold
pip install -r requirements.txt
pytest tests/
```

## Development Priorities

- Add real pytest coverage for data parsing, collation, losses, and model shape
  contracts.
- Decide whether real MSAs should support variable numbers of homologs per
  sample, then update collation or dataset validation accordingly.
- Move checkpoints, downloaded PDB files, and generated caches out of the repo
  root or ensure they are ignored.
- Add experiment configs for the existing training scripts.
- Update docs as phases evolve so the README stays aligned with the code.

## Tech Stack

| Layer | Library |
|---|---|
| Deep learning | PyTorch |
| Biology / structure | Biopython, Biotite, ProDy |
| Datasets | Hugging Face Datasets |
| Transformers | Hugging Face Transformers |
| Visualization | Matplotlib, seaborn, py3Dmol |
| Tensor utilities | einops |
| Testing | pytest |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the multi-agent development workflow,
issue conventions, and coding standards.

## License

[MIT](LICENSE)
