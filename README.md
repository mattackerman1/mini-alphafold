# mini-alphafold

A small-scale, educational re-implementation of DeepMind's AlphaFold2 for protein structure prediction, built with PyTorch.

**Goal**: Understand modern deep learning for structural biology by constructing a functional (but intentionally simplified) protein folding pipeline from first principles.

## Current Status

- Phase 0 (Data Pipeline): In Progress
- Phase 1+: Not started

---

## Project Structure

```
mini-alphafold/
├── src/
│   ├── data/        # Data loading, parsing, tokenization, MSA handling
│   ├── models/      # Neural network architectures (Evoformer, structure module, etc.)
│   ├── training/    # Training loops, loss functions, optimizers
│   └── utils/       # Shared utilities (geometry, metrics, visualization)
├── configs/         # YAML experiment configs
├── notebooks/       # Exploration and analysis notebooks
├── scripts/         # CLI entry points (train, evaluate, predict)
├── tests/           # Unit and integration tests
├── requirements.txt
└── CONTRIBUTING.md
```

---

## Phases

### Phase 0 — Data Pipeline & Tokenization
- Parse PDB / mmCIF structure files with Biopython
- Tokenize amino acid sequences (20-letter alphabet + special tokens)
- Build dataset loaders for CATH / SCOPe subsets
- Extract ground-truth backbone coordinates (Cα, N, C, O)

### Phase 1 — Sequence Modeling
- Predict secondary structure (helix / sheet / coil) from sequence
- Predict residue-residue contact maps
- Baseline: single-sequence transformer encoder

### Phase 2 — Coarse 3D Structure Prediction
- Predict Cα backbone geometry end-to-end
- Frame-aligned point error (FAPE) loss
- Evaluate with TM-score and GDT-TS

### Phase 3 — MSA Integration & Attention Improvements
- Multiple Sequence Alignment (MSA) input encoding
- Row-wise and column-wise attention (simplified Evoformer)
- Triangle multiplicative updates

---

## Tech Stack

| Layer | Library |
|---|---|
| Deep learning | PyTorch ≥ 2.2 |
| Biology / structure | Biopython, ProDy |
| Datasets | Hugging Face Datasets |
| Transformers | Hugging Face Transformers |
| Visualization | Matplotlib, py3Dmol |
| Tensor ops | einops |

---

## Quick Start

```bash
git clone https://github.com/your-username/mini-alphafold.git
cd mini-alphafold
pip install -r requirements.txt
pytest tests/
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the multi-agent development workflow, issue conventions, and coding standards.

---

## License

[MIT](LICENSE)
