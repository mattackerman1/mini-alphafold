# Key Papers

Summaries of the landmark publications in computational protein structure prediction that
directly inform mini-alphafold's architecture and training objectives.

---

## AlphaFold1 (2018 / CASP13)

| Field | Details |
|---|---|
| **Title** | Improved protein structure prediction using potentials from deep learning |
| **Authors** | Andrew W. Senior, Richard Evans, John Jumper, et al. (DeepMind) |
| **Published** | *Nature*, January 2020 |
| **DOI** | 10.1038/s41586-019-1923-7 |
| **Competition** | CASP13 (2018) — won the free-modelling category |

### What it contributed

- Used a deep residual network to predict **inter-residue distance distributions** and
  **backbone torsion (φ/ψ) angle distributions** from MSA-derived features.
- These distributions were used as **potential energy terms**, then minimised with gradient
  descent to produce final 3-D coordinates.
- Demonstrated that predicted distance distributions (not just binary contact maps) were
  far more informative for structure prediction.
- Achieved CASP13 top ranking, but still well below experimental accuracy.

### Key limitation

The two-stage pipeline (predict potentials → optimise separately) was not end-to-end
trainable. The structure optimisation step was slow and could get stuck in local minima.

---

## AlphaFold2 (2021 / CASP14)

| Field | Details |
|---|---|
| **Title** | Highly accurate protein structure prediction with AlphaFold |
| **Authors** | John Jumper, Richard Evans, Alexander Pritzel, et al. (DeepMind) |
| **Published** | *Nature*, August 2021 |
| **DOI** | 10.1038/s41586-021-03819-2 |
| **Competition** | CASP14 (2020) — dominated all categories |

### What it contributed

1. **Evoformer** — a novel transformer-based architecture that jointly processes the MSA
   and a pairwise residue representation through alternating row/column attention and
   triangle multiplicative updates.

2. **Pair representation** — an explicit learned embedding for every ordered pair of
   residues (i, j), encoding distance, orientation, and co-evolutionary signal. This
   is updated throughout the network and biases attention.

3. **Structure Module** — an equivariant network that operates on backbone rigid-body
   frames (SE(3) transformations). Predicts 3-D coordinates directly without fragment
   assembly.

4. **Invariant Point Attention (IPA)** — an attention mechanism that is invariant to
   global rotations and translations, allowing the model to reason about local geometry.

5. **FAPE loss** — Frame-Aligned Point Error: measures structural accuracy in a way
   that is sensitive to local and global geometry simultaneously.

6. **Recycling** — running the full network 3× and feeding predictions back as input,
   allowing iterative refinement without extra parameters.

7. **End-to-end training** — the entire pipeline from MSA features to 3-D coordinates
   is differentiable and trained jointly.

### Impact numbers

- GDT-TS ≥ 90 on most CASP14 targets (experimental noise threshold)
- ~10× better than the second-best system on hardest targets
- Outperformed methods that used experimental templates on template-free targets

---

## AlphaFold-Multimer (2021)

| Field | Details |
|---|---|
| **Title** | Protein complex prediction with AlphaFold-Multimer |
| **Authors** | Richard Evans, Michael O'Neill, Alexander Pritzel, et al. (DeepMind) |
| **Published** | *bioRxiv* preprint, 2021 (later peer-reviewed) |
| **DOI** | 10.1101/2021.10.04.463034 |

### What it contributed

Extended AlphaFold2 to predict **multi-chain protein complexes** (heteromers and homomers).
Key additions:
- Paired MSAs across chains to capture inter-chain co-evolutionary signal.
- Modified input features and chain-break encoding.
- New evaluation metric: interface predicted TM-score (ipTM).

---

## Related reading

These are not required for mini-alphafold but useful context:

| Paper | Relevance |
|---|---|
| **RoseTTAFold** (Baek et al., *Science* 2021) | Baker Lab's concurrent three-track architecture; similar ideas to AF2 arrived at independently |
| **ESMFold** (Lin et al., *Science* 2023) | Meta AI — predicts structure from sequence alone using a protein language model (no MSA) |
| **OpenFold** (Ahdritz et al., 2022) | Open-source PyTorch re-implementation of AF2 with training code — most directly relevant to this project |
| **Jumper et al. Supplementary** | The AF2 paper's supplementary methods section contains full architecture details, loss formulations, and hyperparameters |
