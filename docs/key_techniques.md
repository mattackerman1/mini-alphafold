# Key Techniques in AlphaFold2

A component-by-component explanation of the major technical innovations in AlphaFold2,
written for someone implementing a mini version. Each section covers:

- **Problem** — what limitation the technique addresses
- **How it works** — mechanism at a conceptual level
- **Why it matters** — why this design choice was important
- **mini-alphafold note** — what we implement and what we simplify

---

## 1. Multiple Sequence Alignment (MSA) as Input

### Problem
A single protein sequence contains limited information about which residues are
spatially close. Many different 3-D shapes could fold a given sequence.

### How it works
An MSA collects hundreds to thousands of *homologous* sequences (related proteins from
different species) that have diverged over evolutionary time. The key insight: if two
residue positions tend to mutate *together* across species, they are likely in physical
contact (co-evolution). AlphaFold2 represents the MSA as a matrix of shape
`(num_sequences × sequence_length)` and uses it as a primary input.

Features derived from the MSA:
- **Position-specific scoring matrix (PSSM)** — per-position amino acid frequencies
- **Covariance / co-evolutionary statistics** — correlations between positions
- **Deletion features** — which sequences have insertions/deletions at each position

### Why it matters
MSA-derived features provide evolutionary context that a single sequence cannot. Without
this, the model would have to infer 3-D constraints from chemistry alone.

### mini-alphafold note
Phase 1 uses single sequences only. MSA integration is deferred to Phase 3 (Evoformer).

---

## 2. Pair Representation

### Problem
Standard sequence models (LSTMs, Transformers) represent each residue as a single vector.
They have no explicit representation of *relationships between pairs* of residues — but
distance and orientation between pairs is exactly what defines 3-D structure.

### How it works
AlphaFold2 maintains two coupled representations throughout the network:

- **MSA representation** — shape `(num_seqs, seq_len, c_m)` — one row per homologue
- **Pair representation** — shape `(seq_len, seq_len, c_z)` — one entry per ordered pair (i, j)

The pair representation is initialised from:
- Outer product of the MSA row (first sequence) with itself
- Relative positional encodings (which position i is relative to j)

Both representations are updated in every Evoformer block and inform each other. Attention
in the MSA uses the pair representation as a **bias** on attention logits.

### Why it matters
By explicitly representing all O(L²) pairwise relationships, the model can directly learn
to predict contact maps, distance distributions, and orientation — all of which constrain
the 3-D fold.

### mini-alphafold note
The pair representation is the biggest thing we do not implement in Phase 1. It is
central to Phase 2+ (contact map prediction and structure module).

---

## 3. Evoformer

### Problem
Both the MSA representation and the pair representation need to be updated in a way that
is consistent with each other and that propagates information across the sequence. Standard
attention (applied to the full sequence) is O(L²) in memory and doesn't naturally handle
the 2-D pair matrix.

### How it works
The Evoformer is a stack of 48 identical blocks. Each block applies four operations in order:

#### 3a. Row-wise gated self-attention (MSA → MSA, biased by pairs)
Attention across the sequence dimension (columns) for each MSA row independently.
The pair representation `z[i, j]` is added as a bias to the attention logit between
positions i and j. This lets the pair matrix guide which residues attend to each other.

```
attn_logit[i, j] += linear(z[i, j])   # pair bias
```

#### 3b. Column-wise gated self-attention (MSA → MSA)
Attention across the sequence dimension (rows) for each column independently.
This allows the model to share information across homologues at the same position —
e.g., to recognise that position 42 is always hydrophobic across all species.

#### 3c. MSA transition (feed-forward)
Standard 2-layer MLP applied position-wise to the MSA representation.

#### 3d. Outer product mean (MSA → pair)
Summarises the MSA into a pair update:
```
pair_update[i, j] = mean_over_sequences(outer_product(msa[:, i], msa[:, j]))
```
This is the primary path by which co-evolutionary signal (from the MSA) flows into the
pair representation.

#### 3e. Triangle multiplicative update (pair → pair)
See Section 4 below.

#### 3f. Triangle self-attention (pair → pair)
Attention over rows or columns of the pair matrix, gated by the third edge of each triangle.

#### 3g. Pair transition (feed-forward)
Standard MLP applied to the pair representation.

### Why it matters
The Evoformer is what allows AF2 to jointly reason about sequences across evolutionary
time (via the MSA) and spatial relationships (via the pair matrix) in a single network.
The alternating row/column attention ensures O(L) memory per block rather than O(L²).

### mini-alphafold note
We implement a single-sequence Transformer encoder (Phase 1) as a simplified stand-in.
Full Evoformer implementation is the core of Phase 3.

---

## 4. Triangle Multiplicative Updates

### Problem
The pair representation has a geometric constraint: if residues i and j are close, and
j and k are close, then i and k are probably close too (triangle inequality in 3-D space).
Standard attention does not enforce this — it can learn inconsistent pair representations.

### How it works
Triangle multiplicative updates explicitly compute a new pair embedding for edge (i, k)
by combining the embeddings of the two other edges of the triangle (i, j) and (j, k),
summed over all intermediate nodes j:

**Outgoing triangles (update (i, k) using (i, j) and (j, k)):**
```
z_new[i, k] += sum_j( linear_a(z[i, j]) * linear_b(z[j, k]) )
```

**Incoming triangles (update (i, k) using (j, i) and (j, k)):**
```
z_new[i, k] += sum_j( linear_a(z[j, i]) * linear_b(z[j, k]) )
```

This is essentially a learned matrix multiplication in the pair space, restricted to
preserve triangle-consistent geometry.

### Why it matters
Without triangle updates, the pair representation can predict that A-B = 5 Å,
B-C = 5 Å, and A-C = 20 Å simultaneously — which is geometrically impossible.
Triangle updates gradually enforce transitivity, making the pair matrix more
consistent with an actual 3-D embedding.

### mini-alphafold note
Implemented in Phase 3 as part of the Evoformer block.

---

## 5. Structure Module

### Problem
Given a good pair representation and MSA representation, how do you go from learned
embeddings to actual 3-D Cartesian coordinates?

### How it works
The Structure Module operates on a set of **backbone frames** — one per residue —
represented as SE(3) rigid-body transformations (rotation + translation).
It updates these frames using **Invariant Point Attention (IPA)**.

#### 5a. Backbone frames
Each residue's backbone is parameterised as a rotation matrix R ∈ SO(3) and a
translation t ∈ ℝ³. At initialisation, all frames are set to identity (all residues
at the origin, no rotation).

#### 5b. Invariant Point Attention (IPA)
A novel attention mechanism that:
1. Computes attention weights using both sequence-space features and 3-D point positions.
2. Is invariant to global rotations and translations (equivariant to local frames).
3. Produces updated frame representations by attending over all other residues in 3-D space.

The key idea: each head computes query/key/value points in the *local frame* of each
residue, so the attention is sensitive to relative geometry without being affected by
the global orientation of the protein.

#### 5c. Torsion angle prediction
After IPA, a small MLP predicts backbone torsion angles (φ, ψ, ω) and side-chain
chi angles (χ₁–χ₄). These angles, combined with ideal bond lengths and angles from
the CHARMM/Engh-Huber parameterisation, are used to compute all atom positions via
forward kinematics.

### Why it matters
The Structure Module produces full 3-D coordinates in a single differentiable forward
pass, replacing the multi-stage (predict distances → optimise structure) pipeline of AF1.
IPA ensures the output is geometrically consistent without any explicit physics simulation.

### mini-alphafold note
Phase 2 implements a simplified version: predict Cα positions directly, without full
IPA or torsion angle prediction. Full Structure Module with IPA is Phase 2's stretch goal.

---

## 6. FAPE Loss (Frame-Aligned Point Error)

### Problem
Standard RMSD (root mean squared deviation) between predicted and true coordinates
is a poor training signal because:
- It requires alignment of the global frame (rotation + translation) first.
- It is dominated by flexible/disordered regions.
- It does not distinguish between local accuracy and global accuracy.

### How it works
FAPE measures error in the *local reference frame* of each residue:

1. For each residue i, transform all Cα positions into i's local coordinate system
   (i.e., translate and rotate so that residue i is at the origin with a canonical orientation).
2. Compute the squared distance between predicted and true positions of every residue j
   in that local frame.
3. Average over all frames i and all points j, with a clamping to avoid gradient explosion
   from very wrong predictions early in training.

```
FAPE = (1 / NL) * sum_i sum_j || T_i^{-1}(x_j) - T_i^{-1}(x̂_j) ||
```

where T_i is the predicted frame for residue i, and x̂_j is the true position.

### Why it matters
- **Local sensitivity**: errors in local structure (secondary structure, bond geometry)
  contribute even if the global fold is roughly right.
- **No alignment needed**: the loss is computed in local frames, so there is no global
  superposition step.
- **Scale-independent**: a loop being 2 Å wrong contributes equally regardless of
  protein size.

### mini-alphafold note
Phase 2 implements FAPE over Cα atoms only (backbone-only approximation).

---

## 7. Recycling

### Problem
A single forward pass through the Evoformer and Structure Module produces a prediction,
but there is no guarantee it is internally consistent. Iterative refinement is known to
help in structure prediction, but training a recurrent network over many steps is expensive.

### How it works
AlphaFold2 runs its full forward pass **3 times** (by default). After each pass:
- The pair representation from the Evoformer is added to the initial pair features.
- The predicted Cα positions are encoded as a distance matrix and added to the pair input.
- The predicted MSA representation is added back to the MSA input.

Only the **last pass** contributes to the training loss. The earlier passes are treated
as deterministic preprocessing steps (gradients stop at the recycle boundary during
most of training, then a fraction of batches use full gradient flow through all cycles).

### Why it matters
Recycling allows the model to iteratively correct its predictions without adding recurrent
parameters. The second pass sees a pair matrix that already encodes a rough distance
prediction, allowing it to focus on refinement rather than coarse structure.

### mini-alphafold note
Not implemented in Phases 1–2. Recycling is Phase 3+ territory.

---

## 8. Confidence Estimation (pLDDT and PAE)

### Problem
A structure prediction is not useful without knowing how confident the model is in each part.
Experimental structures include B-factors (per-atom uncertainty estimates). What is the
equivalent for a neural network prediction?

### How it works
AlphaFold2 predicts two confidence scores as auxiliary outputs:

**pLDDT (predicted Local Distance Difference Test):**
- Per-residue score from 0–100 estimating how accurate that residue's predicted position is.
- Trained to match the actual lDDT-Cα score the prediction would receive if evaluated
  against the true structure.
- Computed as the argmax of a 50-bin softmax head applied to the single representation.

**PAE (Predicted Aligned Error):**
- A (L × L) matrix. Entry (i, j) estimates the expected error in residue j's position
  when the structure is aligned on residue i's frame.
- Used to assess inter-domain accuracy and to identify independently folding regions.
- Released publicly via the AlphaFold database viewer.

### Why it matters
pLDDT lets users know which regions are reliably predicted (structured) vs. unreliable
(disordered). Regions with pLDDT < 50 are likely intrinsically disordered and should
not be modelled as having a fixed structure.

### mini-alphafold note
pLDDT prediction is a good stretch-goal auxiliary output for Phase 2 — it requires only
a small additional head and is highly practically useful.

---

## Summary Table

| Technique | Phase | Core idea | Key paper section |
|---|---|---|---|
| MSA features | Phase 3 | Co-evolution encodes contacts | Methods §1.2 |
| Pair representation | Phase 2+ | Explicit (i,j) embeddings | Methods §1.5 |
| Evoformer | Phase 3 | Row/col attention + triangle updates | Methods §1.6–1.7 |
| Triangle updates | Phase 3 | Enforce geometric consistency in pairs | Methods §1.6.5 |
| Structure Module / IPA | Phase 2 | SE(3)-invariant attention on frames | Methods §1.8 |
| FAPE loss | Phase 2 | Local-frame geometric loss | Methods §1.9.1 |
| Recycling | Phase 3 | Iterative self-refinement | Methods §1.10 |
| pLDDT / PAE | Phase 2+ | Per-residue and pairwise confidence | Methods §1.9.6 |
