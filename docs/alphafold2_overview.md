# AlphaFold2 — Overview

## What is protein structure prediction?

Every protein in a living cell is a chain of amino acids that folds into a precise 3-D shape.
That shape determines the protein's function: how it catalyses reactions, binds other molecules,
or transmits signals. For decades, determining a protein's structure required expensive, slow
experimental techniques (X-ray crystallography, cryo-EM, NMR). The dream of *computational*
structure prediction — inferring shape from sequence alone — has been pursued since the 1970s.

---

## What AlphaFold2 did

DeepMind's AlphaFold2 (2021) solved protein structure prediction at near-experimental accuracy
for the first time, as measured at CASP14 (Critical Assessment of Protein Structure Prediction).

Key numbers from CASP14:
- **Median GDT-TS score: 92.4** (out of 100) — previous best was ~75
- **Median TM-score: > 0.9** on free-modelling targets
- For roughly two-thirds of targets, the prediction was within the noise of experimental methods

This was widely described as a 50-year-old grand challenge being solved in a single competition cycle.

---

## Why it was a breakthrough

### 1. End-to-end differentiable geometry
Previous methods assembled structures from fragments or optimised energy functions heuristically.
AlphaFold2 predicted backbone frames (rigid-body transformations) and side-chain torsion angles
directly, in a fully differentiable neural network, trained with a geometric loss (FAPE).

### 2. Evolutionary information as a first-class input
Multiple Sequence Alignments (MSAs) encode millions of years of co-evolution. If two residues
are co-evolving (mutating together), they are likely in contact in the 3-D structure.
AlphaFold2 learned to extract this signal systematically through the Evoformer.

### 3. Pair representation
Rather than reasoning about each residue in isolation, the model maintains a learned
representation for *every pair of residues*, capturing distance, orientation, and contact
information throughout the network. This is the key representational innovation.

### 4. Iterative refinement (recycling)
The model runs its full forward pass multiple times, feeding its own predicted structure back
as input. Each pass refines the prediction — analogous to gradient descent at inference time.

---

## Impact

### Science
- The AlphaFold Protein Structure Database (EMBL-EBI, 2022) contains predicted structures for
  **over 200 million proteins** — essentially every protein known to science.
- Accelerated drug discovery: binding site identification, de novo drug design, understanding
  disease mechanisms (e.g. AlphaFold used in research on Parkinson's, malaria, antibiotic resistance).
- Enabled structural genomics at scale: entire proteomes of model organisms predicted and published.

### Machine learning
- Demonstrated that geometric deep learning + large-scale evolutionary data could match
  decades of wet-lab work.
- Introduced architectural patterns (pair bias in attention, equivariant structure modules)
  now studied broadly in geometric ML.
- Prompted follow-on work: RoseTTAFold (Baker Lab), ESMFold (Meta), OmegaFold, and others.

### Open science
- Model weights and code released open-source (Apache 2.0 license) by DeepMind in 2021.
- Protein structure database made freely available.

---

## Relevance to this project

mini-alphafold re-implements the core ideas of AlphaFold2 at reduced scale:

| AF2 Component | mini-alphafold Phase |
|---|---|
| Sequence tokenisation | Phase 0 (done) |
| Secondary structure prediction | Phase 1 (done) |
| Backbone coordinate prediction, FAPE loss | Phase 2 |
| Evoformer (MSA + pair attention) | Phase 3 |

The goal is not to reproduce AlphaFold2's scale or accuracy, but to understand *why* each
component was designed the way it was, by building it from scratch.
