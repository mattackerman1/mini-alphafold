# docs/

Reference documentation for the mini-alphafold project.

These notes are written for someone actively building a simplified re-implementation of AlphaFold2.
The goal is not to reproduce every detail of the original paper, but to give you enough understanding
of each component to make good engineering decisions during implementation.

## Files

| File | Contents |
|---|---|
| [alphafold2_overview.md](alphafold2_overview.md) | What AlphaFold2 is, why it was a breakthrough, and its scientific and practical impact |
| [key_papers.md](key_papers.md) | Structured summaries of the landmark papers (AF1 and AF2), with authors, venues, and key contributions |
| [key_techniques.md](key_techniques.md) | Deep-dive explanations of every major technical innovation: MSA processing, Evoformer, pair representations, Structure Module, recycling, FAPE loss, and more |
| [multi_agent_workflow.md](multi_agent_workflow.md) | Full agent role definitions, handoff protocols, GitHub label schema, issue and PR templates, and interaction rules |

## How to use these notes

- Start with **alphafold2_overview.md** for the big picture.
- Read **key_papers.md** to understand the scientific lineage and what each paper contributed.
- Use **key_techniques.md** as a reference while implementing each component — each section maps to a module in `src/models/`.

## Implementation mapping

| Technique | Target module |
|---|---|
| Tokenization & MSA input | `src/data/` |
| Evoformer (row/column attention, triangle updates) | `src/models/evoformer.py` *(Phase 3)* |
| Structure Module (IPA, backbone frames) | `src/models/structure_module.py` *(Phase 2)* |
| FAPE loss | `src/training/losses.py` *(Phase 2)* |
| Recycling | `src/models/recycling.py` *(Phase 3)* |
| Baseline sequence encoder | `src/models/baseline.py` *(Phase 1 — done)* |
