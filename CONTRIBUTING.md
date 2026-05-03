# Contributing to mini-alphafold

## Multi-Agent Development Workflow

This project uses a **multi-agent** development model where specialized AI agents collaborate
on distinct concerns, handing off through GitHub Issues and Pull Requests.

For the full reference — agent role definitions, handoff protocols, label schema, issue and
PR templates, and interaction rules — see
[docs/multi_agent_workflow.md](docs/multi_agent_workflow.md).

### Agent Roles (summary)

| Agent | Responsibilities |
|---|---|
| **Architect** | High-level design, module interfaces, phase planning, opens issues |
| **Coder** | Implementation, refactoring, bug fixes |
| **Researcher** | Literature review, algorithm selection, data sourcing, writes to `docs/` |
| **Reviewer** | Code review, correctness checks, PR approval |
| **Tester** | Unit tests, coverage, CI health |

### Workflow (summary)

```
Issue opened (Architect) → Agent implements on branch → PR → Reviewer approves → Merge
```

---

## Code Standards

- **Python 3.10+** — type-annotate all public functions.
- Formatting: `black` (line length 100).
- Linting: `ruff`.
- Tests live in `tests/`; mirror the `src/` directory structure.
- Run `pytest tests/` before opening a PR; all tests must pass.
- Every new module includes a smoke-test under `if __name__ == "__main__"`.

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(data): add CATH dataset loader
fix(models): resolve NaN in FAPE loss for short sequences
docs(readme): update Phase 1 description
test(training): add coverage for gradient clipping
```

## Branch Naming

```
phase0/data-pipeline
phase1/contact-map-model
infra/ci-setup
fix/fape-loss-nan
```

## Adding a New Module

1. Create the file under the appropriate `src/` subdirectory.
2. Add a corresponding test file under `tests/` (or open a `[Test]` follow-up issue).
3. Export public symbols from the subpackage `__init__.py` only when they form a stable API.
4. Update `configs/` with any new hyperparameters.

## Questions

Open a GitHub Discussion or tag the relevant agent in an issue comment.
