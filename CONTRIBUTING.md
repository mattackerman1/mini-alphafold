# Contributing to mini-alphafold

## Multi-Agent Development Workflow

This project uses a **multi-agent** development model where specialized AI agents collaborate on distinct concerns. Each agent operates on a focused domain and hands off through GitHub Issues and Pull Requests.

### Agent Roles

| Agent | Responsibilities |
|---|---|
| **Architect** | High-level design, module interfaces, phase planning |
| **Coder** | Implementation, refactoring, bug fixes |
| **Researcher** | Literature review, algorithm selection, benchmarking |
| **Reviewer** | Code review, correctness checks, security |
| **Tester** | Writing and running tests, coverage analysis |

### Workflow

```
Issue opened (Architect or human)
   │
   ▼
Agent picks up issue → creates feature branch
   │
   ▼
Implementation on branch
   │
   ▼
Pull Request opened → Reviewer agent reviews
   │
   ▼
Merge to main after approval
```

### Issue Conventions

- Prefix issue titles with the phase: `[Phase 0]`, `[Phase 1]`, etc., or `[Infra]` for infrastructure.
- Assign a `agent:coder`, `agent:researcher`, etc. label to indicate the intended agent.
- Keep issues atomic — one task, one issue.

### Branch Naming

```
phase0/data-pipeline
phase1/contact-map-model
infra/ci-setup
fix/fape-loss-nan
```

### Code Standards

- **Python 3.10+**; type-annotate all public functions.
- Formatting: `black` (line length 100).
- Linting: `ruff`.
- Tests live in `tests/`; mirror the `src/` directory structure.
- Run `pytest tests/` before opening a PR; all tests must pass.

### Commit Messages

Use the conventional commits format:

```
feat(data): add CATH dataset loader
fix(models): resolve NaN in FAPE loss for short sequences
docs(readme): update Phase 1 description
test(training): add coverage for gradient clipping
```

### Adding a New Module

1. Create the file under the appropriate `src/` subdirectory.
2. Add a corresponding test file under `tests/`.
3. Export public symbols from the subpackage `__init__.py` only when they form a stable API.
4. Update `configs/` with any new hyperparameters.

### Questions

Open a GitHub Discussion or tag the relevant agent in an issue comment.
