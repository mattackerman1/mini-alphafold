# Multi-Agent Development Workflow

This document defines the roles, responsibilities, and interaction protocols for the
multi-agent development model used in mini-alphafold. It is the authoritative reference
for how agents collaborate via GitHub Issues and Pull Requests.

---

## Agent Roster

### Architect
| Field | Detail |
|---|---|
| **Primary concern** | System design, module boundaries, phase planning |
| **Consumes** | Project goals, Researcher outputs, feedback from Coder/Tester |
| **Produces** | Architecture decision records, interface specs, phase roadmaps, Issue briefs |
| **Owns** | `docs/`, top-level `README.md`, module `__init__.py` public APIs |
| **Does NOT** | Write implementation code, run experiments |

The Architect opens issues for other agents and defines acceptance criteria. When a
phase boundary is reached, the Architect reviews whether the system design needs
updating before the next phase begins.

---

### Coder
| Field | Detail |
|---|---|
| **Primary concern** | Implementation, refactoring, bug fixes |
| **Consumes** | Issues from Architect, specs from Researcher, bug reports from Tester |
| **Produces** | Feature branches, Pull Requests, passing smoke-tests |
| **Owns** | `src/`, `scripts/`, `configs/` |
| **Does NOT** | Decide architecture, write formal test suites (that is Tester's role) |

The Coder includes a smoke-test (`if __name__ == "__main__"`) in every new module
so the code can be verified immediately. Formal unit tests are filed as a follow-up
issue for the Tester.

---

### Researcher
| Field | Detail |
|---|---|
| **Primary concern** | Literature, algorithm selection, benchmarking, data sourcing |
| **Consumes** | Phase goals from Architect, questions tagged `agent:researcher` |
| **Produces** | Notes in `docs/`, dataset recommendations, hyperparameter recommendations, benchmark baselines |
| **Owns** | `docs/key_papers.md`, `docs/key_techniques.md`, experiment analysis |
| **Does NOT** | Write production code, open implementation issues |

The Researcher's outputs are written to `docs/` and referenced in implementation
issues so the Coder has the necessary context.

---

### Reviewer
| Field | Detail |
|---|---|
| **Primary concern** | Code correctness, consistency, security, style |
| **Consumes** | Pull Requests |
| **Produces** | PR review comments, approval or change requests |
| **Owns** | PR review process |
| **Does NOT** | Implement fixes (files comments; Coder acts on them) |

The Reviewer checks that:
- The implementation matches the issue's acceptance criteria
- Type annotations are present on all public functions
- No obvious numerical bugs (e.g. loss applied to PAD tokens, shape mismatches)
- No security issues (e.g. shell injection in scripts, untrusted data executed)
- Code style matches project standards (`black`, `ruff`)

---

### Tester
| Field | Detail |
|---|---|
| **Primary concern** | Test coverage, regression prevention, CI health |
| **Consumes** | Merged PRs, bug reports |
| **Produces** | Test files in `tests/`, coverage reports, CI configuration |
| **Owns** | `tests/` directory structure, CI pipeline |
| **Does NOT** | Write production features |

Every module in `src/` should have a corresponding test file in `tests/` mirroring
the directory structure (e.g. `src/data/loaders.py` → `tests/data/test_loaders.py`).

---

## Issue Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                        OPEN ISSUE                           │
│  Title: [Phase N] Short description                         │
│  Label: agent:<role>                                        │
│  Body:  requirements + acceptance criteria                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    Assigned agent
                    picks up issue
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     IN PROGRESS                             │
│  Branch: phase<N>/short-description                        │
│  Agent works; may comment on issue if blocked               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    Work complete
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    PULL REQUEST                             │
│  Title matches issue title                                  │
│  Body: what changed, why, reviewer notes, test plan         │
│  Links issue: "Resolves #N"                                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    Reviewer approves
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      MERGED                                 │
│  Issue closed automatically via "Resolves #N"              │
│  Follow-up issues opened if needed (e.g. tests, docs)       │
└─────────────────────────────────────────────────────────────┘
```

---

## Handoff Protocols

### Architect → Coder
The Architect opens an issue with:
- A clear **task statement** (one sentence)
- **Acceptance criteria** (bulleted, verifiable)
- Links to relevant `docs/` files or prior issues for context
- The label `agent:coder`

The Coder does not start work until an issue exists with acceptance criteria.

### Researcher → Coder
When research produces an implementation recommendation, the Researcher either:
- Comments on the relevant open issue with findings, or
- Opens a new issue tagged `agent:coder` referencing the research doc in `docs/`

### Coder → Reviewer
The Coder opens a PR and:
- Fills out the full PR template (summary, what changed, reviewer notes, test plan)
- Self-reviews the diff before requesting review
- Marks the PR draft if work is incomplete

### Reviewer → Coder
The Reviewer leaves inline comments on specific lines. For blocking issues, the
Reviewer requests changes. For non-blocking suggestions, the Reviewer uses the
`nit:` prefix in comments. The Coder addresses all blocking comments before re-requesting review.

### Coder → Tester
After a PR is merged, if formal unit tests were deferred, the Coder opens a follow-up
issue: `[Test] Module name — unit tests` tagged `agent:tester`.

---

## GitHub Label Schema

| Label | Meaning |
|---|---|
| `agent:architect` | Issue is for the Architect agent |
| `agent:coder` | Issue is for the Coder agent |
| `agent:researcher` | Issue is for the Researcher agent |
| `agent:reviewer` | Issue is for the Reviewer agent |
| `agent:tester` | Issue is for the Tester agent |
| `phase:0` – `phase:3` | Which project phase this belongs to |
| `infra` | Infrastructure, CI, tooling — not tied to a phase |
| `blocked` | Issue cannot proceed; blocking dependency noted in comments |
| `good first issue` | Well-scoped, low-risk, good for onboarding |

---

## Issue Template

```markdown
## Task
One-sentence description of what needs to be done.

## Context
Why this is needed. Link to relevant docs/, prior issues, or research.

## Acceptance Criteria
- [ ] Specific, verifiable requirement 1
- [ ] Specific, verifiable requirement 2
- [ ] Smoke-test or test command that must pass

## Out of Scope
List anything explicitly NOT required in this issue.
```

---

## PR Template

```markdown
## Summary
One or two sentences. Which issue does this resolve?

## What Changed
Bullet list of files modified and what each change does.

## Reviewer Notes
Anything non-obvious the reviewer should know: design decisions,
tradeoffs, known limitations, or deferred work.

## Test Plan
- [ ] Command or test that was run and passed
- [ ] Known gaps (follow-up issue #N filed)
```

---

## Interaction Rules

1. **One issue, one agent.** If a task spans multiple agents, split it into multiple issues.
2. **No direct pushes to `main`.** All changes go through a PR, regardless of size.
3. **Issues before code.** The Coder does not implement something that has no issue.
4. **Atomic PRs.** A PR should be reviewable in one sitting. If it is growing large, split the issue.
5. **Follow-up issues over scope creep.** If you discover additional work while implementing,
   open a new issue rather than expanding the current PR.
6. **Comments over silence.** If an agent is blocked, it comments on the issue immediately
   rather than waiting. Other agents can unblock or re-scope.
