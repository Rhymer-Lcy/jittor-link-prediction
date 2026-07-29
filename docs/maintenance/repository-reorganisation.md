# Repository consolidation, 2026-07-29

What changed, why, and what was deliberately left alone. Companion document:
[repository-reorganisation-ambiguities.md](repository-reorganisation-ambiguities.md).

## Starting state

| item | value |
|---|---|
| HEAD | `8a2c922f0687b105fd2ba020873a337b150169e0` |
| branch | `main` (clean working tree, in sync with `origin/main`) |
| tracked files | 17 — `README.md`, `docs/data-b-runbook.md`, `requirements.txt`, 13 × `src/*.py` |
| ignored trees | `data/` 261 MB, `outputs/` 25.8 GB, `scratchpad/` 21 GB, `docs_local/` 79 MB |

The gap that motivated the pass: **the two dataset1 mechanisms carrying the accepted score, and
the dataset2 geometry under it, existed only in git-ignored directories.** A collaborator syncing
the repository received none of them. The README had already flagged the MF case as "a clean-checkout
reproduction / B-board liability".

## Safety rules observed

Accepted artifacts were hash-verified before and after every step and never modified: the
accepted archive `6b9f0cd9...` (63,984,734 B) and its members `baa0dc21...` / `dc6928c2...`, plus
the four raw data files. No model was retrained, nothing was submitted, no package was installed,
no branch or worktree changed, nothing was pushed, and no uncertain artifact was deleted.

## What was graduated into tracked code

### dataset1 postprocessors — reproduce the accepted member exactly

| new file | source |
|---|---|
| `src/strategies/ds1/source_slate_recurrence.py` | frozen R2 rule, `scratchpad/round-21a-opus/r2/` |
| `src/strategies/ds1/test_graph_reciprocity.py` | frozen RGR rule, `scratchpad/round-22-opus/rgr/` |
| `src/strategies/shared/frozen_ops.py` | shared frozen primitives |
| `src/strategies/registry.py` | lifecycle vocabulary + the ordered active chain |
| `src/build_ds1_member.py` | reproduction CLI with `--verify` |

These are clean re-implementations of the two frozen rules, not copies of the experiment
scripts — and the check that they are faithful is byte equality, not review:

```
python src/build_ds1_member.py --verify
[1] source_slate_recurrence: 3653 actions  -> b04a45ab...  (the accepted Round-21 member)
[2] test_graph_reciprocity:  1365 actions  -> baa0dc21...  (the accepted Round-22 member)
VERIFY PASSED
```

Both intermediate and final hashes match the accepted artifacts exactly.

**Serialisation detail:** the accepted member is `%.6f` with CRLF, which is what pandas produced
on the Windows host that built it. `write_score_matrix` pins the line terminator explicitly, so
the accepted bytes now reproduce on any platform rather than only on Windows.

### dataset2 MF pipeline — verbatim, path-only changes

| new file | source | change |
|---|---|---|
| `src/ds2_mf_basket_pack.py` | `scratchpad/round-16-opus/build_mf_aux_pack.py` (`d9c6204b...`) | `REPO` from `__file__` instead of a hard-coded `F:\` path; dependency import renamed |
| `src/ds2_basket_featurizer.py` | `scratchpad/archive/ds2_closed_veins/bagging_ensemble_ds2.py` (`bc6f7fd8...`) | one stale docstring path |

Verified by diff: nothing else differs. **Not re-executed** — see ambiguity A1.

### Reusable tooling

`tools/submission/package_component.py`, graduated from
`scratchpad/round-22-opus/harness/scripts/package_component.py` (`77942376...`) where it had 35
self-tests and three dry runs. All five modes, every check and the manifest schema are preserved.
Two changes: the accepted-state constants are now read from `configs/production.json` instead of
being hard-coded, and the archive-size messages were corrected (see below).

## What was created

| path | purpose |
|---|---|
| `configs/production.json` | machine-readable definition of the accepted state |
| `docs/strategy_inventory.json` | 53 strategies with lifecycle status and evidence |
| `docs/STRATEGY_REGISTRY.md` | generated from the inventory |
| `docs/CURRENT_PRODUCTION.md` | what is active, how to reproduce, what not to touch |
| `docs/SUBMISSION_PROTOCOL.md` | measured platform behaviour |
| `docs/architecture/pipeline-overview.md` | both chains end to end |
| `docs/rounds/round-21.md`, `round-22.md` | concise canonical histories |
| `docs/naming-standard.md` | the conventions applied here |
| `tests/` | 55 tests (unit + integration; data-dependent ones skip cleanly) |
| `outputs/inventory.json`, `outputs/deletion_candidates.json` | outputs governance (ignored) |
| `docs_local/agent_runs_index.json`, `scratchpad/index.json` | local indexes (ignored) |

## The correction this pass produced

The Round-21 conclusion of an **archive-size cliff** between ~64.0 MB and ~65.36 MB is **refuted
by our own accepted artifacts**:

| archive | bytes | outcome |
|---|---|---|
| `submission_ds1ranker_footprint.zip` | 67,510,742 | accepted |
| `submission_mf_full_main.zip` | 65,437,854 | **accepted** (hash-verified `60341616...`) |
| `submission_r2_main.zip` | 65,363,753 | **rejected twice** |

The accepted pack is 74,101 bytes **larger** than the rejected one. The 2026-07-29 failure is
unexplained. Deflate-9 is retained as conservative practice; the thresholds are now labelled as
practice, not as a measured limit, everywhere they appear. A regression test
(`test_policy_knows_a_larger_archive_was_accepted`) fails if the refuted claim is reintroduced.

## Local trees renamed to one run-identifier scheme

A follow-up pass on the same day unified the two ignored trees on a single identifier,
`round-<NN>[<sub-round>]-<agent>` (see [naming-standard.md](../naming-standard.md)).

**`scratchpad/`** — 14 run directories renamed (`round21_opus` → `round-21a-opus`,
`negspace_audit` → `round-16-opus`, `phase2_build` → `round-17-opus`, …). Every run stayed a
**direct child of `scratchpad/`**, so the path depth is unchanged and every script that resolves
the repository root with `parents[3]` / `parents[4]` still resolves correctly — verified
explicitly after the move. Map: `scratchpad/migration/legacy-name-map.csv`.

**`docs_local/`** — `advisory/` was dissolved into `agent_runs/<run id>/` with canonical
`prompt.md` / `response.md` / `index.json`; 39 files moved. Content was re-read to classify each
file rather than trusting its name, and each document's own heading (for example `ROUND 22F`) is
preserved verbatim as `document_label` in the index. Map:
`docs_local/migration/legacy-name-map.csv`.

Two artifact classes left `docs_local/` for `scratchpad/`, where the taxonomy puts them: the two
formal reports and the four footprint metrics JSONs. Those JSONs turned out to be the **only
surviving copies** — the canonical `outputs/dataset2-footprint/footprint_ab_probe_metrics.json`
no longer exists — so they are evidence, not duplicates, and are now at
`scratchpad/footprint-ab/probes/`.

## What was deliberately not done

- **`outputs/` was not restructured** — hard-coded paths make a move a reproducibility risk (A2).
- **Nothing under `outputs/` was deleted** — no debris was found; 38 replica directories
  (6.36 GB) are listed for a human Stage-2 pass (A3).
- **Run directories were renamed but not re-nested** — nesting as `round-NN/<agent>/` would add a
  path component and break the depth-based repository-root resolution (A7).
- **Historical manifests inside runs still record their old absolute paths** — they are records of
  what was done at the time; the legacy-name maps resolve them (A7).
- **dataset2 was not rebuilt** to verify the graduated pipeline (A1).

## Verification performed

- `python -m unittest discover -s tests -t .` → **55 tests, OK**
- `python -m py_compile` on every new and migrated Python file → clean
- `python src/build_ds1_member.py --verify` → byte-exact reproduction of the accepted member
- `python tools/submission/package_component.py verify --zip <accepted>` → both members reported
  `IS the accepted member`
- accepted archive and member hashes re-checked after all work → unchanged
- `0.8963013747474597 + 0.6789511047001768 = 1.5752524794476365` (exact decimal; float64 reprs
  the same value as `...366`)
