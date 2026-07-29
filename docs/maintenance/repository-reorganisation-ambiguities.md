# Reorganisation ambiguities and unresolved risks

Recorded 2026-07-29 during the repository consolidation. Each item is something the pass
deliberately did **not** resolve, because a documented legacy path is preferable to a broken
historical record.

## A1 — dataset2 is tracked but not proven reproducible (HIGH)

`src/ds2_mf_basket_pack.py` and `src/ds2_basket_featurizer.py` were graduated **verbatim** from
the scratchpad (only repository-root resolution and the dependency import name changed; the diffs
are recorded in `docs/maintenance/repository-reorganisation.md`). They were **not re-executed**:
a full dataset2 rebuild is a multi-hour LightGBM job and additionally needs the cached train
features that `build_or_load_features` produces under `outputs/`.

So the accepted `dataset2.csv` (sha256 `dc6928c2...`) is currently reproducible only from the
artifact, not from a clean checkout.

**Risk:** if the dataset2 member ever has to be rebuilt — a B-board rerun, a data refresh — this
path is unverified. **Recommended follow-up:** one end-to-end dataset2 rebuild, comparing the
result against the accepted member hash, before any deadline that depends on it.

Note also that `ds2_basket_featurizer.py` is the verbatim body of a **closed** bagging pilot. Only
its featurizer helpers (`build_or_load_features`, `build_features`, `item_profiles`,
`fb_features`, `structural_labels`) are on the production path; the bagging CLI is dead code
retained for fidelity. Splitting them would have meant editing a shipped algorithm during a
migration, which the consolidation refused to do.

## A2 — outputs/ was inventoried but not restructured (MEDIUM)

The proposed `production/ submissions/ experiments/ cache/ archive/` layout was **not** applied.
`src/ensemble_predict.py`, `src/ranker_ds1.py`, `src/ranker_basket_ds2.py` and
`src/ds2_mf_basket_pack.py` read run directories by hard-coded path, and historical manifests and
scratchpad scripts reference the same names. Moving them for tidiness would break reproduction of
the exact runs that produced the accepted scores.

**Recommended follow-up:** if a move is ever wanted, do it as its own change — update every
hard-coded path, re-run `python src/build_ds1_member.py --verify`, and only then commit.

## A3 — 38 directories (6.36 GB) await a human deletion pass (MEDIUM)

`outputs/deletion_candidates.json` lists seed and parameter replicas that no tracked source
references. **Nothing was deleted.**

An earlier draft of the classifier flagged *every* `-s<digits>` directory as redundant, which
would have proposed deleting the shipped multi-seed ensemble members. It was corrected by
resolving the seed tuples from `ensemble_predict._SEEDS`, `ranker_ds1.BPR_SEEDS` and
`ranker_basket_ds2.SEEDS`. **Re-check those constants before removing anything named `-s*`.**

## A4 — the 2026-07-29 submission failure is unexplained (MEDIUM)

The recorded "archive-size cliff between ~64.0 MB and ~65.36 MB" is **refuted**: accepted
archives exist at 65,437,854 B (hash-verified) and 67,510,742 B, both larger than the
65,363,753 B pack that was rejected twice. Size cannot be the discriminator, and the real cause
of that rejection is unknown.

The conservative practice (deflate-9, smallest archive) is retained because it costs nothing, but
it is now labelled as practice rather than as a measured limit, in
`configs/production.json`, `docs/SUBMISSION_PROTOCOL.md` and the packaging tool's `size_policy`.

**Recommended follow-up:** if a submission ever fails again, vary one packing variable at a time
and record the result, rather than re-deriving a threshold from a single pair of observations.

## A5 — best-or-latest leaderboard semantics (MEDIUM)

It is still unknown whether the board keeps the **best** or the **latest** submission. If latest,
a single-member upload on the *main* account would collapse the standing to one component's
score. Until this is settled, single-member ZIPs go to the auxiliary account only. This has been
open since Round 21 and is cheap to resolve by asking the organisers.

## A6 — docs_local classification confidence (LOW)

`docs_local/advisory/` holds 39 files whose names follow `roundNN_<agent>_brief.md`. Content
inspection confirms these are **prompts** (instructions sent to an agent) rather than reports,
with three exceptions that are responses or reports. They were **indexed in place, not moved**:
`docs_local/` is intentionally local-only and ignored, the names are already consistent, and
several round reports elsewhere reference these exact paths. The index at
`docs_local/agent_runs_index.json` records the logical round, agent, artifact type and
confidence for each.

**Recommended follow-up:** none required. If the folder is ever restructured, use the index as
the migration map.

## A7 — scratchpad physical layout kept as-is (LOW)

The canonical future layout is `scratchpad/round-XX/<agent>/`. Existing runs use
`round21_codex`, `round22_opus`, and older topic names (`negspace_audit`, `phase2_build`). They
were **not** moved: scripts inside them resolve the repository root by counting path components
(`parents[3]`, `parents[4]`), several manifests record absolute paths, and two round reports
reference the current names. `scratchpad/index.json` assigns each run a canonical logical
identity and records the physical path, so the naming is queryable without a risky move.

**This is the exact bug the migration convention exists to prevent:** both the R2 and the RGR
migrations hit an off-by-one repository-root depth when a script was copied to a deeper path. New
runs should follow the canonical layout; old ones stay where the evidence was produced.
