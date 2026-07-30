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
path is unverified.

**DEFERRED to post-competition by decision 2026-07-29.** The full-chain rebuild is a multi-hour
job and is explicitly NOT a pre-deadline blocker: the accepted dataset2 member is frozen and valid
on four independent grounds — its accepted online component score, its use in the accepted main
pack, its member SHA256, and byte-level verification against the accepted archive. The rebuild is
a post-competition maintenance task; run it then and compare against the member hash.

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
references. **Nothing was deleted, and nothing will be before the competition deadline** (decision
2026-07-29). `outputs/deletion_candidates.json` is the review queue, not an approved batch; no
output-family deletion happens until after the deadline.

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

## A6 — docs_local round labels are historically inconsistent (LOW)

`docs_local/` was restructured into `agent_runs/<run id>/` with canonical `prompt.md` /
`response.md` / `index.json`. Every file was classified by **reading its content**, not its name:
all 31 `*_brief.md` files are instructions sent to an agent, i.e. prompts. Two records carry
`classification_confidence: medium` and say why — `round14_fable5_application_report.md` is named
"report" but is a prompt (it reports applied results *and* requests the next lever), and its
ordering against the round-14 brief is inferred from content rather than known.

The residual ambiguity is the **letters inside the historical labels**. They are not one scheme:
`18A` / `18B` / `18C` / `19A` / `20A` / `21A` denote sub-rounds, while `22C` / `22F` / `22O` merely
abbreviate Codex / Fable / Opus. The run identifier keeps the first kind and drops the second, and
each document's own heading is preserved verbatim as `document_label`. **The labels inside the
documents were not rewritten** — editing an instruction after the fact to match a later naming
scheme would falsify the record of what was actually sent.

**Recommended follow-up:** none. Use `docs_local/migration/legacy-name-map.csv` to resolve any old
path found in an older note.

## A7 — run directories are flat, and old manifests keep old paths (LOW)

Runs are named `round-<NN>[<sub>]-<agent>` and are **flat children** of their tree --
`scratchpad/round-22-opus/`, not `scratchpad/round-22/opus/`. Nesting would add a path component,
and scripts inside these runs resolve the repository root by counting components (`parents[3]`,
`parents[4]`). **This is the exact bug the convention exists to prevent:** both the R2 and the RGR
migrations hit an off-by-one repository-root depth when a script was copied one level deeper.

The 2026-07-29 rename therefore preserved depth, and root resolution was re-verified for five
representative scripts afterwards. Two directories could not be given a round at all — the
footprint channel and the ds1 offline reconciliation were developed outside the numbered advisory
rounds — so they are named topically rather than assigned a guessed number.

**Residual staleness:** manifests and reports *inside* historical runs still record the old
directory names and, in some cases, absolute `F:\` paths from the machine that produced them.
They were left untouched because they are records of what was done at the time; the two
legacy-name maps resolve them. Only the graduated copies under `src/` had their path handling
corrected.

## A8 — the final dataset2 decoder is not graduated into `src/` (RESOLVED 2026-07-30)

Added at A-board closure, 2026-07-30. **Resolved the same day** during the pre-B-board code-inspection
preparation, before B-board development began.

**What was done.** The decoder is graduated to `src/strategies/ds2/cross_time_exclusivity.py` with the
same `apply(scores, test, train)` call shape as the two dataset1 postprocessors, plus a
byte-preserving `swap_score_tokens` path that swaps the two affected score *tokens* in the base
member's own bytes rather than re-serialising floats. Entry point `src/build_ds2_member.py --verify`.
The registry patch drafted below was applied in full: `lifecycle_status` is now `SHIPPED_ACTIVE`,
`implementation_path` names the tracked module, counts moved `SHIPPED_ACTIVE` 14 to 15 and
`ONLINE_VALIDATED` 1 to 0.

**Result: `BYTE_EXACT_REPRODUCTION`.** The tracked decoder regenerates
`beb13345dc020f32283cea2d132efa072eb52c73d29806f86f60fbf24982f971` from the hash-pinned base matrix,
verified on the Windows host **and** in the official target environment (Ubuntu 22.04, Python 3.10).
All twelve physical census anchors and all four effect counts match. Coverage: 31 unit tests plus 2
integration tests. **No test was weakened**; the suite went from 55 tests to 97, all passing.

One defect was found and fixed during the port. The original scratchpad script asserted that the
top-1 change mask equals the action mask. That holds for the accepted deployment but not in general:
an acted row whose two swapped scores are *equal* changes no ranking, and a tie is legitimate input
that will occur at B-board scale. The graduated module asserts the two invariants that actually hold
— no unacted row moves, and every acted row with a positive margin does move — and counts the
zero-margin case instead of raising on it.

**What remains open is A1, not A8.** The decode is proven from the base matrix; the base matrix is
still not proven reproducible from raw data.

The original record follows for history.

---

The final accepted `dataset2.csv` is produced by
`xte_cross_time_exclusivity_decode` applied on top of the `crf_promote.py` output. That decoder is
worth **+0.0017435797158127** of the final score and its implementation exists **only** in the
git-ignored local provenance tree, at
`scratchpad/round-23-opus/xte/scripts/build_xte_member.py`.

Consequences, stated plainly:

- The accepted state is reproducible **by extraction** — the member bytes are hash-pinned in
  `configs/production.json` and inside the accepted archive, and `tests/` verifies them — but **not
  from tracked code**. This is a weaker guarantee than dataset1 enjoys, where
  `src/build_ds1_member.py --verify` regenerates the member byte for byte.
- `docs/README.md` previously claimed that `src/` carries every mechanism holding the accepted score.
  That claim was corrected rather than left to rot.
- The strategy is recorded in `docs/strategy_inventory.json` as `lifecycle_status: ONLINE_VALIDATED`
  with a separate `operational_lifecycle: SHIPPED_ACTIVE` field, and **not** as `SHIPPED_ACTIVE` in the
  test-guarded field, because `tests/integration/test_accepted_reproduction.py` requires every
  `SHIPPED_ACTIVE` record to name a tracked implementation that exists. Asserting `SHIPPED_ACTIVE`
  there would have meant either weakening that test or pointing it at a git-ignored path. Neither was
  acceptable, so the two-axis record was used instead.

**Why it was not fixed in the finalisation pass.** Graduating the decoder means porting roughly 380
lines of byte-level CSV token manipulation and proving it reproduces
`beb13345dc020f32283cea2d132efa072eb52c73d29806f86f60fbf24982f971` exactly. That is a code change with
a real chance of a subtle mismatch, and the finalisation pass was explicitly scoped to evidence
consolidation, documentation, hygiene and release — not to model or pipeline code. Doing it badly under
time pressure would have been worse than recording it honestly.

**Recommended follow-up.** Port the decoder to `src/strategies/ds2/cross_time_exclusivity.py` with the
same `apply(scores, test, train)` shape as the two dataset1 postprocessors, add a
`src/build_ds2_member.py --verify` entry point that asserts the member hash, extend
`tests/integration/test_accepted_reproduction.py` with a dataset2 chain-reproduction test that skips
cleanly when local assets are absent, then flip the inventory record to `SHIPPED_ACTIVE` with the new
`implementation_path`. A concrete registry patch is drafted at
`scratchpad/a-board-finalisation-opus/registry_patch_proposal.md`.
