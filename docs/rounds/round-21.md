# Round 21 — dataset1 source-slate recurrence (R2)

**Outcome: shipped. Total 1.540916537029636 → 1.567242782636773, rank #3 → #2.** 2026-07-28/29.

## What shipped

`source_slate_recurrence` — a deterministic dataset1 postprocessor over the existing ranker
output. No model retrained, no feature regenerated, no threshold swept.

On each physical test row whose control top-1 is **non-historical** for that source, count every
candidate's occurrences across all observable candidate slates of the same source (column
position is never used), take the **unique** maximum-recurrence non-history candidate, require it
in **≥ 2 other slates**, and promote it to strict top-1. Acts on **3,653 / 61,051 rows (5.98%)**.

| evidence | value |
|---|---|
| offline ds1 replay | +0.021366 (non-repeat +0.042897, rotation null −0.036393) |
| isolated auxiliary A/B | control 0.9136871602584966 → treatment 0.9400134058656336 |
| online delta | **+0.0263262456071370** against a locked +0.0005 gate |
| accepted components | ds1 0.8882916779365962 + ds2 0.6789511047001768 |

**It was adjudicated online before shipping.** That is what made the direction safe either way,
and it produced the first offline→online agreement in the recent record.

Origin: Codex round 21. Executed in `scratchpad/round21_opus/r2/`. Now tracked at
`src/strategies/ds1/source_slate_recurrence.py`.

## What closed

The Round-20 family was adjudicated and closed permanently in this window:

- **E1 / E1b / E1c within-event collision exclusion** — the frozen T1 rule was built as a
  byte-exact auxiliary pair and **inverted online: −0.0146884639811103** from a clean +0.004041
  replay. The pair-q4/q5 pseudo-label gate had called the sign (−0.031667); the replay had not.
- **E2 chronological-prefix stand-in training** — stage-0 fail.
- **E3 test-native pseudo-label residual ranker** — stage-J fail, ~400× short of its gate.
- **Round 21F (Fable)** — the ds1 graph-motif and non-repeat-specialist axes closed; **XLIST**
  retained only as a low-upside standby, not a live direction.

## The two lessons

1. **Trust the serve-side pseudo-label gate over the replay.** The P1-family serve gate is 4/4 on
   online signs; the ds2 OOF replay is 0/2 on the last two shipping decisions. When they
   disagree, reject.
2. **The round-19A replay numbers are no longer reproducible on this machine** (post-CRF
   0.6459965253 vs a recorded 0.6458207392) with the MF geometry and all three LightGBM histogram
   paths proven bit-identical. Pair every arm on one regenerated base; never compare absolute
   MRRs across sessions.

## Submission incident

The first main pack (`submission_r2_main.zip`, 65,363,753 B, deflate level 6) was **rejected
twice** with an empty `Submission failed:` message. The identical members repacked at deflate
level 9 (64,004,492 B) were accepted. This was recorded at the time as an archive-size cliff.

**That conclusion has since been refuted** — see [SUBMISSION_PROTOCOL.md](../SUBMISSION_PROTOCOL.md).
An accepted archive of 65,437,854 B already existed. The failure remains unexplained.

Full working record: `scratchpad/round21_opus/r2/{report.md,manifest.json}`.
