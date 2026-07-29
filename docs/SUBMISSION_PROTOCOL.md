# Submission protocol

Everything here is **measured on the live platform**, not assumed. Where a belief was later
refuted, both the belief and the refutation are recorded — the refuted claim is more useful than
a silent correction.

## The scoring model

**The total is strictly additive: `total = ds1_MRR + ds2_MRR`.** Verified to 16 digits on two
independent occasions:

```
0.8619654323294592 + 0.6789511047001768 = 1.540916537029636   (accepted 2026-07-28)
0.8963013747474597 + 0.6789511047001768 = 1.5752524794476365  (accepted 2026-07-29)
```

**A dataset absent from the ZIP scores 0. It is NOT carried over from a previous upload.**
A single-member dataset1 ZIP scored 0.8883, not `0.8883 + stored ds2`. Consequences:

- a single-member upload can never reach a combined total;
- **a main-account pack must contain both `dataset1.csv` and `dataset2.csv`**;
- an all-zero placeholder member contributes exactly `0.0517217279290374` (the 100-way tie
  baseline, ~H(100)/100), which is why two-member probes read ~0.05 higher than an isolated
  component.

## Auxiliary experiments — the default

**A ZIP containing exactly one dataset CSV is accepted and scored directly, and its score IS
that component's score.** So:

- ship a **single-CSV archive** for every isolated experiment;
- do **not** build zero-filled placeholders unless something specifically requires them;
- do **not** resubmit a control while the frozen component base is unchanged.

Frozen component baselines: **dataset1 0.8963013747474597**, **dataset2 0.6789511047001768**.
Predicted combined totals: `new_ds1 + 0.6789511047001768` or `0.8963013747474597 + new_ds2`.

Only after an isolated component passes online may its exact accepted CSV be merged with the
frozen accepted CSV of the other dataset.

For every component experiment record: CSV member SHA256, ZIP SHA256, exact shape, changed-row
count, top-1 change count, baseline component score, treatment component score, exact delta, and
the predicted combined total. `tools/submission/package_component.py` emits all of it.

## Archive construction

| property | value |
|---|---|
| members | exactly `dataset1.csv`, `dataset2.csv` (in that order) — or one of them alone |
| compression | standard deflate, **method 8, level 9** |
| directory entries | none |
| duplicate members | none |
| archive comment | empty |
| CSV bytes | **opaque** — never re-serialise, never normalise line endings |

**Line endings differ between the two accepted members**: `dataset1.csv` is CRLF and
`dataset2.csv` is LF, an artifact of the two different writers. Any tool that "helpfully"
normalises them changes both hashes. Members are always copied as bytes.

Do **not** use BZIP2 or LZMA. They compress much better (53.3 MB and 53.6 MB for the same
members) but the platform has never been verified to accept them, and Java unzippers commonly
support only stored + deflate.

## Archive size — a hypothesis and its refutation

On 2026-07-29 a combined archive of **65,363,753 bytes (deflate level 6) was rejected twice**
with a generic, empty `Submission failed:` message. The identical CSV members repacked at
deflate level 9 (**64,004,492 bytes**) were accepted and scored normally. This was recorded at
the time as an *archive-size cliff between ~64.0 MB and ~65.36 MB*.

**That conclusion is refuted by our own record.** Larger archives had already been accepted:

| archive | bytes | outcome |
|---|---|---|
| `submission_ds1ranker_footprint.zip` | 67,510,742 | accepted (total 1.5284127, 2026-07-27) |
| `submission_footprint_FULL.zip` | 66,221,215 | accepted (total 1.5264, 2026-07-26) |
| `submission_mf_full_main.zip` | **65,437,854** | **accepted** (total 1.540916537029636, hash-verified) |
| `submission_r2_main.zip` | 65,363,753 | **rejected twice**, empty error |
| `submission_r2_main_d9.zip` | 64,004,492 | accepted (total 1.567242782636773) |
| `r22_rgr_main_d9.zip` | 63,984,734 | accepted (total 1.5752524794476366) |

The accepted `submission_mf_full_main.zip` is **74,101 bytes larger** than the pack that was
rejected. Size cannot be the discriminator.

**The 2026-07-29 rejection is therefore unexplained.** Two single-member ZIPs uploaded between
the failures and the success both succeeded, so the account and the session were healthy; a
transient platform-side failure fits the evidence at least as well as anything about the file.

What follows from this, practically:

- keep packing at **deflate level 9** — a smaller archive is strictly safer and costs nothing;
- the size bands in `configs/production.json` (`OK` <= 63.9 MB, `ACCEPTABLE-NO-MARGIN` above it,
  `REVIEW-REQUIRED` >= 65 MB) are **conservative practice, not a measured limit**;
- **no size is a guarantee** in either direction, and a small archive is not proof of anything;
- **a generic "Submission failed" is not evidence about the prediction content.** When every
  content check passes, retry and vary one packing variable at a time before theorising about
  accounts, quotas or deadlines. Two such theories were floated in Round 21 (a closed submission
  window, then a main-account quota) and both were killed by uploads succeeding at that moment.

## Failure interpretation

| symptom | what it is NOT | what to do |
|---|---|---|
| empty `Submission failed:` | not necessarily invalid predictions | verify members, repack at level 9, retry, then vary one variable |
| score exactly ~0.0517 for a dataset | not a scoring bug | that member was all-zero or degenerate |
| combined total below a component | not a platform error | a member was missing from the ZIP; it scored 0 |

## Accounts

The **primary account** is the sole competition account and receives every complete two-dataset
submission. The **auxiliary account** is for isolated single-dataset probes; its scores are < 1.0
by construction. Sending a complete two-dataset pack to the auxiliary account is a
competition-compliance red line.

**Unresolved:** whether the board keeps the **best** or the **latest** submission. If it keeps
the latest, a single-member upload on the *main* account would collapse the standing to that one
component. Until this is settled, single-member uploads go to the auxiliary account only.

## The tool

```bash
PY=python
H=tools/submission/package_component.py

# 1. validate + package an isolated component
$PY $H ds1 --csv <treatment_dataset1.csv> --experiment <name> --inversions

# 2. upload the ZIP on the AUXILIARY account; its score IS the component score

# 3. close the arithmetic
$PY $H score --manifest outputs/submissions/round22_components/<name>_ds1_manifest.json \
             --observed <full-precision score>

# 4. only if the delta is positive, build the main candidate
$PY $H main --ds1 <treatment_dataset1.csv> --ds2 accepted --experiment <name>
```

`--ds2 accepted` pulls the frozen member straight out of the accepted archive, so a one-sided
main pack never needs a hand-managed copy. Swap `ds1`/`ds2` for a dataset2 experiment.
