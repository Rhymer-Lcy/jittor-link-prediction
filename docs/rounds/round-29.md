# Round 29 — full-slate softmax role representation (FSSR): the online failure

**Outcome: ONLINE FAILURE. Submitted total 1.5494486598552333 against an active
1.5752524794476366.** 2026-07-30. Lifecycle `ONLINE_FAILED` / `CLOSED_ONLINE_FAILURE`.

This is the most informative negative result the project produced. It is recorded in full because a
failure of this shape is the only thing that falsifies the project's offline-to-online model.

## What was tested

`full_slate_softmax_role_representation` (`R29F-FSSR`) — the separately pre-registered successor to
the Round-28 `H0` control (see [round-28.md](round-28.md)). A frozen unit-norm cut-LINE base
(43,215 x 400) plus trainable source and destination delta tables,
`score = 8.0 * dot(E0[u] + Ds[u], E0[v] + Dd[v])`, retrained with **full-slate softmax only — no
hard negatives, no propagation** — and exposed as one scalar feature over the control ranker under
the accepted 250-tree replay protocol. 34,572,000 parameters per fold model.

## The offline evidence was the strongest in the project

| arm | MRR | increment over control | note |
|---|---:|---:|---|
| control (accepted + LINE-direct scalar) | `0.6679147193356217` | — | rerun **bit-exact** against Round 28 |
| `H0-R` exact reconstruction | `0.6871038122560784` | **`+0.0191890929204567`** | **bit-identical** to Round 28 |
| `H0-S2` independent second seed | `0.6869116882129457` | `+0.0189969688773241` | reproduces to 0.00019 |
| `I0-H0` no LINE warm start | `0.6749913838161145` | `+0.0070766644804928` | LINE increment `+0.0121124284399639` |
| `A0-H0` alignment null | `0.6678638342435226` | `-0.0000508850920991` | clean |
| `L0-H0` label null | `0.6630571980956743` | `+0.0002001071270444` | clean |
| `R0` random feature | `0.6681793315194922` | `+0.0002646121838705` | clean |

Folds `+0.017801737889956912` / `+0.02056823484334874`; time halves `+0.015579007987827916` /
`+0.022799228527345217`; source-clustered 95% CI `[+0.01709733980037992, +0.021243471431162085]` —
strictly positive. Physical serving PASS: all 61,051 rows and 6,105,100 cells scored, finite,
deterministic across two independent passes. Every registered gate A-L passed, the confirmation plan
was hash-locked before execution, and no gate was lowered afterward.

Conservative online projection at the then-standing 9.1x replay-to-online shrink:
**`+0.0020876888876181`**, against a pre-registered `+0.0020` requirement.

## What happened online

| quantity | value |
|---|---:|
| observed total | **`1.5494486598552333`** |
| dataset2 component (unchanged member) | `0.6789511047001768` |
| implied dataset1 component | `0.8704975551550565` |
| dataset1 before the read | `0.8963013747474597` |
| **observed dataset1 delta** | **`-0.0258038195924032`** |
| observed total delta | `-0.0258038195924033` |

The implied dataset1 figure is `INFERRED`, not measured: the platform reports a total, and the
inference is sound only because the dataset2 member was byte-identical to the accepted one, fixing
its component by a prior observation.

## The finding

| quantity | value |
|---|---:|
| offline replay increment | `+0.0191890929204567` |
| predicted online gain (9.1x shrink) | `+0.0020876888876181` |
| **actual online delta** | **`-0.0258038195924032`** |
| prediction error | `0.0278915084800213` |
| **realised transfer coefficient** | **`-1.345`** (assumed `+0.109`) |

The shrink model did not merely overstate the magnitude — it got the **sign** wrong, and the realised
change was 1.345x the offline delta in the opposite direction. The precedent that calibrated 9.1x was
the shipped ranker at `+0.0187` offline → `+0.00205` online: almost exactly the same offline
magnitude as Round 29, opposite online outcome. A single-observation shrink model cannot distinguish
those two cases, so **it is retired as a predictor for this family.**

Three explanations are consistent with the persisted evidence. None is established, and Round 29 was
not re-run to adjudicate between them:

1. **The dataset1 replay is not a faithful image of the physical serving chain.** This was already a
   standing caveat before the round.
2. **Representation retraining is not additive with the frozen deterministic postprocessors.** The
   member was built through `src/build_ds1_member.py --control` (`source_slate_recurrence` 4,093
   actions → `test_graph_reciprocity` 1,077 actions). A learned score distribution that differs
   materially from the accepted one changes which rows those rules fire on.
3. **The declared row-normalisation amendment.** It is strictly monotone and therefore rank-preserving
   within a row, but it changes absolute score scale, which a cross-row threshold can depend on.

Explanation 2 best fits the magnitude: the damage is far too large for feature noise, and the offline
arm never exercised the physical chain.

## Consequences

No rescue, recalibration or reinterpretation was performed, and the locked
`no_rescue_after_observation` clause was honoured. The package and all Round-29 evidence are
**deliberately retained** at `outputs/submissions/round29_component/` and
`scratchpad/round-29-fable/`.

The failure is what made Round 30 necessary: with FSSR falsified, the only asset left with a measured
positive online observation was the Round-23 XTE decode. See [round-30.md](round-30.md).

## Standing rule this produced

> Any future learned-representation candidate must be evaluated **through the physical postprocessor
> chain**, not only on the replay surface, before an online read is spent on it. Exact reproduction,
> seed confirmation, clean nulls and a strictly positive clustered interval are jointly insufficient.

Local provenance: `scratchpad/round-29-fable/`, `scratchpad/round-29-opus/adjudication.md`.

One record correction, held here rather than by editing history: the package manifest at
`outputs/submissions/round29_component/r29f_fssr_d1swap.manifest.json` still carries
`"not_uploaded_not_submitted": true`. That was accurate when written and became stale when the
package was uploaded and scored. The file is historical evidence and was not modified.
