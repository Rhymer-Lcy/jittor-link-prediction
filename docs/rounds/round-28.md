# Round 28 — hard-negative role representation (RACHN): refuted by its own controls

**Outcome: NO CANDIDATE. Nothing was built, nothing was submitted.** Lifecycle
`CLOSED_MECHANISM_REFUTED_AT_TESTED_RUNG`.

Round 28 measured the largest offline gain in the project's history and killed it anyway, on a
pre-registered mechanism audit. That ordering is the point of the round.

## What was tested

`hard_negative_role_representation` (`R28F-RACHN`) — a frozen unit-norm LINE base (43,215 x 400) plus
trainable source and destination role delta tables, trained with full-slate softmax cross-entropy
**plus** a hard-negative auxiliary `0.5 * mean softplus(logit_hn - logit_pos)` using `H = 32` hard
negatives drawn from frozen replay competitor scores. One scalar feature over the control ranker.
34,572,000 parameters per fold model.

Registered hypothesis: hard negatives deliver a materially larger and more transferable ranking gain
than the scalar LINE-direct column.

## The numbers passed

| quantity | value |
|---|---:|
| control A (accepted-aligned) | `0.6638121262583483` |
| control B (control A + LINE-direct scalar) | `0.6679147193356217` |
| full-OOF treatment | `0.6864789067958795` |
| **increment over control B** | **`+0.018564187460257984`** |
| folds | `+0.01746393643287265` / `+0.019657925028662114` |
| time halves | `+0.01513399539081903` / `+0.02199442767882552` |
| source-clustered 95% CI | `[+0.01661860933901996, +0.0206434844615614]` |
| non-repeat / cold slices | `+0.03679513478909429` / `+0.12613037438427818` |
| improved / damaged queries | 24,147 / 22,469 |
| projected online gain | `+0.00204` |

Roughly 4.5x the validated LINE-direct scalar, sign-stable across both folds and both halves, with a
strictly positive clustered interval. Numeric gates 1-6 and 15 passed, and the projection would have
cleared the `+0.002` online gate. **The candidate died scientifically while economically viable**,
which is the correct ordering.

## The mechanism was refuted

Pre-registered kill rule: any control accounting for 30% or more of the treatment gain kills the
candidate.

| control | full-OOF | increment | share | reading |
|---|---:|---:|---:|---|
| `H0` hard-negative auxiliary removed | `0.6871038122560784` | `+0.0191890929204567` | **103.37%** | the auxiliary contributes **nothing**; removing it is slightly better |
| `H1` equal-count random negatives | `0.686949565714526` | `+0.0190348463789043` | **102.54%** | hard-negative *selection* adds nothing over random slate negatives |
| `I0` LINE warm start removed | `0.6746184397255104` | `+0.0067037203898887` | **36.11%** | generic parameter capacity alone clears the threshold |
| `A0` alignment null | `0.6680018749367127` | `+0.000087155601091` | 0.47% | candidate alignment is load-bearing — PASS |
| `L0` label null | `0.6628290662032332` | `-0.0000280247653967` | 0.00% | label alignment is load-bearing — PASS |
| `R0` random feature | `0.6681793315194922` | `+0.0002646121838705` | 1.43% | an extra ranker column alone is insufficient — PASS |

Gate 7 failed. The gain is real; it is simply not produced by the mechanism that was registered. The
three clean alignment nulls are what make this a **mechanism** refutation rather than a measurement
artefact.

## Rung selection

| rung | screening MRR | increment | decision |
|---|---:|---:|---|
| `P0` no propagation | `0.6838613540989652` | `+0.0210042631303353` | **selected**, carried to full OOF |
| `P1` one sparse symmetric normalised step | `0.6801434989647116` | `+0.0172864079960817` | `CLOSED_INFERIOR_RUNG` |
| `P2` average of layers 0, 1, 2 | `0.6808411554216328` | `+0.0179840644530029` | `CLOSED_INFERIOR_RUNG` |

Both propagation rungs passed their screening gate and were still not advanced, under the
pre-registered maximum-one-rung rule.

## The best process decision of the late competition

`H0` — the control built to isolate the hard-negative auxiliary — **outscored the treatment it was
controlling**, at `+0.0191890929204567`. It was not promoted. The pre-registered rule that a control
may not be reinterpreted as a candidate inside the round that generated it was honoured, and `H0` was
carried forward as a **new, separately pre-registered Round-29 hypothesis** instead.

Round 29 then falsified it online. The discipline was still correct: had `H0` been promoted directly
here, the same online failure would have occurred with strictly less evidence explaining it. See
[round-29.md](round-29.md).

## Carried forward

- Hard-negative mining is closed at this rung: inert against both its own removal and random
  negatives.
- Generic parameter capacity accounts for approximately 36% of the gain in this family. Any successor
  must report an equal-capacity arm.
- A control that beats its treatment must be re-registered, never promoted in place.

Local provenance: `scratchpad/round-28-fable/` (experiment plan hash-locked before execution),
`scratchpad/round-28-opus/adjudication.md`.
