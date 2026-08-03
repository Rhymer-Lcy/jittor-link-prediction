# Stage completion contract

The canonical pipeline uses explicit completion records to prevent stale,
partial, or cross-configuration artifacts from being reused.

## Completion definition

A stage is complete only when all of the following are true:

1. the process exited successfully;
2. the expected artifact exists and passes its type-specific validator;
3. the artifact was published atomically;
4. a completion record was written after publication;
5. the record binds the current stage identifier, command, code revision,
   configuration, input hashes, and output hash.

The presence of an output file alone is never evidence of completion.

## Reuse decision

Validated resume is the default. An existing artifact is reusable only when its
completion record is present, internally valid, and matches the current
execution request. A mismatch causes the stage to be rejected with a diagnostic
rather than adopted or repaired automatically.

`--fresh` rejects every existing stage result. It does not delete artifacts.

## Atomic publication

A stage writes to a temporary sibling path, validates the temporary artifact,
and replaces the destination only after validation succeeds. The completion
record is the final write. An interruption can therefore leave temporary data,
but cannot create a valid-looking completion record for an incomplete output.

## Quarantine

An artifact that conflicts with its completion record may be moved under the
configured output root's `_quarantine/` directory. Quarantine preserves
diagnostic evidence and prevents accidental reuse. It is not an archive and may
be removed after the failure has been investigated and any required evidence
has been retained elsewhere.

## Path contract

Data, output, and log locations are resolved from command-line arguments or the
package location. Completion records store repository-relative or output-root-
relative paths where possible; they must not depend on a particular username,
drive letter, or host directory.

## Validators

Validators are specific to each artifact class. Examples include:

- required CSV shape, finite values, and line-ending convention;
- array shape, dtype, and finite-value checks;
- archive member names, order, compression method, and member hashes;
- model and feature-cache metadata consistent with the requested dataset and
  stage configuration.

Validation rules are part of the stage contract. Weakening a validator requires
the same review as changing the stage that produces the artifact.

## Operational commands

```bash
python run_all.py --plan --data-root /path/to/data --output-root /path/to/outputs
python run_all.py --data-root /path/to/data --output-root /path/to/outputs
python run_all.py --fresh --data-root /path/to/data --output-root /path/to/outputs
```

The plan command performs no model computation. It reports the resolved graph
and the disposition of each stage.
