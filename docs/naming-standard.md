# Repository conventions

These rules apply to source code, tests, configuration, documentation, logs,
generated artifacts, directories, and Git commits.

## Language and tone

- Use formal English for public code, comments, logs, and documentation.
- Preserve official Chinese competition names and required Chinese file names.
- Avoid colloquialisms, decorative symbols, emoji, unsupported certainty, and
  references to a particular assistant or private conversation.
- Define acronyms at first use unless they are conventional library names.

## Python

Follow PEP 8 naming:

| Element | Convention | Example |
|---|---|---|
| module | `snake_case.py` | `graph_reciprocity.py` |
| class | `PascalCase` | `RunContext` |
| function or variable | `snake_case` | `resolve_output_root` |
| constant | `UPPER_SNAKE_CASE` | `DATASETS` |
| test module | `test_<subject>.py` | `test_stage_contract.py` |

Use descriptive mechanism-oriented strategy identifiers. Do not encode a
round, assistant name, concrete seed, or temporary version in a strategy ID.

## Files and directories

- Use lowercase `snake_case` for Python packages and machine-consumed working
  directories.
- Use lowercase `kebab-case.md` for nested prose documents.
- Retain established uppercase names for top-level contracts such as
  `README.md`, `CURRENT_PRODUCTION.md`, and `SUBMISSION_PROTOCOL.md`.
- Add a date only to a genuine immutable snapshot or archival record, using
  `YYYY-MM-DD`.
- Do not create unexplained roots such as `tmp`, `misc`, `new`, `final2`, or
  `_work`.

## Paths and configuration

- Source code must not contain user-specific or host-specific absolute paths.
- Resolve defaults from the repository or package location, not the caller's
  current working directory.
- Expose data and output roots through documented command-line options or a
  single configuration boundary.
- Treat accepted prediction matrices and archives as opaque byte artifacts.

## Generated artifacts and logs

- Generated data belongs under the configured output root, never beside source
  modules.
- A filename must describe its role; do not use subjective suffixes such as
  `latest`, `best`, or `final` without a corresponding immutable record.
- Logs must be machine-neutral, avoid secrets and personal data, and state the
  command, status, and relevant relative artifact path.
- Temporary test artifacts must be created under an explicit temporary root and
  removed after the run.

## Documentation

- Prefer one authoritative location for each mutable fact.
- Link to configuration or commands instead of copying volatile counts.
- Label observations, calculations, hypotheses, and official requirements
  distinctly.
- Use SHA-256 for byte identity and preserve the original line endings of CSV
  artifacts.

## Git

Use Conventional Commits with an ASCII, imperative, single-line subject:

```text
type(scope): concise imperative subject
```

Keep commits atomic. Stage explicit paths, inspect the staged diff, and run the
relevant checks before committing. Do not add generated artifacts, credentials,
personal contact data, or co-author trailers. Rewriting published history
requires an explicit reason, a verified backup, a mapping of affected anchors,
and a lease-protected force push.
