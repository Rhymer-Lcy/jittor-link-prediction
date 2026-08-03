# Competition requirements

This directory separates official competition requirements from project
implementation decisions.

| Document | Purpose |
|---|---|
| [official_competition_dossier.md](official_competition_dossier.md) | Consolidated official rules and project interpretations |
| [source_register.md](source_register.md) | Source provenance, authority, verification status, and open tensions |
| [ab_algorithm_consistency_contract.md](ab_algorithm_consistency_contract.md) | A/B algorithmic consistency and permitted configuration changes |
| [official_package_inventory.md](official_package_inventory.md) | Files permitted in the organiser-facing code package |

For the implemented result, see `../CURRENT_PRODUCTION.md`.

## Authority order

When sources conflict, apply the highest relevant authority:

1. written organiser clarification addressed to the team;
2. later official competition notice;
3. track-specific official page;
4. general official competition rules;
5. official Jittor documentation;
6. archived official capture;
7. project interpretation.

Project policy must not be presented as an official rule. A later official
notice supersedes this repository's interpretation within the scope it covers.

## Maintenance

Update the dossier and source register together when an official source changes
or an organiser clarification is received. Preserve the prior interpretation,
record the revision date and impact, and keep unresolved questions explicit.
