# -*- coding: utf-8 -*-
"""Registry of prediction strategies and the active production chain.

Lifecycle vocabulary (one status per strategy, evidence-based, not
location-based):

``SHIPPED_ACTIVE``      part of the current accepted submission
``SHIPPED_SUPERSEDED``  was in an accepted submission, later replaced
``ONLINE_VALIDATED``    passed an isolated online test, never shipped in the accepted state
``CANDIDATE``           passed the offline gate, no online read yet
``STANDBY``             weak-but-positive evidence, deliberately inactive
``CLOSED``              refuted offline or online; do not reopen in the same formulation
``PROBE``               exploratory, no promotion decision

The full evidence-bearing inventory for every strategy lives in
``docs/strategy_inventory.json``; this module carries only what production code
needs -- the callables and their execution order.

The dataset1 postprocessor chain is ORDER-SENSITIVE: reciprocity reads the rank-2
candidate of the recurrence-adjusted matrix, so swapping the two changes the
output. The order below is the one that was online-adjudicated and accepted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .ds1 import source_slate_recurrence, test_graph_reciprocity
from .ds2 import cross_time_exclusivity

SHIPPED_ACTIVE = "SHIPPED_ACTIVE"
SHIPPED_SUPERSEDED = "SHIPPED_SUPERSEDED"
ONLINE_VALIDATED = "ONLINE_VALIDATED"
CANDIDATE = "CANDIDATE"
STANDBY = "STANDBY"
CLOSED = "CLOSED"
PROBE = "PROBE"

LIFECYCLE_STATUSES = (
    SHIPPED_ACTIVE, SHIPPED_SUPERSEDED, ONLINE_VALIDATED,
    CANDIDATE, STANDBY, CLOSED, PROBE,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = REPO_ROOT / "docs" / "strategy_inventory.json"

#: Ordered dataset1 score postprocessors of the accepted submission.
#: Each entry: (strategy id, module, apply callable).
DS1_POSTPROCESSOR_CHAIN: list[tuple[str, object, Callable]] = [
    (source_slate_recurrence.STRATEGY_ID, source_slate_recurrence, source_slate_recurrence.apply),
    (test_graph_reciprocity.STRATEGY_ID, test_graph_reciprocity, test_graph_reciprocity.apply),
]

#: Ordered dataset2 score postprocessors of the accepted submission. One stage:
#: the final cross-time exclusivity decode applied on top of the CRF output.
DS2_POSTPROCESSOR_CHAIN: list[tuple[str, object, Callable]] = [
    (cross_time_exclusivity.STRATEGY_ID, cross_time_exclusivity, cross_time_exclusivity.apply),
]

#: Postprocessor chain per dataset, so an entry point can select by configuration
#: rather than by branching on a dataset name.
POSTPROCESSOR_CHAINS: dict[str, list[tuple[str, object, Callable]]] = {
    "dataset1": DS1_POSTPROCESSOR_CHAIN,
    "dataset2": DS2_POSTPROCESSOR_CHAIN,
}


def load_inventory() -> dict:
    """Load the full strategy lifecycle inventory."""
    with INVENTORY_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def strategies_by_status(status: str) -> list[dict]:
    """All inventory records carrying ``status``."""
    if status not in LIFECYCLE_STATUSES:
        raise ValueError(f"unknown lifecycle status {status!r}")
    return [s for s in load_inventory()["strategies"] if s["lifecycle_status"] == status]


def active_ds1_chain_ids() -> list[str]:
    """Strategy ids of the ordered dataset1 postprocessor chain."""
    return [sid for sid, _, _ in DS1_POSTPROCESSOR_CHAIN]


def active_ds2_chain_ids() -> list[str]:
    """Strategy ids of the ordered dataset2 postprocessor chain."""
    return [sid for sid, _, _ in DS2_POSTPROCESSOR_CHAIN]


def active_chain_ids(dataset: str) -> list[str]:
    """Strategy ids of the ordered postprocessor chain for ``dataset``."""
    if dataset not in POSTPROCESSOR_CHAINS:
        raise ValueError(f"unknown dataset {dataset!r}; "
                         f"known: {sorted(POSTPROCESSOR_CHAINS)}")
    return [sid for sid, _, _ in POSTPROCESSOR_CHAINS[dataset]]
