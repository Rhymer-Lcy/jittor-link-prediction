#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Candidate-ID x query-time exposure features for dataset2.

This is the exact 85-column feature block used by the round-11 best gate:

    57 comprehensive non-current-day temporal features
     3 strict non-local lag-0 features
    25 non-local weekday / burst / periodic features

The block was appended directly to the existing 18 ranker columns.  It is not
an OOF score and must not be collapsed to a single scalar if exact gate
reproduction is required.

Modes
-----
``cut-frozen``
    Replays ``exp_hzeval_build_ds2.py`` exactly: construct all 153,420
    simulated slates with RNG 20260723, build the transductive footprint on
    all of them, then gather the saved 60,000 hzeval rows.  "Frozen" refers to
    the backing ranker features being frozen at CUT; the label-free footprint
    deliberately sees the complete simulated inference batch.

``full-test``
    Build the same label-free statistics from all 153,420 physical dataset2
    test rows and emit features in physical CSV row order.

The default CLI only computes a bounded sanity sample.  ``--output`` writes a
streamed NumPy ``.npy`` file through ``open_memmap``; the full-test tensor is
about 5.2 GB, so it is never materialised as one in-memory array.

Public API
----------
``build_generator(mode, ...) -> (generator, default_row_indices)``
``generator.transform_chunk(row_indices) -> float32[Q,100,85]``
``generator.iter_features(row_indices, chunk_size)``
``FEATURE_NAMES``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


CUT = 1261958400.0
N_FOOTPRINT = 153420
N_FEATURE = 60000
N_SLATE = 100
NEG_PER_SLATE = 99.0
REPLAY_SEED = 20260723
ANNULI: Tuple[Tuple[int, int], ...] = (
    (1, 3),
    (4, 7),
    (8, 14),
    (15, 30),
    (31, 60),
    (61, 120),
)
RADII: Tuple[int, ...] = (3, 7, 14, 30, 60, 120)
QUANTILES: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
BURST_THRESHOLDS: Tuple[int, ...] = (2, 3)
BURST_RADII: Tuple[int, ...] = (7, 30, 120)

# Repo root is two levels up from this file (src/footprint_feature.py).
DEFAULT_REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = DEFAULT_REPO / "data" / "data_A" / "dataset2"
# The hzeval replay asset is kept with the rest of the footprint reproduction
# chain under outputs/ (gitignored). It is only needed by cut-frozen / gate
# reproduction; full-test feature generation does not touch it.
DEFAULT_HZEVAL = DEFAULT_REPO / "outputs" / "dataset2-footprint" / "hzeval_ds2.npz"


def _feature_names() -> Tuple[str, ...]:
    names = []
    for a, b in ANNULI:
        names.extend(
            (
                f"poisson_z_past_{a}_{b}",
                f"poisson_z_future_{a}_{b}",
            )
        )
    for a, b in ANNULI:
        names.extend(
            (
                f"conditional_z_past_{a}_{b}",
                f"conditional_z_future_{a}_{b}",
            )
        )
    for a, b in ANNULI:
        names.extend(
            (
                f"positive_residual_past_{a}_{b}",
                f"positive_residual_future_{a}_{b}",
            )
        )
    names.extend(f"kernel_conditional_z_pm{r}" for r in RADII)
    names.extend(f"future_minus_past_z_pm{r}" for r in RADII)
    names.extend(
        (
            "leave_day_first_distance",
            "leave_day_q25_distance",
            "leave_day_median_distance",
            "leave_day_q75_distance",
            "leave_day_last_distance",
        )
    )
    names.extend(
        (
            "log1p_candidate_frequency",
            "positive_mass_over_noise",
            "leave_day_cdf_delta",
            "leave_day_cdf_z",
        )
    )
    names.extend(
        (
            "strict_nonlocal_lag0_z",
            "strict_nonlocal_lag0_share_delta",
            "strict_nonlocal_lag0_positive_residual",
        )
    )
    names.extend(
        (
            "leave_day_same_weekday_z",
            "leave_day_same_weekday_share_delta",
            "leave_day_same_weekday_positive_residual",
        )
    )
    for threshold in BURST_THRESHOLDS:
        for radius in BURST_RADII:
            names.extend(
                (
                    f"burst_ge{threshold}_past_active_days_{radius}",
                    f"burst_ge{threshold}_future_active_days_{radius}",
                    f"burst_ge{threshold}_future_minus_past_z_{radius}",
                )
            )
        names.extend(
            (
                f"burst_ge{threshold}_previous_distance",
                f"burst_ge{threshold}_next_distance",
            )
        )
    assert len(names) == 85
    return tuple(names)


FEATURE_NAMES: Tuple[str, ...] = _feature_names()
N_FEATURES = len(FEATURE_NAMES)


def _candidate_columns(frame: pd.DataFrame) -> Sequence[str]:
    expected = [f"c{i}" for i in range(1, N_SLATE + 1)]
    missing = [c for c in expected if c not in frame.columns]
    if missing:
        raise ValueError(f"test CSV is missing candidate columns: {missing[:5]}")
    return expected


def _file_md5(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class FootprintFeatureGenerator:
    """Chunked implementation of the exact round-11 85-column block."""

    def __init__(
        self,
        candidates: np.ndarray,
        src: np.ndarray,
        time: np.ndarray,
        *,
        universe_size: Optional[int] = None,
        mode: str = "arrays",
    ) -> None:
        candidates = np.asarray(candidates)
        src = np.asarray(src)
        time = np.asarray(time)
        if candidates.ndim != 2 or candidates.shape[1] != N_SLATE:
            raise ValueError(
                f"candidates must have shape (R,{N_SLATE}); got {candidates.shape}"
            )
        if len(src) != len(candidates) or len(time) != len(candidates):
            raise ValueError("src/time lengths must equal candidate row count")
        if not np.issubdtype(candidates.dtype, np.integer):
            raise TypeError("candidate IDs must be integer")
        if candidates.size == 0 or int(candidates.min()) < 0:
            raise ValueError("candidate IDs must be non-negative and non-empty")

        self.mode = str(mode)
        self.candidates = np.ascontiguousarray(candidates, dtype=np.int32)
        self.src = np.ascontiguousarray(src, dtype=np.int64)
        self.time = np.ascontiguousarray(time, dtype=np.float64)
        self.R, self.M = self.candidates.shape
        self.C = int(self.candidates.max()) + 1
        actual_universe = int(np.unique(self.candidates).size)
        self.universe_size = (
            actual_universe if universe_size is None else int(universe_size)
        )
        if self.universe_size <= 0:
            raise ValueError("universe_size must be positive")

        unique_time, time_inverse = np.unique(self.time, return_inverse=True)
        day_float = (unique_time - unique_time.min()) / 86400.0
        day = np.rint(day_float).astype(np.int64)
        if np.max(np.abs(day_float - day)) > 1e-6:
            raise ValueError("timestamps are not aligned to integral 86400-second days")
        self.row_day = day[time_inverse]
        self.D = int(day.max()) + 1
        if self.D >= 1000:
            raise ValueError(
                "exact burst implementation uses int16 sentinels and requires D < 1000"
            )

        flat_c = self.candidates.ravel().astype(np.int64, copy=False)
        flat_d = np.repeat(self.row_day, self.M)
        # Match the audited gate's uint16 candidate-day table.  A crude bound
        # of ``rows_per_day * 100`` is far above uint16 even though the actual
        # mass is spread over ~110k candidates, so it is not a useful safety
        # test for this dataset.
        self.ct = np.zeros((self.C, self.D), dtype=np.uint16)
        np.add.at(self.ct, (flat_c, flat_d), 1)
        self.cs = np.pad(
            np.cumsum(self.ct, axis=1, dtype=np.uint32), ((0, 0), (1, 0))
        )
        self.cf = np.bincount(flat_c, minlength=self.C).astype(np.float64)
        self.rv = np.bincount(self.row_day, minlength=self.D).astype(np.float64)
        self.rvs = np.pad(np.cumsum(self.rv), (1, 0))
        self.lambda_total = (
            NEG_PER_SLATE * float(self.R) / float(self.universe_size)
        )

        self._event_id: Optional[np.ndarray] = None
        self._event_n: Optional[np.ndarray] = None
        self._event_pair_key: Optional[np.ndarray] = None
        self._event_pair_count: Optional[np.ndarray] = None
        self._row_candidate_key: Optional[np.ndarray] = None
        self._dow_ct: Optional[np.ndarray] = None
        self._dow_volume: Optional[np.ndarray] = None
        self._burst_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    @classmethod
    def from_arrays(
        cls,
        candidates: np.ndarray,
        src: np.ndarray,
        time: np.ndarray,
        *,
        universe_size: Optional[int] = None,
    ) -> "FootprintFeatureGenerator":
        """Construct from an inference-batch-shaped exposure population.

        Row indices used later by :meth:`transform_chunk` refer to this compact
        array order.  In particular, the strict ``+/-2`` exclusion is in this
        compact order, matching the original hzeval replay.
        """

        return cls(
            candidates,
            src,
            time,
            universe_size=universe_size,
            mode="arrays",
        )

    def _prepare_event_cache(self) -> None:
        if self._event_id is not None:
            return
        event_key = self.src * np.int64(self.D) + self.row_day
        _, event_id = np.unique(event_key, return_inverse=True)
        self._event_id = event_id.astype(np.int64, copy=False)
        self._event_n = np.bincount(self._event_id).astype(np.int32)

        flat_c = self.candidates.ravel().astype(np.int64, copy=False)
        event_pair = (
            np.repeat(self._event_id, self.M) * np.int64(self.C) + flat_c
        )
        pair_key, pair_count = np.unique(event_pair, return_counts=True)
        self._event_pair_key = pair_key
        self._event_pair_count = pair_count.astype(np.int32)

        sorted_candidates = np.sort(self.candidates.astype(np.int64), axis=1)
        self._row_candidate_key = (
            np.arange(self.R, dtype=np.int64)[:, None] * np.int64(self.C)
            + sorted_candidates
        ).ravel()

    def _prepare_weekday_cache(self) -> None:
        if self._dow_ct is not None:
            return
        dow_ct = np.zeros((self.C, 7), dtype=np.uint32)
        all_days = np.arange(self.D)
        for dow in range(7):
            dow_ct[:, dow] = self.ct[:, all_days % 7 == dow].sum(
                axis=1, dtype=np.uint32
            )
        self._dow_ct = dow_ct
        self._dow_volume = np.asarray(
            [self.rv[all_days % 7 == dow].sum() for dow in range(7)],
            dtype=np.float64,
        )

    def _prepare_burst_cache(
        self, threshold: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self._burst_cache.get(int(threshold))
        if cached is not None:
            return cached
        active = self.ct >= int(threshold)
        active_cs = np.pad(
            np.cumsum(active, axis=1, dtype=np.uint16), ((0, 0), (1, 0))
        )
        day_index = np.arange(self.D, dtype=np.int16)
        previous = np.maximum.accumulate(
            np.where(active, day_index[None, :], np.int16(-1000)), axis=1
        )
        nxt = np.minimum.accumulate(
            np.where(active, day_index[None, :], np.int16(1000))[:, ::-1],
            axis=1,
        )[:, ::-1]
        cached = (active_cs, previous, nxt)
        self._burst_cache[int(threshold)] = cached
        return cached

    def _exact_summary(
        self,
        obs: np.ndarray,
        total: np.ndarray,
        probability: np.ndarray,
        excluded_negative_lambda: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        exp = total * probability[:, None]
        var = exp * (1.0 - probability[:, None]) + 1.0
        z = ((obs - exp) / np.sqrt(var)).astype(np.float32)
        share = (
            np.divide(obs, total, out=np.zeros_like(obs), where=total > 0)
            - probability[:, None]
        ).astype(np.float32)
        positive_denominator = (
            np.maximum(total - excluded_negative_lambda, 0.0)
            + np.sqrt(self.lambda_total)
            + 1.0
        )
        positive = (
            np.maximum(obs - exp, 0.0) / positive_denominator
        ).astype(np.float32)
        return z, share, positive

    def _leave_day_quantile(
        self,
        candidates: np.ndarray,
        query_day: np.ndarray,
        probability: float,
    ) -> np.ndarray:
        q, m = candidates.shape
        flat_c = candidates.ravel()
        flat_d = np.repeat(query_day, m)
        flat_current = self.ct[flat_c, flat_d].astype(np.float32)
        flat_total = self.cf[flat_c].astype(np.float32) - flat_current
        target = (
            np.floor(
                float(probability) * np.maximum(flat_total - 1.0, 0.0)
            ).astype(np.int32)
            + 1
        )
        low = np.zeros(len(flat_c), dtype=np.int16)
        high = np.full(len(flat_c), self.D - 1, dtype=np.int16)
        for _ in range(10):
            mid = (
                (low.astype(np.int32) + high.astype(np.int32)) // 2
            ).astype(np.int16)
            cumulative = self.cs[
                flat_c, mid.astype(np.int32) + 1
            ].astype(np.float32)
            cumulative -= flat_current * (flat_d <= mid)
            go_left = cumulative >= target
            high = np.where(go_left, mid, high).astype(np.int16)
            low = np.where(go_left, low, mid + 1).astype(np.int16)
        quantile_day = low.astype(np.float32)
        quantile_day[flat_total <= 0] = flat_d[flat_total <= 0]
        return (
            (flat_d.astype(np.float32) - quantile_day)
            / max(self.D - 1, 1)
        ).reshape(q, m)

    def transform_chunk(self, row_indices: Sequence[int]) -> np.ndarray:
        """Return exact ``float32[Q,100,85]`` features for population rows."""

        idx = np.asarray(row_indices, dtype=np.int64)
        if idx.ndim != 1:
            raise ValueError("row_indices must be one-dimensional")
        if len(idx) == 0:
            return np.empty((0, self.M, N_FEATURES), dtype=np.float32)
        if int(idx.min()) < 0 or int(idx.max()) >= self.R:
            raise IndexError("row index outside exposure population")

        candidates = self.candidates[idx].astype(np.int64, copy=False)
        query_day = self.row_day[idx]
        q = len(idx)
        output = np.empty((q, self.M, N_FEATURES), dtype=np.float32)
        col = 0
        candidate_frequency = self.cf[candidates].astype(np.float32)
        positive_mass = np.maximum(
            candidate_frequency - self.lambda_total, 0.0
        )

        annulus_records = []
        for a, b in ANNULI:
            past_lo = np.maximum(0, query_day - b)
            past_hi = np.maximum(0, query_day - a + 1)
            future_lo = np.minimum(self.D, query_day + a)
            future_hi = np.minimum(self.D, query_day + b + 1)
            pair = []
            for lo, hi in ((past_lo, past_hi), (future_lo, future_hi)):
                obs = (
                    self.cs[candidates, hi[:, None]]
                    - self.cs[candidates, lo[:, None]]
                ).astype(np.float32)
                volume = self.rvs[hi] - self.rvs[lo]
                pair.append((obs, volume))
                lam = NEG_PER_SLATE * volume / self.universe_size
                output[:, :, col] = (
                    (obs - lam[:, None]) / np.sqrt(lam[:, None] + 1.0)
                ).astype(np.float32)
                col += 1
            annulus_records.append(pair)

        for pair in annulus_records:
            for obs, volume in pair:
                p_bin = (volume / self.R).astype(np.float32)
                expected = candidate_frequency * p_bin[:, None]
                variance = expected * (1.0 - p_bin[:, None]) + 1.0
                output[:, :, col] = (
                    (obs - expected) / np.sqrt(variance)
                ).astype(np.float32)
                col += 1

        positive_denominator = (
            positive_mass + np.sqrt(self.lambda_total) + 1.0
        )
        for pair in annulus_records:
            for obs, volume in pair:
                p_bin = (volume / self.R).astype(np.float32)
                expected = candidate_frequency * p_bin[:, None]
                output[:, :, col] = (
                    np.maximum(obs - expected, 0.0) / positive_denominator
                ).astype(np.float32)
                col += 1

        for radius in RADII:
            lo = np.maximum(0, query_day - radius)
            hi = np.minimum(self.D, query_day + radius + 1)
            obs = (
                self.cs[candidates, hi[:, None]]
                - self.cs[candidates, lo[:, None]]
                - self.cs[candidates, (query_day + 1)[:, None]]
                + self.cs[candidates, query_day[:, None]]
            ).astype(np.float32)
            volume = (
                (self.rvs[hi] - self.rvs[lo])
                - (self.rvs[query_day + 1] - self.rvs[query_day])
            )
            p_bin = (volume / self.R).astype(np.float32)
            expected = candidate_frequency * p_bin[:, None]
            variance = expected * (1.0 - p_bin[:, None]) + 1.0
            output[:, :, col] = (
                (obs - expected) / np.sqrt(variance)
            ).astype(np.float32)
            col += 1

        for radius in RADII:
            lo = np.maximum(0, query_day - radius)
            hi = np.minimum(self.D, query_day + radius + 1)
            past_obs = (
                self.cs[candidates, query_day[:, None]]
                - self.cs[candidates, lo[:, None]]
            ).astype(np.float32)
            future_obs = (
                self.cs[candidates, hi[:, None]]
                - self.cs[candidates, (query_day + 1)[:, None]]
            ).astype(np.float32)
            past_volume = self.rvs[query_day] - self.rvs[lo]
            future_volume = self.rvs[hi] - self.rvs[query_day + 1]
            past_p = (past_volume / self.R)[:, None]
            future_p = (future_volume / self.R)[:, None]
            past_expected = candidate_frequency * past_p
            future_expected = candidate_frequency * future_p
            variance = (
                past_expected * (1.0 - past_p)
                + future_expected * (1.0 - future_p)
                + 1.0
            )
            output[:, :, col] = (
                (
                    (future_obs - future_expected)
                    - (past_obs - past_expected)
                )
                / np.sqrt(variance)
            ).astype(np.float32)
            col += 1

        for probability in QUANTILES:
            output[:, :, col] = self._leave_day_quantile(
                candidates, query_day, probability
            )
            col += 1

        current = self.ct[candidates, query_day[:, None]].astype(np.float32)
        total_excluding_day = np.maximum(
            candidate_frequency - current, 0.0
        )
        past_all = self.cs[
            candidates, query_day[:, None]
        ].astype(np.float32)
        global_total_excluding_day = (
            self.R
            - (self.rvs[query_day + 1] - self.rvs[query_day])
        )
        global_past = self.rvs[query_day]
        global_cdf = np.divide(
            global_past,
            global_total_excluding_day,
            out=np.zeros(q),
            where=global_total_excluding_day > 0,
        )
        observed_cdf = np.divide(
            past_all,
            total_excluding_day,
            out=np.zeros_like(past_all),
            where=total_excluding_day > 0,
        )
        cdf_delta = (observed_cdf - global_cdf[:, None]).astype(np.float32)
        cdf_variance = (
            total_excluding_day
            * global_cdf[:, None]
            * (1.0 - global_cdf[:, None])
            + 1.0
        )
        cdf_z = (
            (
                past_all
                - total_excluding_day * global_cdf[:, None]
            )
            / np.sqrt(cdf_variance)
        ).astype(np.float32)
        output[:, :, col] = np.log1p(candidate_frequency).astype(np.float32)
        col += 1
        output[:, :, col] = (
            positive_mass / (np.sqrt(self.lambda_total) + 1.0)
        ).astype(np.float32)
        col += 1
        output[:, :, col] = cdf_delta
        col += 1
        output[:, :, col] = cdf_z
        col += 1
        assert col == 57

        self._prepare_event_cache()
        assert self._event_id is not None
        assert self._event_n is not None
        assert self._event_pair_key is not None
        assert self._event_pair_count is not None
        assert self._row_candidate_key is not None

        query_event = self._event_id[idx]
        query_pair = (
            query_event[:, None] * np.int64(self.C) + candidates
        )
        pair_location = np.searchsorted(self._event_pair_key, query_pair)
        if np.any(pair_location >= len(self._event_pair_key)):
            raise RuntimeError("query candidate missing from its exposure event")
        event_count = self._event_pair_count[pair_location].astype(np.float32)

        adjacent_extra = np.zeros((q, self.M), dtype=np.float32)
        adjacent_rows = np.zeros(q, dtype=np.float32)
        for offset in (-2, -1, 1, 2):
            neighbour = idx + offset
            valid = (neighbour >= 0) & (neighbour < self.R)
            safe = np.clip(neighbour, 0, self.R - 1)
            valid &= self.row_day[safe] == query_day
            valid &= self._event_id[safe] != query_event
            keys = safe[:, None] * np.int64(self.C) + candidates
            location = np.searchsorted(self._row_candidate_key, keys)
            bounded = np.minimum(location, len(self._row_candidate_key) - 1)
            hit = (
                (location < len(self._row_candidate_key))
                & (self._row_candidate_key[bounded] == keys)
            )
            adjacent_extra += hit.astype(np.float32) * valid[:, None]
            adjacent_rows += valid.astype(np.float32)

        day_rows = (
            self.rvs[query_day + 1] - self.rvs[query_day]
        ).astype(np.float32)
        excluded_rows = (
            self._event_n[query_event].astype(np.float32) + adjacent_rows
        )
        nonlocal_obs = np.maximum(
            current - event_count - adjacent_extra, 0.0
        )
        nonlocal_total = np.maximum(
            candidate_frequency - event_count - adjacent_extra, 0.0
        )
        nonlocal_same_rows = np.maximum(day_rows - excluded_rows, 0.0)
        nonlocal_denominator_rows = np.maximum(
            self.R - excluded_rows, 1.0
        )
        nonlocal_probability = (
            nonlocal_same_rows / nonlocal_denominator_rows
        ).astype(np.float32)
        lag0 = self._exact_summary(
            nonlocal_obs,
            nonlocal_total,
            nonlocal_probability,
            (
                NEG_PER_SLATE
                * nonlocal_denominator_rows[:, None]
                / self.universe_size
            ),
        )
        for feature in lag0:
            output[:, :, col] = feature
            col += 1
        assert col == 60

        self._prepare_weekday_cache()
        assert self._dow_ct is not None
        assert self._dow_volume is not None
        weekday = query_day % 7
        weekday_obs = (
            self._dow_ct[candidates, weekday[:, None]].astype(np.float32)
            - current
        )
        weekday_volume = self._dow_volume[weekday] - day_rows
        weekday_total = np.maximum(
            candidate_frequency - current, 0.0
        )
        rows_excluding_day = np.maximum(self.R - day_rows, 1.0)
        weekday_probability = (
            weekday_volume / rows_excluding_day
        ).astype(np.float32)
        weekday_features = self._exact_summary(
            weekday_obs,
            weekday_total,
            weekday_probability,
            (
                NEG_PER_SLATE
                * rows_excluding_day[:, None]
                / self.universe_size
            ),
        )
        for feature in weekday_features:
            output[:, :, col] = feature
            col += 1

        for threshold in BURST_THRESHOLDS:
            active_cs, previous, nxt = self._prepare_burst_cache(threshold)
            for radius in BURST_RADII:
                lo = np.maximum(0, query_day - radius)
                hi = np.minimum(self.D, query_day + radius + 1)
                active_past = (
                    active_cs[candidates, query_day[:, None]]
                    - active_cs[candidates, lo[:, None]]
                ).astype(np.float32)
                active_future = (
                    active_cs[candidates, hi[:, None]]
                    - active_cs[
                        candidates, (query_day + 1)[:, None]
                    ]
                ).astype(np.float32)
                output[:, :, col] = np.log1p(active_past).astype(np.float32)
                col += 1
                output[:, :, col] = np.log1p(active_future).astype(np.float32)
                col += 1
                output[:, :, col] = (
                    (active_future - active_past)
                    / np.sqrt(active_future + active_past + 1.0)
                ).astype(np.float32)
                col += 1

            previous_index = np.maximum(query_day - 1, 0)
            next_index = np.minimum(query_day + 1, self.D - 1)
            previous_day = previous[
                candidates, previous_index[:, None]
            ].astype(np.float32)
            next_day = nxt[candidates, next_index[:, None]].astype(np.float32)
            previous_distance = np.where(
                (query_day[:, None] > 0) & (previous_day >= 0),
                query_day[:, None] - previous_day,
                self.D,
            ).astype(np.float32)
            next_distance = np.where(
                (query_day[:, None] < self.D - 1)
                & (next_day < self.D),
                next_day - query_day[:, None],
                self.D,
            ).astype(np.float32)
            output[:, :, col] = (previous_distance / self.D).astype(
                np.float32
            )
            col += 1
            output[:, :, col] = (next_distance / self.D).astype(np.float32)
            col += 1

        assert col == N_FEATURES, (col, N_FEATURES)
        if not np.isfinite(output).all():
            bad = np.argwhere(~np.isfinite(output))[0]
            raise FloatingPointError(
                f"non-finite footprint feature at local index {tuple(bad)}"
            )
        return output

    def iter_features(
        self,
        row_indices: Optional[Sequence[int]] = None,
        *,
        chunk_size: int = 1024,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield ``(row_indices, features)`` without creating the full tensor."""

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        indices = (
            np.arange(self.R, dtype=np.int64)
            if row_indices is None
            else np.asarray(row_indices, dtype=np.int64)
        )
        for begin in range(0, len(indices), int(chunk_size)):
            part = indices[begin : begin + int(chunk_size)]
            yield part, self.transform_chunk(part)

    def sanity_report(
        self,
        row_indices: Sequence[int],
        *,
        chunk_size: int = 512,
    ) -> dict:
        """Stream per-column distribution checks over the requested rows."""

        count = 0
        total = np.zeros(N_FEATURES, dtype=np.float64)
        total_sq = np.zeros(N_FEATURES, dtype=np.float64)
        minima = np.full(N_FEATURES, np.inf, dtype=np.float64)
        maxima = np.full(N_FEATURES, -np.inf, dtype=np.float64)
        finite = True
        for _, features in self.iter_features(
            row_indices, chunk_size=chunk_size
        ):
            matrix = features.reshape(-1, N_FEATURES).astype(
                np.float64, copy=False
            )
            finite &= bool(np.isfinite(matrix).all())
            count += len(matrix)
            total += matrix.sum(axis=0)
            total_sq += np.square(matrix).sum(axis=0)
            minima = np.minimum(minima, matrix.min(axis=0))
            maxima = np.maximum(maxima, matrix.max(axis=0))
        if count == 0:
            raise ValueError("sanity_report received no rows")
        mean = total / count
        variance = np.maximum(total_sq / count - mean * mean, 0.0)
        columns = []
        for i, name in enumerate(FEATURE_NAMES):
            columns.append(
                {
                    "index": i,
                    "name": name,
                    "mean": float(mean[i]),
                    "std": float(np.sqrt(variance[i])),
                    "min": float(minima[i]),
                    "max": float(maxima[i]),
                }
            )
        return {
            "mode": self.mode,
            "population_rows": int(self.R),
            "evaluated_query_rows": int(count // self.M),
            "candidate_cells": int(count),
            "slate_size": int(self.M),
            "feature_count": N_FEATURES,
            "universe_size": int(self.universe_size),
            "days": int(self.D),
            "lambda_total": float(self.lambda_total),
            "all_finite": bool(finite),
            "columns": columns,
        }


def _load_dataset(dataset_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = dataset_dir / "train.csv"
    test_path = dataset_dir / "test.csv"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            f"expected train.csv and test.csv under {dataset_dir}"
        )
    train = pd.read_csv(train_path).drop_duplicates(
        subset=["src", "dst", "time"]
    ).reset_index(drop=True)
    train["src"] = train["src"].astype(np.int64)
    train["dst"] = train["dst"].astype(np.int64)
    train["time"] = train["time"].astype(float)
    test = pd.read_csv(test_path)
    return train, test


def build_cut_frozen(
    dataset_dir: Path = DEFAULT_DATASET,
    hzeval_path: Path = DEFAULT_HZEVAL,
    *,
    verify_hzeval: bool = True,
) -> Tuple[FootprintFeatureGenerator, np.ndarray]:
    """Exact full-footprint replay followed by the saved 60k gather."""

    train, test = _load_dataset(Path(dataset_dir))
    candidate_columns = _candidate_columns(test)
    num_entity = int(max(train["src"].max(), train["dst"].max())) + 1
    split1 = train[train["time"] > CUT].reset_index(drop=True)
    universe = np.unique(
        np.clip(
            test[candidate_columns].to_numpy(np.int64).ravel(),
            0,
            num_entity - 1,
        )
    )
    test_src = set(test["src"].astype(int).values)
    events = split1[split1["src"].isin(test_src)].reset_index(drop=True)
    rng = np.random.default_rng(REPLAY_SEED)
    selected = (
        rng.choice(len(events), N_FOOTPRINT, replace=False)
        if len(events) > N_FOOTPRINT
        else np.arange(len(events))
    )
    events = events.iloc[np.sort(selected)].reset_index(drop=True)
    r = len(events)
    src = events["src"].to_numpy(np.int64)
    truth = events["dst"].to_numpy(np.int64)
    time = events["time"].to_numpy(float)

    slates = np.empty((r, N_SLATE), dtype=np.int64)
    draw = universe[
        rng.integers(0, len(universe), size=(r, N_SLATE + 24))
    ]
    for i in range(r):
        values = draw[i]
        values = values[
            (values != truth[i]) & (values != src[i])
        ]
        values = pd.unique(values)[: N_SLATE - 1]
        if len(values) < N_SLATE - 1:
            extra = universe[rng.integers(0, len(universe), size=400)]
            extra = extra[
                (extra != truth[i]) & (extra != src[i])
            ]
            values = pd.unique(
                np.concatenate([values, extra])
            )[: N_SLATE - 1]
        if len(values) != N_SLATE - 1:
            raise RuntimeError(f"failed to fill replay slate {i}")
        slates[i, : N_SLATE - 1] = values
        slates[i, N_SLATE - 1] = truth[i]

    keep = np.sort(rng.choice(r, min(N_FEATURE, r), replace=False))
    if verify_hzeval:
        hzeval_path = Path(hzeval_path)
        if not hzeval_path.is_file():
            raise FileNotFoundError(f"hzeval asset not found: {hzeval_path}")
        with np.load(hzeval_path) as asset:
            if not np.array_equal(
                slates[keep].astype(np.int32), asset["cand"]
            ):
                raise AssertionError("replayed candidates do not match hzeval")
            if not np.array_equal(
                src[keep].astype(np.int32), asset["src"]
            ):
                raise AssertionError("replayed src does not match hzeval")
            if not np.array_equal(time[keep], asset["t"]):
                raise AssertionError("replayed time does not match hzeval")

    generator = FootprintFeatureGenerator(
        slates,
        src,
        time,
        universe_size=len(universe),
        mode="cut-frozen",
    )
    return generator, keep.astype(np.int64)


def build_full_test(
    dataset_dir: Path = DEFAULT_DATASET,
) -> Tuple[FootprintFeatureGenerator, np.ndarray]:
    """Use every physical dataset2 test row as the exposure population."""

    _, test = _load_dataset(Path(dataset_dir))
    candidate_columns = _candidate_columns(test)
    candidates = test[candidate_columns].to_numpy(np.int64)
    src = test["src"].to_numpy(np.int64)
    time = test["time"].to_numpy(float)
    universe_size = int(np.unique(candidates).size)
    generator = FootprintFeatureGenerator(
        candidates,
        src,
        time,
        universe_size=universe_size,
        mode="full-test",
    )
    return generator, np.arange(len(test), dtype=np.int64)


def build_generator(
    mode: str,
    dataset_dir: Path = DEFAULT_DATASET,
    hzeval_path: Path = DEFAULT_HZEVAL,
    *,
    verify_hzeval: bool = True,
) -> Tuple[FootprintFeatureGenerator, np.ndarray]:
    mode = mode.lower()
    if mode == "cut-frozen":
        return build_cut_frozen(
            dataset_dir,
            hzeval_path,
            verify_hzeval=verify_hzeval,
        )
    if mode == "full-test":
        return build_full_test(dataset_dir)
    raise ValueError(f"unsupported mode: {mode}")


def write_npy(
    generator: FootprintFeatureGenerator,
    row_indices: Sequence[int],
    output_path: Path,
    *,
    chunk_size: int = 512,
) -> dict:
    """Stream an ``(Q,100,85)`` float32 tensor to an open-memmap ``.npy``."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    indices = np.asarray(row_indices, dtype=np.int64)
    mapped = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(indices), N_SLATE, N_FEATURES),
    )
    cursor = 0
    for _, features in generator.iter_features(
        indices, chunk_size=chunk_size
    ):
        mapped[cursor : cursor + len(features)] = features
        cursor += len(features)
        mapped.flush()
    del mapped
    metadata = {
        "mode": generator.mode,
        "shape": [int(len(indices)), N_SLATE, N_FEATURES],
        "dtype": "float32",
        "feature_names": list(FEATURE_NAMES),
        "output": str(output_path.resolve()),
        "md5": _file_md5(output_path),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata


def _sanity_indices(
    default_indices: np.ndarray, requested_rows: int
) -> np.ndarray:
    if requested_rows == 0 or requested_rows >= len(default_indices):
        return default_indices
    if requested_rows < 0:
        raise ValueError("--sanity-rows must be non-negative")
    positions = np.linspace(
        0, len(default_indices) - 1, requested_rows, dtype=np.int64
    )
    return default_indices[positions]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("cut-frozen", "full-test"),
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=DEFAULT_DATASET
    )
    parser.add_argument(
        "--hzeval", type=Path, default=DEFAULT_HZEVAL
    )
    parser.add_argument(
        "--no-verify-hzeval",
        action="store_true",
        help="skip the exact cand/src/time assertion in cut-frozen mode",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=512
    )
    parser.add_argument(
        "--sanity-rows",
        type=int,
        default=512,
        help="evenly spaced query rows; 0 means every output row",
    )
    parser.add_argument(
        "--sanity-json",
        type=Path,
        help="optional path for the full per-column sanity report",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional streamed .npy output; full-test is about 5.2 GB",
    )
    args = parser.parse_args(argv)

    generator, default_indices = build_generator(
        args.mode,
        args.dataset_dir,
        args.hzeval,
        verify_hzeval=not args.no_verify_hzeval,
    )
    selected = _sanity_indices(default_indices, args.sanity_rows)
    report = generator.sanity_report(
        selected, chunk_size=args.chunk_size
    )
    compact = dict(report)
    compact["columns"] = {
        item["name"]: {
            "mean": item["mean"],
            "std": item["std"],
            "min": item["min"],
            "max": item["max"],
        }
        for item in report["columns"]
    }
    print(json.dumps(compact, ensure_ascii=False))
    if args.sanity_json:
        args.sanity_json.parent.mkdir(parents=True, exist_ok=True)
        args.sanity_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if args.output:
        metadata = write_npy(
            generator,
            default_indices,
            args.output,
            chunk_size=args.chunk_size,
        )
        print(json.dumps(metadata, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
