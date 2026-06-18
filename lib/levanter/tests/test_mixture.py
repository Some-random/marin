# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import jax
import numpy as np
import pytest

from levanter.data import ListAsyncDataset, MixtureDataset, PermutationDataset
from levanter.data.dataset import AsyncDataset
from levanter.data.mixture import ConcatDataset, StopStrategy, rescale_mixture_schedule_for_batch_schedule
from levanter.schedule import BatchSchedule, ScheduleStep


def datasets():
    ds1 = ListAsyncDataset([1, 2, 3, 4, 5])
    ds2 = ListAsyncDataset([10, 20, 30, 40, 50])
    ds3 = ListAsyncDataset([100, 200, 300, 400, 500])
    return {"ds1": ds1, "ds2": ds2, "ds3": ds3}


def weights():
    return {"ds1": 0.5, "ds2": 0.3, "ds3": 0.2}


def block_size():
    return 10


def key():
    return jax.random.PRNGKey(42)


class InfiniteCounterDataset(AsyncDataset[int]):
    async def async_len(self) -> int:
        raise ValueError("Infinite dataset has no length")

    def is_finite(self) -> bool:
        return False

    async def get_batch(self, indices: Sequence[int]) -> Sequence[int]:
        return [1000 + idx for idx in indices]


@pytest.mark.asyncio
async def test_mixture_dataset_getitem():
    mixture_ds = MixtureDataset(datasets(), weights(), 10, key=key(), randomize_blocks=False)

    item = await mixture_ds.getitem_async(0)
    assert item in [1, 10, 100], f"Unexpected item: {item}"


@pytest.mark.asyncio
async def test_mixture_dataset_get_batch():
    mixture_ds = MixtureDataset(datasets(), weights(), 10, key=key(), randomize_blocks=False)

    batch = await mixture_ds.get_batch([0, 1, 2])
    assert len(batch) == 3
    assert all(item in [1, 2, 3, 10, 20, 30, 100, 200, 300] for item in batch)


@pytest.mark.asyncio
async def test_mixture_dataset_block_assignments():
    mixture_ds = MixtureDataset(datasets(), weights(), 10, key=key())

    block_assignment = mixture_ds._get_block(0)
    assert block_assignment is not None
    assert len(block_assignment) == 10


@pytest.mark.asyncio
async def test_mixture_dataset_stop_strategy_first():
    mixture_ds = MixtureDataset(
        datasets(),
        weights(),
        10,
        key=key(),
        randomize_blocks=False,
        stop_strategy=StopStrategy.FIRST_STOP_STRATEGY,
    )

    assert mixture_ds.is_finite()
    assert await mixture_ds.async_len() == 5
    assert await mixture_ds.get_batch([0, 1, 2, 3, 4]) == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_mixture_dataset_stop_strategy_all():
    mixture_ds = MixtureDataset(
        datasets(),
        weights(),
        10,
        key=key(),
        randomize_blocks=False,
        stop_strategy=StopStrategy.ALL_STOP_STRATEGY,
    )

    assert mixture_ds.is_finite()
    assert await mixture_ds.async_len() == 29
    assert await mixture_ds.getitem_async(10) == 1  # wraparound of ds1
    assert await mixture_ds.getitem_async(28) == 500


@pytest.mark.asyncio
async def test_mixture_dataset_all_strategy_can_exhaust_before_tail_stage():
    staged_weights = [(0, {"ds1": 0.5, "ds2": 0.5}), (20, {"ds1": 1.0, "ds2": 0.0})]
    dses = {"ds1": ListAsyncDataset([1, 2, 3, 4, 5]), "ds2": ListAsyncDataset([10, 20, 30, 40, 50])}
    mixture_ds = MixtureDataset(
        dses,
        staged_weights,
        10,
        key=key(),
        randomize_blocks=False,
        stop_strategy=StopStrategy.ALL_STOP_STRATEGY,
    )

    assert mixture_ds.is_finite()
    assert await mixture_ds.async_len() == 10


@pytest.mark.asyncio
async def test_mixture_dataset_first_strategy_exhausts_before_tail_stage():
    staged_weights = [(0, {"finite": 1.0, "infinite": 0.0}), (10, {"finite": 0.0, "infinite": 1.0})]
    dses = {"finite": ListAsyncDataset([1, 2, 3, 4, 5]), "infinite": InfiniteCounterDataset()}
    mixture_ds = MixtureDataset(
        dses,
        staged_weights,
        10,
        key=key(),
        randomize_blocks=False,
        stop_strategy=StopStrategy.FIRST_STOP_STRATEGY,
    )

    assert mixture_ds.is_finite()
    assert await mixture_ds.async_len() == 5
    assert await mixture_ds.get_batch([0, 1, 2, 3, 4]) == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_mixture_dataset_stop_strategy_restart():
    mixture_ds = MixtureDataset(
        datasets(), weights(), block_size=10, key=key(), stop_strategy=StopStrategy.RESTART_STRATEGY
    )

    with pytest.raises(ValueError):
        await mixture_ds.async_len()


@pytest.mark.asyncio
async def test_mixture_dataset_simulated_data_size():
    weights = {"ds1": 1 / 3, "ds2": 1 / 3, "ds3": 1 / 3}
    mixture_ds = MixtureDataset(
        {name: dataset.slice_dataset(end_index=1) for name, dataset in datasets().items()},
        weights,
        block_size=10,
        key=key(),
        randomize_blocks=False,
        stop_strategy=StopStrategy.RESTART_STRATEGY,
    )
    for _ in range(10):
        batch = await mixture_ds.get_batch([0, 1, 2])
        assert len(batch) == 3
        assert all(item in [1, 10, 100] for item in batch)

    mixture_ds = MixtureDataset(
        {name: dataset.slice_dataset(end_index=2) for name, dataset in datasets().items()},
        weights,
        block_size=10,
        key=key(),
        randomize_blocks=False,
        stop_strategy=StopStrategy.RESTART_STRATEGY,
    )
    for _ in range(10):
        batch = await mixture_ds.get_batch([0, 1, 2])
        assert len(batch) == 3
        assert all(item in [1, 2, 10, 20, 100, 200] for item in batch)


@pytest.mark.asyncio
async def test_mixture_dataset_normalized_weights():
    weights = {"ds1": 0, "ds2": 0.5, "ds3": 0.5}
    mixture_ds = MixtureDataset(datasets(), weights, block_size=10, key=key(), randomize_blocks=False)

    batch = await mixture_ds.get_batch([0, 1, 2])
    assert len(batch) == 3
    assert all(item in [10, 20, 30, 100, 200, 300] for item in batch)


@pytest.mark.asyncio
async def test_mixture_dataset_unpermuted_ids():
    mixture_ds = MixtureDataset(datasets(), weights(), block_size=10, key=key())

    unpermuted_ids = mixture_ds._compute_unpermuted_ids(
        mixture_ds._compute_expected_counts_per_block(weights(), block_size())
    )
    assert len(unpermuted_ids) == 10
    assert unpermuted_ids[0] >> 32 in range(3)  # Ensure the dataset ID is valid


def test_expected_counts_hamilton_remainder():
    """Regression for the per-block apportionment bug caught in the C5-v4 audit,
    fixed 2026-06-15.

    Mirrors the EXACT C5-v4 phase 2 setup that triggered the bug: SlimPajama-NL
    in 228 row-proportional per-part components (128 small chunk1 parts at
    ~99,892 rows each + 100 large chunk2 parts at ~518,293 rows each), plus 4
    code/markup components at their intended 80/20-of-10% inner split. With
    block_size=2048 this gives a 189-sample remainder; the old "dump to
    argmax" behavior pooled all 189 into code_se_python (floor 88 → actual
    277), shifting the bucket totals from intended 90/8/2 NL/code/markup to
    actual 81/17/2 (caught in Dongwei review, 2026-06-15).

    Hamilton's method bounds per-component drift to ≤ 1 sample. Bucket totals
    end within 1 sample of intended.
    """
    # Row counts: chunk1 = 128 × 99,892 = 12,786,235; chunk2 = 100 × 518,293
    # = 51,829,342; total = 64,615,577. (Match shard_ledger.json for the
    # 2026-06-15 SP-NL en-filtered cache.)
    chunk1_rows_each = 99_892
    chunk2_rows_each = 518_293
    n_chunk1, n_chunk2 = 128, 100
    sp_total_rows = chunk1_rows_each * n_chunk1 + chunk2_rows_each * n_chunk2
    sp_share = 0.90
    sp_weights = {
        **{f"sp_chunk1_{i:03d}": sp_share * chunk1_rows_each / sp_total_rows for i in range(n_chunk1)},
        **{f"sp_chunk2_{i:03d}": sp_share * chunk2_rows_each / sp_total_rows for i in range(n_chunk2)},
    }
    # 10% code+markup at 80/20 inner split, code split per C5-v4's 8.8/7.3/0.2 ratio.
    code_total = 8.8 + 7.3 + 0.2
    bucket_weights = {
        "code_se_python": 0.10 * 0.80 * (8.8 / code_total),
        "code_nemotron_cc": 0.10 * 0.80 * (7.3 / code_total),
        "code_nemotron_ua": 0.10 * 0.80 * (0.2 / code_total),
        "markup_se_markdown": 0.10 * 0.20 * 1.0,
    }
    all_weights = {**sp_weights, **bucket_weights}
    dses = {name: ListAsyncDataset([0]) for name in all_weights}
    block_size = 2048
    md = MixtureDataset(dses, all_weights, block_size=block_size, key=key())

    counts = md._compute_expected_counts_per_block(all_weights, block_size)
    assert int(counts.sum()) == block_size

    # Per-component drift ≤ 1 sample (the Hamilton invariant).
    targets = np.array([all_weights[n] * block_size for n in all_weights])
    drift = np.abs(counts - targets)
    assert drift.max() < 1.0, (
        f"Hamilton per-component drift exceeds 1 sample (max={drift.max()}). "
        "This regresses to the dump-to-largest behavior."
    )

    # Bucket totals within 1 sample of intended (the audit-relevant property).
    names = list(all_weights.keys())
    sp_count = sum(int(counts[i]) for i, n in enumerate(names) if n.startswith("sp_"))
    code_count = sum(int(counts[i]) for i, n in enumerate(names) if n.startswith("code_"))
    markup_count = sum(int(counts[i]) for i, n in enumerate(names) if n.startswith("markup_"))
    sp_target = 0.90 * block_size
    code_target = 0.08 * block_size
    markup_target = 0.02 * block_size
    assert abs(sp_count - sp_target) < 2, f"SP-NL bucket drift too large: {sp_count} vs {sp_target}"
    assert abs(code_count - code_target) < 2, f"CODE bucket drift too large: {code_count} vs {code_target}"
    assert abs(markup_count - markup_target) < 2, f"MARKUP bucket drift too large: {markup_count} vs {markup_target}"

    # The exact bug-scenario witness: code_se_python must be near its intended
    # floor (~88), NOT the old dump-to-largest result (~277).
    se_python_idx = names.index("code_se_python")
    se_python_target = bucket_weights["code_se_python"] * block_size
    assert abs(int(counts[se_python_idx]) - se_python_target) < 1.0, (
        f"code_se_python count {counts[se_python_idx]} far from intended "
        f"{se_python_target:.2f}. Old 'dump-to-largest' gave 277 here — "
        "this regression test must catch that."
    )


@pytest.mark.asyncio
async def test_mixture_dataset_remap_indices():
    dses = datasets()
    mixture_ds = MixtureDataset(dses, weights(), block_size=10, key=key())

    remapped_indices = await mixture_ds._remap_indices(dses["ds1"], [0, 1, 2])
    assert len(remapped_indices) == 3
    assert remapped_indices == [0, 1, 2]

    # check wrap around
    len_ds1 = await dses["ds1"].async_len()
    remapped_indices = await mixture_ds._remap_indices(dses["ds1"], [len_ds1 - 1, len_ds1, len_ds1 + 1])
    assert len(remapped_indices) == 3

    assert remapped_indices == [len_ds1 - 1, 0, 1]


@pytest.mark.asyncio
async def test_mixture_dataset_restart_rejects_empty_finite_dataset():
    dses = {"empty": ListAsyncDataset([]), "full": ListAsyncDataset([10, 20, 30])}
    mixture_ds = MixtureDataset(
        dses,
        {"empty": 0.5, "full": 0.5},
        block_size=4,
        key=key(),
        randomize_blocks=False,
        stop_strategy=StopStrategy.RESTART_STRATEGY,
    )

    with pytest.raises(ValueError, match="empty finite dataset"):
        await mixture_ds.get_batch([0])


@pytest.mark.asyncio
async def test_mixture_dataset_respects_weights():
    w = weights()
    mixture_ds = MixtureDataset(datasets(), w, block_size(), key=key())

    # Check that the dataset respects the weights
    num_samples = 1000
    samples = await mixture_ds.get_batch(list(range(num_samples)))

    counts = {"ds1": 0, "ds2": 0, "ds3": 0}
    for sample in samples:
        if sample < 10:
            counts["ds1"] += 1
        elif sample < 100:
            counts["ds2"] += 1
        else:
            counts["ds3"] += 1

    for dataset, count in counts.items():
        assert abs(count / num_samples - w[dataset]) < 0.1, f"Dataset {dataset} has unexpected weight"


@pytest.mark.asyncio
async def test_mixture_dataset_randomizes_blocks():
    mixture_ds = MixtureDataset(datasets(), weights(), block_size=10, key=key())

    block_assignment_1 = mixture_ds._get_block(0)
    block_assignment_2 = mixture_ds._get_block(0)

    assert np.all(block_assignment_1 == block_assignment_2), "Block assignments should be randomized"

    block_assignment_3 = mixture_ds._get_block(1)
    assert not np.all(block_assignment_1 == block_assignment_3), "Block assignments should be randomized"


@pytest.mark.asyncio
async def test_mixture_dataset_samples_all_elements():
    mixture_ds = MixtureDataset(datasets(), weights(), block_size=10, key=key())

    num_samples = 1000
    samples = await mixture_ds.get_batch(list(range(num_samples)))

    assert len(samples) == num_samples
    assert set(samples) == {1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500}


def test_rescale_mixture_schedule_for_batch_schedule():
    mixture_schedule = [(0, {"ds1": 0.5, "ds2": 0.5}), (10, {"ds1": 0.2, "ds2": 0.8})]
    batch_schedule = BatchSchedule([ScheduleStep(start=0, value=10), ScheduleStep(start=5, value=20)])

    rescaled_schedule = rescale_mixture_schedule_for_batch_schedule(mixture_schedule, batch_schedule)

    expected_schedule = [(0, {"ds1": 0.5, "ds2": 0.5}), (150, {"ds1": 0.2, "ds2": 0.8})]
    assert rescaled_schedule == expected_schedule

    # double check changing on the cusp
    batch_schedule = BatchSchedule([ScheduleStep(start=0, value=10), ScheduleStep(start=10, value=20)])

    rescaled_schedule = rescale_mixture_schedule_for_batch_schedule(mixture_schedule, batch_schedule)

    expected_schedule = [(0, {"ds1": 0.5, "ds2": 0.5}), (100, {"ds1": 0.2, "ds2": 0.8})]

    assert rescaled_schedule == expected_schedule


# --- ConcatDataset tests ---


@pytest.mark.asyncio
async def test_concat_dataset_getitem_consistent_with_get_batch():
    ds1 = ListAsyncDataset([1, 2, 3])
    ds2 = ListAsyncDataset([10, 20])
    concat = ConcatDataset({"a": ds1, "b": ds2})
    batch = await concat.get_batch([0, 1, 2, 3, 4])
    for i in range(5):
        assert await concat.getitem_async(i) == batch[i]


@pytest.mark.asyncio
async def test_concat_with_permutation_is_a_permutation():
    """Every index maps to a unique element — no duplicates, no missing."""
    ds1 = ListAsyncDataset(list(range(50)))
    ds2 = ListAsyncDataset(list(range(50, 100)))
    concat = ConcatDataset({"a": ds1, "b": ds2})
    shuffled = PermutationDataset(concat, key=key())
    total = await shuffled.async_len()
    batch = await shuffled.get_batch(list(range(total)))
    assert sorted(batch) == list(range(100))


@pytest.mark.asyncio
async def test_concat_with_permutation_nests_in_mixture_dataset():
    """ConcatDataset + PermutationDataset can be used as a MixtureDataset component."""
    ds1 = ListAsyncDataset(list(range(10)))
    ds2 = ListAsyncDataset(list(range(10, 20)))
    concat = ConcatDataset({"a": ds1, "b": ds2})
    shuffled = PermutationDataset(concat, key=key())

    ds3 = ListAsyncDataset(list(range(100, 110)))
    mixture = MixtureDataset(
        {"flat": shuffled, "other": ds3},
        {"flat": 0.5, "other": 0.5},
        block_size=10,
        key=key(),
        randomize_blocks=False,
    )

    batch = await mixture.get_batch(list(range(20)))
    assert len(batch) == 20
    all_values = set(range(20)) | set(range(100, 110))
    assert all(item in all_values for item in batch)
