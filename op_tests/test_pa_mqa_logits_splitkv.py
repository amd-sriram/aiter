#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``SplitKV`` only partitions work in ``deepgemm_fp8_paged_mqa_logits``.

The host heuristic turns ``TotalCuCount`` into ``SplitKV``, which decides how many
CTAs share one (sequence, next_n) tile. Each CTA writes a disjoint slice of the
context, so the logits must not depend on that choice at all -- every element is
produced by exactly one CTA, with no cross-split reduction to reassociate.

That invariant is what makes the grid a free tuning knob, so it is worth pinning
down: a future retune of the heuristic is only safe if changing ``SplitKV`` cannot
change the result. The shapes below cover both sides of the current heuristic's
branch (``TileQCount`` dividing ``WavePerEU * TotalCuCount`` or not) and both a
single split per tile and heavy oversplitting.
"""

import pytest
import torch
from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

from aiter import dtypes

dev = "cuda"
SEED = 1234
HEADS = 64
HEAD_DIM = 128
BLOCK_SIZE = 64
CHUNK_K = 256
SENTINEL = 12345.0

# Spans SplitKV from 1 (no split) to 130 (far more splits than context chunks, so
# most CTAs exit immediately) via the TotalCuCount knob.
TOTAL_CU_COUNTS = [16, 64, 256, 512, 1024, 2048]


def _make_inputs(batch_size, next_n, context_len):
    torch.manual_seed(SEED)
    max_block_len = (context_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = batch_size * max_block_len

    q_fp8 = torch.randint(
        1, 64, (batch_size, next_n, HEADS, HEAD_DIM), dtype=torch.uint8, device=dev
    ).view(dtypes.fp8)

    kv_bits = torch.randint(
        1, 64, (num_blocks, BLOCK_SIZE, 1, HEAD_DIM + 4), dtype=torch.uint8, device=dev
    )
    # the 4 fp8 bytes after each packed token are read as one fp32 scale
    kv_bits[..., HEAD_DIM:] = torch.tensor(
        [0, 0, 128, 63], dtype=torch.uint8, device=dev
    )
    kv_cache = kv_bits.view(dtypes.fp8)

    weights = torch.rand((batch_size * next_n, HEADS), dtype=torch.float32, device=dev)
    context_lens = torch.full((batch_size,), context_len, dtype=torch.int32, device=dev)
    # distinct blocks per sequence, shuffled, so a wrong split cannot alias its way
    # to the right answer
    block_tables = (
        torch.randperm(num_blocks, device=dev)
        .to(torch.int32)
        .view(batch_size, max_block_len)
    )
    return q_fp8, kv_cache, weights, context_lens, block_tables


def _run(inputs, context_len, total_cu_count):
    q_fp8, kv_cache, weights, context_lens, block_tables = inputs
    out = torch.full(
        (weights.shape[0], context_len), SENTINEL, dtype=torch.float32, device=dev
    )
    deepgemm_fp8_paged_mqa_logits(
        q_fp8,
        kv_cache,
        weights,
        out,
        context_lens,
        block_tables,
        context_len,
        Preshuffle=True,
        KVBlockSize=BLOCK_SIZE,
        ChunkK=CHUNK_K,
        TotalCuCount=total_cu_count,
        WavePerEU=2,
    )
    torch.cuda.synchronize()
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm GPU")
@pytest.mark.parametrize(
    "batch_size, next_n",
    [
        (16, 1),  # TileQCount divides WavePerEU * TotalCuCount
        (64, 1),
        (12, 1),  # ... and does not
        (5, 4),
    ],
)
@pytest.mark.parametrize("context_len", [1024, 8192])
def test_logits_independent_of_splitkv(batch_size, next_n, context_len):
    inputs = _make_inputs(batch_size, next_n, context_len)

    ref = _run(inputs, context_len, TOTAL_CU_COUNTS[0])
    assert not bool(
        (ref == SENTINEL).any()
    ), "reference left part of the output unwritten"

    for total_cu_count in TOTAL_CU_COUNTS[1:]:
        got = _run(inputs, context_len, total_cu_count)
        unwritten = int((got == SENTINEL).sum())
        assert unwritten == 0, (
            f"TotalCuCount={total_cu_count} left {unwritten} elements unwritten; "
            "the splits do not cover the whole context"
        )
        assert torch.equal(got, ref), (
            f"TotalCuCount={total_cu_count} changed the logits "
            f"(max abs diff {(got - ref).abs().max().item():.3e}); SplitKV must only "
            "partition work"
        )


if __name__ == "__main__":

    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
