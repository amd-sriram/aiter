#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression test: ``deepgemm_fp8_paged_mqa_logits`` must never store a NaN.

Background
----------
The sparse-attention top-k that consumes these logits (vLLM's
``top_k_per_row_decode``) builds a histogram over the raw float patterns and
then rescatters the candidates with float comparisons. A NaN compares false in
the scatter pass but was counted in the histogram pass, so some of the
shared-memory output slots are left unwritten; the uninitialized values are
copied into the global top-k index buffer, and sparse decode then uses them to
address the block table -> out-of-bounds KV read -> GPU memory access fault
(vllm-project/vllm#49714).

vLLM worked around this by scrubbing the whole ``[rows, max_model_len]`` fp32
logits workspace with ``nan_to_num_`` after every call, which costs a full
read-modify-write pass over a buffer that is two orders of magnitude wider than
the part top-k reads. The producer can uphold the same contract for free by
folding non-finite accumulations into ``-inf`` before the store.

NaN is injected through ``weights``, the only kernel input that reaches the
stored logits without passing through the ``max(o, 0)`` relu (which returns the
non-NaN operand and therefore hides NaNs coming from the KV cache). A NaN weight
on any head poisons that row's whole reduction, which is exactly the failure
signature reported upstream: a fully non-finite row inside the valid range.
"""

import pytest
import torch
from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

from aiter import dtypes

dev = "cuda"
SEED = 1234
HEADS = 128
HEAD_DIM = 128
NEXT_N = 1
BATCH_SIZE = 4
CONTEXT_LEN = 1024
MAX_MODEL_LEN = 2048
# the row whose weights are poisoned; the others must stay untouched
POISONED_ROW = 1


def _make_inputs(block_size):
    torch.manual_seed(SEED)
    q_bits = torch.randint(
        1, 64, (BATCH_SIZE, NEXT_N, HEADS, HEAD_DIM), dtype=torch.uint8, device=dev
    )
    q_fp8 = q_bits.view(dtypes.fp8)

    # The kernel reads a block as [block_size * HEAD_DIM fp8 values | block_size
    # fp32 scales], so build that flat layout and only reshape for the wrapper.
    num_blocks = (CONTEXT_LEN + block_size - 1) // block_size
    flat = torch.randint(
        1,
        64,
        (num_blocks, block_size * (HEAD_DIM + 4)),
        dtype=torch.uint8,
        device=dev,
    )
    flat[:, block_size * HEAD_DIM :].view(torch.float32).fill_(1.0)
    kv_cache = flat.view(num_blocks, block_size, 1, HEAD_DIM + 4).view(dtypes.fp8)

    weights = torch.ones((BATCH_SIZE * NEXT_N, HEADS), dtype=torch.float32, device=dev)
    weights[POISONED_ROW, 0] = float("nan")

    context_lens = torch.full((BATCH_SIZE,), CONTEXT_LEN, dtype=torch.int32, device=dev)
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device=dev).repeat(
        BATCH_SIZE, 1
    )
    return q_fp8, kv_cache, weights, context_lens, block_tables


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm GPU")
@pytest.mark.parametrize("block_size", [64, 256])
def test_paged_mqa_logits_never_writes_nan(block_size):
    q, kv, w, ctx_lens, block_tables = _make_inputs(block_size)

    rows = BATCH_SIZE * NEXT_N
    out = torch.zeros((rows, MAX_MODEL_LEN), dtype=torch.float32, device=dev)
    deepgemm_fp8_paged_mqa_logits(
        q,
        kv,
        w,
        out,
        ctx_lens,
        block_tables,
        MAX_MODEL_LEN,
        Preshuffle=True,
        KVBlockSize=block_size,
        ChunkK=256,
        WavePerEU=2,
    )
    torch.cuda.synchronize()

    valid = out[:, :CONTEXT_LEN]
    nan_count = int(torch.isnan(valid).sum())
    assert nan_count == 0, (
        f"{nan_count} NaN logits stored in the valid range; top-k requires "
        "orderable values, so the producer must emit -inf instead."
    )

    # the poisoned row degrades to unselectable padding, not to NaN
    poisoned = valid[POISONED_ROW]
    assert bool(
        (poisoned == float("-inf")).all()
    ), "poisoned row should be all -inf, got a mix of values"

    # every other row keeps real logits
    others = torch.cat([valid[:POISONED_ROW], valid[POISONED_ROW + 1 :]])
    assert bool(torch.isfinite(others).all()), "unpoisoned rows must stay finite"


if __name__ == "__main__":

    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
