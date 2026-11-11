# Unified Adaptive Memory Manager (UAMM) for KV-Cache Management on Edge Processors

Term paper project, ELL305. Authors: Ishanvi Prasad (2024EE10406) and Sweety (2024MT10535).
The full write-up is in [`report/UAMM2_Report.pdf`](report/UAMM2_Report.pdf): problem description, our approach (UAMM-1, its gaps, UAMM-2, comparison, problems faced), literature review, implementation, results, remaining problems, timeline, and Git setup.

This repository contains two generations of one idea: **UAMM-1**, our first prototype, and **UAMM-2**, the redesign
that fixes what UAMM-1 could not do.

---

## 1. The problem

Large language models are hard to run on edge processors because of memory. Every generated token adds a key and a value
vector to the **KV cache** for every layer and head. The cache grows with the context length, has to be read in full at each
decoding step, and shares a few gigabytes of memory with the model weights. On a small device it quickly becomes the
bottleneck.

Published work attacks this from different directions: paging the cache in fixed blocks (PagedAttention), evicting
unimportant tokens (H2O and related methods), quantising keys and values to fewer bits (KIVI and related methods),
selecting only the blocks a query needs (Quest), and offloading to slower memory. Each answers only part of the problem,
and they are rarely combined in one system.

## 2. UAMM-1: our first prototype

UAMM-1 put three ideas into one memory manager:

* **Paged blocks** of 16 tokens as the unit of storage, in the style of PagedAttention.
* **H2O-style importance**: a token that has received much attention is kept.
* **A three-tier controller** (fast, medium, slow memory) that scores each block as
  `0.60 * importance + 0.25 * recency + 0.15 * frequency` and compares the score with two fixed thresholds.
  A small Verilog module made the same decision in hardware.

UAMM-1 worked as a demonstration but had real limits. Placement was by fixed thresholds with no notion of a memory
budget, and every block stayed in FP16, so when memory ran out the only option was to delete blocks. Data could be
dropped silently, the importance value saturated at 1.0, migrations were free in the cost model, and the Verilog
used a divider and ignored its `protected` input. Chapters 4 and 5 of the report analyse these defects.

## 3. UAMM-2: the redesign

The central change: **a block's placement is a pair, (tier, codec)**, chosen together.

* Cold blocks are not deleted. They are stored at lower precision (8-bit, 4-bit or 2-bit) and in a slower tier.
* A **utility planner** chooses the pair for every block from its heat, its measured quantisation error and the cost of
  the memory it occupies, under per-tier byte limits.
* The first 4 tokens (attention sinks) and the most recent 32 tokens are protected and stay in FP16.
* **Precision only goes down** (a one-way ratchet), and permanent deletion is an explicit opt-in.
* **Query-aware attention** reads only the blocks that can matter for the current query, using per-block minimum and
  maximum key bounds.
* Three Verilog blocks and a vector-processor kernel cover the inner loops (see section 5).

---

## 4. Repository layout

```
uamm2/            Python engine (real tensors, real codecs, real quality metric)
  config.py       all settings and the ASSUMED cost constants (replace with measurements)
  codecs.py       fp16 / int8 / kivi4 / kivi2 (+ zlib "z" variants), per-channel keys, per-token values
  paged_cache.py  paged blocks holding encoded payloads; key min/max summaries; calibration when a block is sealed
  h2o.py          decayed attention score aggregated to block importance (normalised)
  attention.py    block-sparse attention (min/max key bound) and a dense FP32 reference
  planner.py      UtilityPlanner (UAMM-2) and ThresholdPlanner (the UAMM-1 policy, kept for comparison)
  simulator.py    end-to-end loop;  ablation.py;  cli.py;  workload.py (synthetic queries, keys, values)
  hwmodel.py      bit-exact golden models and test-vector generator for the Verilog
  ara.py          Python binding and vector-compile check for ara/kv_kernels.c
  runtime.py      RuntimeAdapter, ReferenceAdapter (works), CompilerRuntimeAdapter (skeleton only)
uamm1_original/   the original UAMM-1 Python and Verilog code, as listings
hw/rtl/           uamm_sweep_scorer.v, uamm_kv_codec.v (quant4 / dequant4), uamm_block_table.v
hw/tb/            self-checking testbenches;  hw/Makefile
ara/              kv_kernels.{h,c}: fused query-key product on packed 4-bit keys, scalar and RISC-V vector versions
experiments/      experiment scripts (E1 to E10), mutation test, figure scripts (make_figures.py, make_report_figs.py), raw results in experiments/results/
docs/             waveform dumps (wave/), GTKWave views (gtkwave/), synthesis reports (synth/), figures, captures
tests/            pytest suite (59 tests)
report/           the final report (PDF)
```

### Run it

```bash
pip install numpy pytest                 # also: pip install verilator (needs g++ with C++20) for the Verilog tests
python3 -m uamm2 simulate --tokens 1024
python3 -m uamm2 ablate --tokens 1536
python3 -m pytest -q tests               # 59 tests
make -C hw test VERILATOR=verilator      # Verilog regression;  make -C hw lint
python3 experiments/run_experiments.py   # all studies, results in experiments/results/*.json
bash docs/gtkwave/view_waves.sh 3        # open a waveform in GTKWave with signals preloaded (1 to 6)
```

## 5. How UAMM-2 works

### Placement

| Tier | Where (modelled) | Allowed codecs |
|---|---|---|
| T0 | near-compute SRAM / vector register file | protected sink and recent window (FP16, never moved); also hot blocks |
| T1 | DRAM | any non-entropy codec: fp16, int8, kivi4, kivi2 |
| T2 | DRAM | kivi4, kivi2 |
| T3 | flash | int8z, kivi4z, kivi2z (zlib-compressed) or raw |

The planner assigns blocks greedily from hottest to coldest. Cost per candidate (lower is better):

```
cost = p_access * (read_ms + decode_ms)  +  lambda * heat * rel_err(block, codec)  +  congestion(bytes, fill)
```

* `p_access` is a running average of how often attention actually fetched the block; `heat` is the mean of that and the
  H2O importance.
* `rel_err` is measured for each block when it is sealed, by encoding and decoding it with every codec.
* Hysteresis (a minimum residency and a required relative improvement) stops blocks from moving back and forth.
* Re-encoding may only lower precision. Changing only the zlib layer is a lossless change of representation.

### Attention

Attention reads the protected blocks plus the top `attn_block_budget` blocks ranked by
`sum_c max(q_c * kmin_c, q_c * kmax_c)`, an upper bound on `q . k` over the block (tested as a true bound). Skipped blocks
are never fetched or decoded. The H2O scores come from the attention weights that the step already computed.

### Hardware and vector kernel

* `uamm_sweep_scorer.v`: scores all blocks with integer weights 154/64/38 (sum 256, so `>> 8` replaces a divider).
* `uamm_kv_codec.v`: 4-bit quantiser and dequantiser, 16 lanes, using a precomputed reciprocal (multiply and shift, no divider).
* `uamm_block_table.v`: 256-entry table mapping a logical block to `{tier, codec, physical slot}`.
* `ara/kv_kernels.c`: computes the query-key product directly on packed 4-bit keys, so no FP16 key block is built in memory.

## 6. What changed from UAMM-1 to UAMM-2

| UAMM-1 problem | UAMM-2 |
|---|---|
| `decision.evict` never used; data silently deleted when the slowest tier was full | dropping is explicit and opt-in (`allow_drop`), importance-gated; otherwise overflow is reported |
| importance clamped at 1.0 (H2O sums saturate it) | normalised by the 95th percentile of block scores |
| recency constant or unused | recency enters through the running hit average (selection frequency) |
| no cascade demotion, no hysteresis | greedy capacity-aware assignment with hysteresis |
| latency charged per token, migrations free | charged per block, per tier and codec; migration and re-encode cost counted separately |
| O(N) Python loop per token in H2O | one vectorised decay per step |
| Verilog `/100` divider | integer weights 154/64/38 and `>> 8` |
| Verilog and Python disagreed on `protected` | the Verilog honours `protected` |
| 3-vector testbench, no boundary cases | 17 sweeps including the 173/174, 88/89 and 242/243 edges; 400 codec vectors; randomised table test |
| a weight-and-activation method listed as a KV solution | replaced by KV-specific KIVI-style codecs |

## 7. What is verified, and how

* **Python:** 59 pytest tests, including property checks (the selection bound is a true upper bound, pack and unpack
  round-trip, entropy variants are lossless, per-channel keys beat per-token keys on outlier channels, the precision
  ratchet, hysteresis reduces churn, capacities are respected).
* **Verilog** (Verilator, bit-exact against `hwmodel.py`): sweep scorer 1088 checks, codec 400 vectors, block table
  6000 random operations. **Mutation-tested:** 12 injected bugs were all caught. One survived at first (the quantiser clamp)
  because no vector exercised it; we added vectors and it is now caught. Lint clean with `-Wall`. Yosys synthesis and
  `check -assert` pass with no latches.
* **Vector kernel:** the scalar path runs on the host and matches NumPy dequantise-then-dot to about 1e-5. The vector path
  compiles for `riscv64 -march=rv64gcv` and lowers to real vector instructions (`vle8`, `vzext.vf4`, `vfcvt`, `vfmacc`,
  `vfredusum`).

## 8. What is NOT verified (read before trusting any number)

* **The query, key and value stream is synthetic** (`workload.py`). Quality numbers are attention-output error against a dense
  FP32 reference on that stream. `attn_rel_l2` is large partly because random values average to a small-norm output, so
  compare variants with each other rather than with an absolute bar. No real language model and no tokenizer was run.
* **All latency, bandwidth and decode-cost constants are assumptions** (`config.py`). `fetch_ms` is a model output, not a
  measurement. The tiers (SRAM, DRAM, flash) are byte-accounting structures in ordinary RAM.
* **Nothing was run on the vector processor or on any other target hardware.** The vector kernel was only compiled.
* **Hardware:** not placed-and-routed, no timing or area numbers. The 16 parallel 8x16 multipliers in the quantiser are
  area-hungry; a shift-based reciprocal would be cheaper.
* The planner is a greedy heuristic, not an optimal knapsack. `lambda_quality` was tuned on synthetic seeds 7 to 9; the
  results below are on held-out seeds 11 to 13.
* **Not implemented:** prefetch, multi-sequence, multi-layer and multi-head layout (one flattened key and value vector per
  token), copy-on-write prefix sharing, real flash I/O, and the future scope in Chapter 7 of the report (those are
  design ideas, not code).

## 9. Ablation (1536 tokens; synthetic workload, assumed latencies; see `docs/ablation_results.txt` for the full table)

| variant | resident / FP16 | + key summaries | dropped tokens | rel. L2 error | fetch ms (model) |
|---|---|---|---|---|---|
| UAMM-1 threshold, FP16 | 0.37 | | 973 | 0.479 | 0.92 |
| No planner, every block 4-bit | 0.364 | | 0 | 0.152 | 0.068 |
| Planner, INT8 + KIVI-4 | 0.371 | | 0 | 0.122 | 0.116 |
| Planner, all codecs | 0.310 | 0.435 | 0 | 0.166 | 0.112 |
| Planner, FP16 only | does not fit (258 KB overflow) | | 0 | 0.092 | 1.04 |

Honest reading: a joint (tier, codec) choice lets 1,536 tokens fit a 168 KB budget that FP16 cannot, with no deleted tokens.
A simple all-4-bit baseline does that too and is faster in the model. The planner gives lower error at equal memory, lower
memory (with 2-bit) and feasibility in tighter budgets, not speed. The advantage depends on the budget being much smaller
than the data (at 3x the budget FP16 fits and wins). The key min/max summaries are not charged in the main runs; when they
are (E9), INT8 + KIVI-4 and all-4-bit overflow. Section 5.3 of the report lists every correction made to the first draft.
