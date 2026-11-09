# UAMM-2: tiered, codec-aware KV-cache manager

Combines our UAMM-1 prototype had (PagedAttention-style blocks, H2O-style importance, tiering controller,
Verilog decision engine) with: KIVI-style KV quantization, entropy-coded cold storage, a utility-based
(tier, codec) planner with hysteresis, query-aware block-sparse attention, an RVV kernel for PULP Ara,
three new RTL blocks, and a runtime-adapter seam for a compiler-based runtime.

```
uamm2/            Python reference engine (real tensors, real codecs, real quality metric)
  config.py       all knobs + ASSUMED cost constants (replace with measurements)
  codecs.py       fp16 / int8 / kivi4 / kivi2 (+ zlib "z" variants), per-channel K, per-token V
  paged_cache.py  paged blocks holding encoded payloads; key min/max summaries; calibration at seal
  h2o.py          H2O-style decayed attention score -> block importance (normalized)
  attention.py    block-sparse attention (min/max key bound selection) + dense FP32 oracle
  planner.py      UtilityPlanner (UAMM-2) and ThresholdPlanner (original UAMM-1, for ablation)
  simulator.py    end-to-end loop; ablation.py; cli.py; workload.py (synthetic q/k/v)
  hwmodel.py      bit-exact golden models + test-vector generator for the RTL
  ara.py          ctypes binding + RVV compile check for ara/kv_kernels.c
  runtime.py      RuntimeAdapter, ReferenceAdapter (works), CompilerRuntimeAdapter (skeleton)
hw/rtl/           uamm_sweep_scorer.v, uamm_kv_codec.v (quant4/dequant4), uamm_block_table.v
hw/tb/            self-checking testbenches;  hw/Makefile
ara/              kv_kernels.{h,c}: fused QK on packed 4-bit keys, scalar + RVV 1.0
tests/            pytest suite (56 tests)
```

## Run it

```bash
pip install numpy pytest            # + `pip install verilator` for the RTL tests (needs g++ with C++20)
python -m uamm2 simulate --tokens 1024
python -m uamm2 ablate --tokens 1536
python -m pytest -q tests           # 56 tests, ~30 s
make -C hw test VERILATOR=verilator # RTL regression;  make -C hw lint
```

## Design

A placement is a **(tier, codec) pair**, not just a tier.

| Tier | Where | Allowed codecs |
|---|---|---|
| T0 | near-compute SRAM / VRF window | sink + recent window (protected, fp16, never moved); also hot blocks |
| T1 | DRAM | any non-entropy codec (fp16, int8, kivi4, kivi2) |
| T2 | DRAM | kivi4, kivi2 |
| T3 | flash | int8z, kivi4z, kivi2z (zlib) or raw |

Planner cost per candidate (lower is better), greedy in descending block heat under per-tier byte caps:

```
cost = p_access * (read_ms + decode_ms)  +  lambda * heat * rel_err(block, codec)  +  congestion(bytes, fill)
```

* `p_access` = EMA of how often attention actually fetched the block; `heat` = mean of that and H2O importance.
* `rel_err` is measured per block at seal time by really encoding/decoding it with every codec.
* Hysteresis: minimum residency + relative-improvement margin.
* **Precision is a one-way ratchet**: promoting a 4-bit block to a "fast tier" does not restore precision, so
  re-encoding may only lower it. (Same-precision changes, i.e. toggling zlib, are lossless blob transforms.)
* Attention reads only protected blocks + the top-`attn_block_budget` blocks ranked by
  `sum_c max(q_c*kmin_c, q_c*kmax_c)`, an upper bound on `q.k` over the block (tested as a true bound).
  Skipped blocks are never fetched or decoded; H2O scores come from the attention weights the step already computed.

## What changed relative to your UAMM-1 code (the issues I flagged earlier)

| UAMM-1 problem | UAMM-2 |
|---|---|
| `decision.evict` never used; data silently deleted when NVMe full | drop is explicit, opt-in (`allow_drop`), importance-gated; otherwise overflow is reported |
| importance clamped at 1.0 (H2O sums saturate it) | normalized by the 95th percentile of block scores |
| recency constant / unused | recency enters via `hit_ema` (selection frequency) |
| no cascade demotion, no hysteresis | greedy capacity-aware assignment + hysteresis |
| latency charged per token, migrations free | per block, per tier/codec; migration + re-encode cost accounted separately |
| O(N) Python loop per token in H2O | one vectorized decay per step |
| Verilog `/100` divider | integer weights 154/64/38 (sum 256), `>> 8` |
| Verilog/Python disagree on `protected` | RTL honours `protected` |
| 3-vector testbench, no boundaries | 17 sweeps incl. 173/174, 88/89, 242/243 pressure edges; 400 codec vectors; randomized table test |
| "SmoothQuant" listed as a KV solution | replaced by KV-specific KIVI-style codecs (SmoothQuant targets weights/activations) |

## What is verified, and how

* **Python**: 56 pytest tests, including property checks (bound is a true upper bound, pack/unpack round-trips,
  entropy variants lossless, per-channel keys beat per-token on outlier channels, precision ratchet instrumented,
  hysteresis reduces churn, capacities respected).
* **RTL** (Verilator 5.48, bit-exact vs `hwmodel.py`): sweep scorer 1088 checks, codec 400 vectors, block table
  6000 random ops. **Mutation-tested**: 12 injected bugs (changed weights/thresholds, removed rounding/clamp/saturation,
  dropped valid bit, swapped fields, accepted restart, removed `protected` guard) were all caught. One survived at first (the
  quantizer clamp) because no vector exercised it; I added vectors and it is now caught.
  Lint clean with `-Wall`. Yosys `synth` + `check -assert` pass, no latches inferred.
* **Ara kernel**: scalar path executed on the host and matches NumPy dequantize-then-dot to ~1e-5.
  The RVV path **compiles** for `riscv64 -march=rv64gcv` and lowers to real vector instructions
  (`vle8`, `vzext.vf4`, `vfcvt`, `vfmacc` with tail-undisturbed, `vfredusum`).

## What is NOT verified (read this before trusting any number)

* **The q/k/v stream is synthetic** (`workload.py`); quality numbers are attention-output error against a dense FP32
  oracle on that stream. `attn_rel_l2` is large partly because random V averages to a small-norm output; compare variants to each
  other, not to an absolute bar.
* **All latency/bandwidth/decode-cost constants are assumptions** (`config.py`). `fetch_ms` is a model output, not a measurement.
* Hardware: not placed-and-routed, no timing or area numbers. The 16 parallel 8x16 multipliers in the quantizer are area-hungry;
  a shift-based reciprocal would be cheaper. The block-table and scorer SRAMs are generic arrays that FPGA tools should map to BRAM.
* The planner is a greedy heuristic, not an optimal knapsack. `lambda_quality` was tuned on synthetic seeds 7-9; results below
  are on held-out seeds 11-13.
* Not implemented: prefetch, multi-sequence/multi-layer/multi-head layout (one flattened K/V vector per token), copy-on-write
  prefix sharing, real flash I/O.

## Ablation (1536 tokens; mean of held-out seeds 11-13; synthetic workload, assumed latencies)

```
variant                     resident/fp16  dropped_tok  rel_l2_mean  mass_recall  fetch_ms*  overflow_B
v1_threshold_fp16                  0.368        970.7        0.506       0.750       0.919           0
v2_planner_fp16                    1.000          0          0.081       0.945       1.04       256700   <- infeasible in budget
v2_planner_int8_kivi4              0.372          0          0.116       0.945       0.115           0
v2_planner_all_codecs              0.310          0          0.159       0.945       0.111           0
```

Reading it: in this setup, the fixed 60/25/15 threshold policy has to throw away ~63% of the tokens to fit the budget, losing most of
the attention mass. FP16-only cannot fit at all. Adding INT8/4-bit codecs fits everything with modest error; adding 2-bit saves
another ~17% of memory for ~40% more error, a knob (`lambda_quality`, `codecs`) rather than a free win.
Full table: `docs/ablation_results.txt`.
