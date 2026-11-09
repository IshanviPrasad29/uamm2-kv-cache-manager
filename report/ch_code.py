from data import *
def build(R):
    R.h1("6. Project Structure and Code Walkthrough")
    R.p("This chapter explains how the project is organised and what each part of the code does, so that a reader can find any behaviour described in this report.", align="j")
    R.h2("6.1 Directory layout")
    R.code("""uamm2_project/
  README.md  requirements.txt
  uamm2/                 Python reference engine
    config.py              all parameters and the assumed cost constants
    codecs.py              quantisation, packing, entropy coding, calibration
    paged_cache.py         blocks, sealing, re-encoding, dropping
    tiers.py               tier usage and the latency model
    h2o.py                 attention-based importance
    attention.py           block selection, sparse attention, dense oracle
    planner.py             utility planner and the original threshold planner
    simulator.py           decode loop, metrics, per-step trace
    workload.py            synthetic query/key/value generator
    ablation.py cli.py __main__.py     variants and command line
    hwmodel.py             bit-exact models of the RTL + test-vector writer
    ara.py                 host build of the C kernel; RISC-V compile check
    runtime.py             runtime adapter interface, reference adapter, skeleton
  hw/
    rtl/                   uamm_sweep_scorer.v  uamm_kv_codec.v  uamm_block_table.v
    tb/                    self-checking testbenches (+trace writes VCD files)
    vectors/               generated stimulus and expected outputs (hex)
    Makefile               test, lint, vectors, clean
  ara/                     kv_kernels.c / .h : scalar and RISC-V vector kernel
  tests/                   56 pytest tests
  experiments/             run_experiments.py, mutation_test.py, figure scripts, results/
  docs/                    waveform dumps, synthesis reports, captured tool output
  report/                  scripts that build this document""", size=8)
    R.h2("6.2 Python engine, module by module")
    rows = [
        ["`config.py`", "`Config`, `TierSpec`", "Holds geometry (64 channels, 16-token blocks), protection (4 sinks, 32 recent), planner settings (interval 8, hysteresis 0.25, residency 16, λ = 2), codec set and the assumed latency, bandwidth and decode constants. Validates inputs on creation."],
        ["`codecs.py`", "`encode`, `decode`, `calibrate`, `pack_bits`, `parse_codec`", "Turns a block of keys and values into a byte payload: asymmetric quantisation (keys per channel, values per token in groups of 32), bit packing, optional zlib. `calibrate` encodes a block with every codec and returns its error and size tables. Codec names: fp16, int8, kivi4, kivi2 and their `z` forms."],
        ["`paged_cache.py`", "`KVBlock`, `PagedKVCache`", "`append_token` fills the open block; at 16 tokens `_seal` rounds to FP16, stores the payload, computes key min/max and calibrates. `reencode` changes codec (lossless for the entropy wrapper, lossy otherwise); `drop` removes a block."],
        ["`tiers.py`", "`usage`, `read_cost_ms`, `move_cost_ms`, `decode_cost_ms`", "Accounts bytes per tier and computes the modelled read, decode and migration costs. No data actually moves."],
        ["`h2o.py`", "`H2OImportance`", "`update` decays all token scores and adds the attention weight each token received; `refresh` aggregates to blocks (mean, max or sum) and normalises by the 95th percentile."],
        ["`attention.py`", "`select_blocks`, `sparse_attention`, `dense_oracle`", "Ranks candidate blocks by the key-bound formula and returns the top budget; computes attention over the read blocks; the oracle computes full FP32 attention for the error metrics."],
        ["`planner.py`", "`UtilityPlanner`, `ThresholdPlanner`, `codec_allowed`", "The utility planner enumerates allowed (tier, codec) pairs, evaluates the cost, visits blocks by heat, applies capacity and hysteresis and returns a plan. The threshold planner reproduces the original policy for comparison."],
        ["`simulator.py`", "`Simulator.run`", "For every token: generate q/k/v, append, update protection, select blocks, run sparse and dense attention, record metrics, update hit rates and importance, and every 8 steps plan and apply. Writes a per-step trace."],
        ["`workload.py`", "`SyntheticWorkload`", "Generates sinks, outlier channels, topic phases and local recency (Chapter 7)."],
        ["`ablation.py`, `cli.py`", "`variants`, `main`", "Defines the ablation variants; commands `simulate`, `ablate`, `hwvectors`."],
        ["`hwmodel.py`", "`scorer_model`, `quant4_model`, `dequant4_model`, `write_vectors`", "Integer-exact copies of the RTL arithmetic and the generator of boundary and random vectors that the testbenches read."],
        ["`ara.py`", "`AraKernels`, `rvv_compile_check`", "Builds the C kernel for the host and calls it through ctypes to score packed 4-bit keys; compiles the vector path for a RISC-V target."],
        ["`runtime.py`", "`RuntimeAdapter`, `ReferenceAdapter`", "Interface to an inference runtime (write token, read block, migrate block, report attention); a working reference implementation and a skeleton for a compiler-based runtime."]]
    R.table(["File", "Main items", "What it does"], rows, caption="Python modules.", widths=[1.2, 1.7, 3.7], size=8)
    R.h2("6.3 Worked examples of the main routines")
    R.h3("6.3.1 Sealing and calibration (`paged_cache.py`, `codecs.py`)")
    R.p("When the sixteenth token of a block arrives, `_seal` runs. (1) The staged K and V rows are stacked and rounded to FP16; this rounded copy is the reference from then on. (2) `kmin` and `kmax` are taken over the 16 key rows. (3) `calibrate` encodes the block with each enabled codec (fp16, int8, kivi4, kivi2 and the `z` forms), decodes it again and records the relative squared error ‖x − x̂‖² / ‖x‖² and the byte size per codec. (4) The block is stored as an FP16 payload in tier 0. For a typical 16 × 64 block the size table reads about 4,096 B (fp16), 2,432 B (int8), 1,408 B (kivi4) and 896 B (kivi2), and the error table about 4 × 10⁻⁸, 2 × 10⁻⁵, 5 × 10⁻³ and 0.13.", align="j")
    R.h3("6.3.2 Selecting blocks (`attention.py`)")
    R.p("For a query q and a candidate block with key minimum kmin and maximum kmax, `select_blocks` computes bound(b) = Σ_c max(q_c·kmin_c, q_c·kmax_c). Suppose d = 2, q = (2, −1), and a block with kmin = (0, 1), kmax = (3, 4). The bound is max(0, 6) + max(−1, −4) = 6 − 1 = 5; every key inside the box [0,3] × [1,4] has q·k ≤ 5. The candidates are ranked by the bound and the best `attn_block_budget` are read; the protected blocks are always added.", align="j")
    R.h3("6.3.3 Updating importance (`h2o.py`)")
    R.p("After each step the weights w_i of the tokens that were read are available from the softmax. `update` multiplies all token scores by the decay (0.95) and adds w_i to the tokens read. Every plan interval `refresh` computes the mean token score per block, divides by the 95th percentile over blocks and clips to [0, 1]. A block that is not read decays: after 20 steps its score is 0.95²⁰ ≈ 0.36 of its former value.", align="j")
    R.h3("6.3.4 One planning decision (`planner.py`)")
    R.p("Consider a block with hit_ema 0.30 and importance 0.50, so heat = 0.40, with tables err(kivi4) = 0.005, err(kivi2) = 0.13, size(kivi4) = 1,408 B and size(kivi2) = 896 B. With λ = 2 the quality terms are 2 × 0.40 × 0.005 = 0.004 ms-equivalent and 2 × 0.40 × 0.13 = 0.104. The latency terms are the access probability times the read cost. Reading the 4-bit block from DRAM costs 0.002 ms latency + 1,408 B / 12.8 GB/s (0.0001 ms) + decode 0.0002 ms ≈ 0.0023 ms, times 0.30 = 0.0007; from flash it costs about 0.1 + 0.0014 + 0.0002 ≈ 0.1016 ms, times 0.30 = 0.030. Total (DRAM, kivi4) ≈ 0.0047; (flash, kivi2) ≈ 0.030 + 0.104 = 0.134. The planner therefore prefers DRAM at 4 bits, unless DRAM is full, in which case the congestion term and the capacity check push the block to the next feasible pair. Finally the hysteresis rule applies: if the block already sits in a pair whose cost is within 25 % of the best, or moved fewer than 16 steps ago, it stays.", align="j")
    R.h3("6.3.5 Decode loop (`simulator.py`)")
    R.numbered(["Generate q, k, v for token t and append the token to the cache (a block is sealed on the 16th token).",
                "Mark protected blocks: those holding the first four tokens, the last 32 tokens, or the open block.",
                "Rank the other sealed blocks by the key bound and select the budget; read protected plus selected blocks and compute sparse attention.",
                "Compute the dense FP32 attention over all tokens (the oracle) and record error, recall and modelled fetch time.",
                "Update each selected block's hit rate, and the token scores with the step's attention weights.",
                "Every eighth step refresh importance, run the planner, apply the plan (re-encode and move) and record tier bytes, codec counts and counters."])
    R.h2("6.4 Hardware")
    R.table(["File", "What it does"],
            [["`uamm_sweep_scorer.v`", "Metadata SRAM (25-bit words) and a 2-stage pipeline. A `start` pulse latches the pressure value and streams all blocks through `score = (154·imp + 64·rec + 38·frq) >> 8`, the tier decision (protected override, thresholds 174 and 89) and the eviction flag (pressure ≥ 243, score < 89, not protected)."],
             ["`uamm_kv_codec.v`", "`uamm_kv_quant4`: 16 lanes of subtract, multiply by the Q4.12 reciprocal, round, clamp to 15, pack nibbles. `uamm_kv_dequant4`: multiply by the Q8.8 step, round, add zero point, saturate at 255."],
             ["`uamm_block_table.v`", "256-entry table of `{tier, codec, phys}` with valid bits and a one-cycle lookup returning `hit`."],
             ["`tb_*.v`", "Read the generated vectors, drive the module, compare every output and print PASS or FAIL; with `+trace` they write a VCD."],
             ["`Makefile`", "`make test` regenerates vectors and runs the three benches in Verilator; `make lint` runs `-Wall` lint."]],
            caption="Hardware files.", widths=[1.8, 4.8], size=8)
    R.h2("6.5 Vector kernel")
    R.p("`kv_qk_prepare` folds the per-channel scale into the query (split into even and odd channels) and returns the bias Σ q·zero. `kv_qk_dot_kivi4_scalar` loops over tokens and packed bytes with two multiply–adds per byte. `kv_qk_dot_kivi4_rvv` does the same with vector instructions (byte load, nibble extraction, widen, convert, fused multiply–add with tail-undisturbed policy, reduction). `kv_qk_dot_kivi4` selects the vector path when compiled for a vector target.", align="j")
    R.h2("6.6 Tests")
    R.table(["File", "Tests", "What is checked"],
            [["`test_codecs.py`", "10", "Pack/unpack round trip and nibble order; FP16 exactness; error and size ordering across codecs; entropy variants lossless; per-channel keys beat per-token on outlier channels; constant input; bad inputs"],
             ["`test_cache_h2o_attention.py`", "8", "Block sealing and partial tail; ratchet and lossless entropy switch; the selection bound is a true upper bound; sparse equals dense when everything is read; heavy hitters rank higher; decay forgets"],
             ["`test_system.py`", "12", "Combined system fits without data loss; threshold policy must drop; combined system beats it on quality here; capacities respected; protected blocks stay hot and lossless; precision never increases; hysteresis lowers churn; drop only when enabled; adapter round trip; skeleton honesty"],
             ["`test_ara_hw.py`", "6", "C kernel against NumPy; rejects other codecs; vector path compiles; golden scorer properties; codec error bounded; full RTL regression in Verilator"]],
            caption="Test files.", widths=[2.0, 0.6, 4.0], size=8)
    R.h2("6.7 How to run everything")
    R.code("""pip install numpy pytest verilator        # g++ with C++20 needed for the RTL tests
python -m uamm2 simulate --tokens 1024     # one configuration
python -m uamm2 ablate --tokens 1536       # UAMM-1 vs UAMM-2 variants
python -m pytest -q tests                  # 56 tests, ~40 s
make -C hw test ; make -C hw lint          # RTL regression and lint
python experiments/run_experiments.py      # all studies -> experiments/results/*.json
python experiments/mutation_test.py        # RTL mutation testing
gtkwave docs/wave/trace_sweep_scorer.vcd   # view a waveform""", size=8)
