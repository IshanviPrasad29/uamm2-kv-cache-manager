from data import *

def build(R):
    R.h1("9. Hardware Results: RTL Simulation, Waveforms and Verification")
    R.p("This chapter presents the results for the three Verilog modules: how they were verified, the waveforms recorded during the testbench runs, the lint and mutation results and the synthesis statistics. Every figure showing signals was rendered directly from the value-change-dump (VCD) files written by the Verilator simulations (`docs/wave/trace_*.vcd`).", align="j")
    R.callout("How to read the waveform figures", "The figures are screenshots of the GTKWave waveform viewer (GTK 3 build, run in a virtual X display). Each opens one of the VCD dumps written by Verilator from the testbenches in `hw/tb` (rerun with `+trace`), with the signals, radix, zoom range and marker placed by a Tcl script kept in `docs/gtkwave/`. Buses are shown in hexadecimal or decimal as indicated in the signal values; the vertical line is the marker.")
    R.h2("9.1 Verification method")
    R.numbered(["**Golden models.** `uamm2/hwmodel.py` re-implements each module's integer arithmetic line for line.",
                "**Vector generation.** The same file writes stimulus and expected-output hex files (`hw/vectors/*.hex`), including directed boundary cases and random cases.",
                "**Self-checking testbenches.** Each testbench reads the vectors, drives the module, compares every output with the expectation and prints `PASS` or `FAIL` with a count.",
                "**Verilator 5.** `make -C hw test` builds and runs all three benches with timing support; `+trace` additionally dumps a VCD.",
                "**Lint.** `verilator --lint-only -Wall` on every module.",
                "**Mutation testing.** Twelve single-fault variants of the RTL are generated automatically; the regression must fail for every one.",
                "**Logic synthesis check.** Yosys `synth` with `check -assert` confirms the designs elaborate to generic gates with no inferred latches."])
    R.h2("9.2 Sweep scorer")
    R.p("The scorer test applies four metadata datasets of 64 blocks, each swept at four pressure levels (0, 242, 243, 255), which gives 16 sweeps; a further sweep exercises restart behaviour (17 in total). Dataset 0 is a set of directed boundary cases: for each of the scores 0, 1, 88, 89, 90, 173, 174, 175, 254 and 255 a metadata word is generated whose weighted score equals that value, once unprotected and once protected, followed by corner cases (all zeros, all 255, single maxima) and random fill. The other three datasets are random with a 20 % chance of the `protected` bit.", align="j")
    R.fig("g01_scorer_start.png", "Sweep start. `start` is asserted for one clock; `busy` rises on the next edge; the read counter then increments once per cycle. `s0_data` shows the metadata words read from the SRAM; the results appear two cycles later as `out_valid` with the block index, score and tier.", 6.4)
    R.p("The first results are the directed boundary cases, which makes the figure self-explanatory: block 0 has score 0 and tier 2 (cold) because it is unprotected; block 1 has the same metadata with `protected` set and is forced to tier 0; block 2 has score 1; blocks 4 and 5 have score 88 (just below the warm threshold of 89, cold when unprotected); blocks 6 and 7 have score 89 (exactly at the warm threshold: warm when unprotected, hot when protected); blocks 8 and 9 have score 90; block 10 has score 173 (one below the hot threshold: warm). The alternation of tier values 2, 0, 2, 0, 2, 0, 1, 0, 1, 0, 1 is the pair-wise unprotected/protected pattern.", align="j")
    R.fig("g03_scorer_evict.png", "Eviction under pressure. The pressure latched for the sweep changes from 242 to 243; in the sweep at 243, `out_evict` is raised for blocks 0, 2 and 4 (unprotected, score below 89) and not for the protected blocks or for blocks 6 and 7 (score 89).", 6.4)
    R.p("This is the behaviour the UAMM-1 RTL got wrong (it ignored `protected`) and the main reason for the boundary vectors: at pressure 242 no block is flagged for eviction; at 243 exactly the unprotected blocks whose score is below 89 are flagged.", align="j")
    R.fig("g02_scorer_done.png", "End of a sweep. The read counter reaches 64 and `busy` falls; the last result (block 63) is emitted one cycle later together with the `done` pulse. A new `start` can follow immediately.", 6.4)
    R.p("The latency is as designed: for NBLK = 64 a sweep produces its last result NBLK + 3 cycles after `start`, and it sustains one block per cycle. At a 100 MHz clock a 64-block sweep takes under 0.7 µs; for 4,096 blocks it would take under 41 µs, so the sweep is cheap compared with the planning interval of several decode steps. (These are cycle counts from the simulation multiplied by an assumed clock, not timing-closure results.)", align="j")
    R.fig("g04_scorer_reset.png", "Asynchronous reset in the middle of a sweep. `rst_n` is pulled low; `busy`, `s0_v` and `out_valid` drop immediately, the read counter returns to zero and a later `start` begins a clean sweep.", 6.4)
    R.table(["Check", "Count", "Result"],
            [["Sweeps (4 datasets × 4 pressures + restart)", "17", "PASS"], ["Total per-block comparisons (score, tier, evict, index, valid)", "1,088", "PASS"],
             ["`start` while busy ignored", "directed", "PASS"], ["Reset in the middle of a sweep", "directed", "PASS"]],
            caption="Sweep scorer results.", widths=[3.8, 1.0, 1.0])
    R.h2("9.3 Codec pipeline")
    R.p("The quantiser and dequantiser testbench feeds 400 vectors of 16 lanes. The vectors are produced in four modes in rotation: full-range random, narrow range around a random centre, constant group (range zero) and extremes (0, 1, 254, 255). Further variations set the zero point inside the data or deliberately **below** the true minimum (stale statistics), which exercises the clamp at code 15. The testbench checks both the 16 packed 4-bit codes and the 16 dequantised bytes.", align="j")
    R.fig("g05_codec_pipeline.png", "Two vectors and a constant group through `uamm_kv_quant4` and `uamm_kv_dequant4`. `x_flat` is the 128-bit input; `q_flat` is the packed result one cycle after `in_valid`; the dequantiser is fed the code vector and step and produces `x′` one cycle later.", 6.4)
    R.p("The third vector in the figure is the constant group (`inv_q12 = 0`, `step_q8 = 0`): all lanes are `7e`, every code is zero and the dequantiser reproduces `7e` exactly, as the model predicts. The first two vectors have non-trivial ranges (parameters 251 and 4,389 for inv_q12), show the one-cycle latency of each stage, and match the expected nibble words.", align="j")
    R.table(["Check", "Count", "Result"], [["Codec vectors (quantiser and dequantiser outputs)", "400", "PASS"]], caption="Codec results.", widths=[3.8, 1.0, 1.0])
    R.h2("9.4 Block table")
    R.p("The block-table testbench performs 6,000 random operations (writes, lookups, and the occasional same-cycle read-and-write to one address) and compares every lookup with a behavioural reference table. It also checks that the valid bits are cleared by reset and that a read in the same cycle as a write returns the old entry.", align="j")
    R.fig("g06_block_table.png", "Random updates and lookups. `we` pulses write `{tier, codec, phys}` to the address on `waddr`; `rd_en` pulses start a lookup whose result (`hit`, `tier`, `codec`, `phys`) appears one cycle later through `rd_valid`. A lookup of a never-written address returns `hit = 0`; a hit returns the stored fields.", 6.4)
    R.table(["Check", "Count", "Result"], [["Writes", "3,049", "PASS"], ["Reads", "2,971", "PASS"], ["Reads that hit", "2,313", "PASS"]], caption="Block-table results.", widths=[3.8, 1.0, 1.0])
    R.h2("9.5 Regression, lint and mutation results")
    R.fig("t02_rtl_regression.png", "Output of `make -C hw test`: all three testbenches print PASS with their check counts.", 6.4)
    R.fig("t03_rtl_lint.png", "Output of `make -C hw lint` (`verilator --lint-only -Wall` on every module). Exit status 0, no warnings.", 6.4)
    R.p("**Mutation testing** measures the strength of the testbenches rather than the design: a testbench that cannot detect an injected fault is not protecting anything. Twelve mutants were generated, each a single small change to the RTL source, and the full regression was run on each.", align="j")
    R.fig("t04_mutation.png", "Mutation results: 12 of 12 injected faults are detected.", 6.0)
    R.p("The mutants cover weights, thresholds, rounding, clamping, saturation, the protected-block override and the valid-bit write. During development one mutant (removing the clamp at code 15) initially **survived** because no vector exercised an out-of-range code; the stale-zero-point vectors described in Section 9.3 were added to close that gap, and the mutant is now detected. This is the usual value of mutation testing: it found a hole in the tests that a pass count alone would not have shown.", align="j")
    R.h2("9.6 Synthesis statistics")
    import json
    sm = json.loads((ROOT / "docs/synth/summary.json").read_text())
    rows = []
    for n, lab in (("uamm_sweep_scorer", "Sweep scorer (64 blocks)"), ("uamm_kv_quant4", "Quantiser (16 lanes)"), ("uamm_kv_dequant4", "Dequantiser (16 lanes)"), ("uamm_block_table", "Block table (256 entries)")):
        d = sm[n]; ff = sum(v for k, v in d.items() if k.startswith("$_DFF") or k.startswith("$_SDFF")); xor = d.get("$_XOR_", 0) + d.get("$_XNOR_", 0)
        rows.append([lab, f"{d['cells']:,}", f"{ff:,}", f"{d['cells']-ff:,}", f"{d.get('$_MUX_',0):,}", f"{xor:,}"])
    R.table(["Module", "Generic cells", "Flip-flops", "Combinational", "Muxes", "XOR/XNOR"], rows,
            caption="Yosys `synth -flatten` generic-cell statistics (technology-independent). Raw reports are in Appendix C.", widths=[2.0, 1.0, 1.0, 1.1, 0.8, 0.8])
    R.fig("f21_yosys_cells.png", "Cell counts by module and split between sequential and combinational logic.", 6.0)
    R.bullets(["The **block table and the scorer metadata** appear as thousands of flip-flops because the memories are written as generic arrays and Yosys was run without a memory-mapping step. On an FPGA or ASIC flow they would be mapped to block RAM or SRAM macros. The flip-flop counts are therefore an upper bound that reflects storage, not control logic.",
               "The **quantiser is the largest logic block** (about 11,300 cells), dominated by the sixteen parallel 8 × 16-bit multipliers. A shift-based reciprocal (power-of-two ranges) or time-multiplexing fewer lanes would shrink it; this is noted as a design trade-off in Chapter 12.",
               "These are technology-independent counts. **No place-and-route, timing or power analysis has been done.**"])
