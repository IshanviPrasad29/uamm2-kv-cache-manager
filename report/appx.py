from data import *
import json
KEYS = [("resident_vs_fp16", "res/FP16"), ("dropped_tokens", "drop"), ("attn_rel_l2_mean", "err"), ("attn_rel_l2_p95", "err p95"), ("attn_mass_recall_mean", "recall"),
        ("sim_fetch_ms_per_token", "fetch ms"), ("flash_block_reads", "flash"), ("migrations", "migr"), ("reencodes", "reenc"), ("overflow_total", "overflow B")]
def tab(R, title, d, nested=False):
    rows = []
    for k, v in d.items():
        rows.append([k] + [ms(v, kk, "{:.4g}") for kk, _ in KEYS])
    R.table(["setting"] + [l for _, l in KEYS], rows, caption=title, size=6.5)

def build(R):
    R.h1("Appendix A. Glossary")
    R.table(["Term", "Meaning"],
            [["KV cache", "Stored keys and values of all previous tokens, per layer and head"], ["Block", "Fixed group of 16 consecutive tokens; unit of placement"],
             ["Tier", "Memory level (T0 SRAM, T1/T2 DRAM, T3 flash)"], ["Codec", "Storage format of a block (fp16, int8, kivi4, kivi2, optionally entropy-coded)"],
             ["Heavy hitter", "Token that accumulates a large share of attention"], ["Sink", "Initial tokens that attract attention irrespective of content"],
             ["Ratchet", "Rule that precision may only decrease"], ["Hysteresis", "Rule that suppresses a move unless it is clearly better"],
             ["Recall (attention mass)", "Share of the dense attention probability falling on tokens that were read"], ["RVV", "RISC-V Vector extension"],
             ["VCD", "Value change dump: waveform file written by simulators, viewable in GTKWave"]],
            caption="Glossary.", widths=[1.6, 5.0])
    R.h1("Appendix B. Raw Experiment Tables")
    R.p("Mean ± standard deviation over seeds. Columns: resident / FP16, dropped tokens, mean relative L2 error, 95th-percentile error, attention-mass recall, modelled fetch ms per token, flash reads, migrations, re-encodes, overflow bytes.", align="j")
    tab(R, "E1 ablation (5 seeds, 1,536 tokens)", load("e1_ablation"))
    tab(R, "E2 context scaling: combined system, all codecs (3 seeds)", load("e2_context_scaling")["v2_planner_all_codecs"])
    tab(R, "E2 context scaling: original threshold policy (3 seeds)", load("e2_context_scaling")["v1_threshold_fp16"])
    tab(R, "E3 attention block budget", load("e3_budget"))
    tab(R, "E4 quality weight lambda", load("e4_lambda"))
    tab(R, "E5 capacity factor, combined system", load("e5_capacity")["v2"])
    tab(R, "E5 capacity factor, original policy", load("e5_capacity")["v1"])
    R.h1("Appendix C. Core Source Listings")
    R.p("The complete project is supplied as a separate archive; the listings below are the parts the report describes most closely.", align="j")
    for f in ["hw/rtl/uamm_sweep_scorer.v", "hw/rtl/uamm_kv_codec.v", "hw/rtl/uamm_block_table.v", "ara/kv_kernels.c", "uamm2/codecs.py", "uamm2/planner.py", "uamm2/attention.py", "uamm2/h2o.py", "uamm2/paged_cache.py", "uamm2/simulator.py", "uamm2/config.py", "uamm2/hwmodel.py", "hw/tb/tb_sweep_scorer.v", "hw/tb/tb_kv_codec.v", "hw/tb/tb_block_table.v", "uamm2/tiers.py", "uamm2/workload.py", "uamm2/runtime.py", "tests/test_system.py", "tests/test_codecs.py"]:
        R.file_listing(f)
    R.h1("Appendix D. UAMM-1 Source Code (Original Prototype)")
    R.p("This appendix reproduces, unchanged, the prototype that Chapter 3 analyses: the Python modules (configuration, paged cache, controller, simulator) followed by the Verilog controller and its testbench. The concatenated listings are also stored in the archive under `uamm1_original/`. They are the authors' own earlier code and are included so that the comparison in Chapters 4 and 8 can be checked against the actual implementation.", align="j")
    R.file_listing("uamm1_original/uamm1_python_listing.txt", title="D.1  UAMM-1 Python modules", size=7.2)
    R.file_listing("uamm1_original/uamm1_verilog_listing.txt", title="D.2  UAMM-1 Verilog controller and testbench", size=7.2)
    R.code((ROOT / "docs/captures/rtl_regression.txt").read_text(), title="RTL regression log", size=6.5)
