"""Architecture / concept diagrams for the report (drawn with matplotlib, no data dependency)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle   # noqa: E402

FIG = Path(__file__).resolve().parent.parent / "docs" / "figs"
FIG.mkdir(parents=True, exist_ok=True)

C = {"blue": "#222222", "lblue": "#ececec", "green": "#222222", "lgreen": "#f6f6f6", "orange": "#222222",
     "lorange": "#dddddd", "purple": "#222222", "lpurple": "#f0f0f0", "gray": "#222222", "lgray": "#e6e6e6",
     "red": "#222222", "lred": "#d4d4d4", "yellow": "#222222", "lyellow": "#ffffff"}


def canvas(w=10, h=6, xlim=(0, 100), ylim=(0, 60)):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fc="lblue", ec="blue", fs=9, bold=False, tc="black", ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=C.get(fc, fc),
                       ec=C.get(ec, ec), lw=1.8, ls=ls)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, wrap=True)


def arrow(ax, x1, y1, x2, y2, text=None, color="gray", style="-|>", fs=7.5, ls="-", rad=0.0, tdx=0, tdy=1.2):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12, color=C.get(color, color),
                        lw=1.8, ls=ls, connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if text:
        ax.text((x1 + x2) / 2 + tdx, (y1 + y2) / 2 + tdy, text, ha="center", va="center", fontsize=fs,
                color=C.get(color, color))


def save(fig, name):
    fig.savefig(FIG / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("fig", name)


def d_system():
    fig, ax = canvas(11, 7.4, ylim=(0, 74))
    box(ax, 2, 62, 96, 8, "Application / OpenAI-style API   (chat, completion)", "lgray", "gray", 10, True)
    box(ax, 2, 49, 96, 9, "Model runtime  -  compiler-based runtime (TVM-compiled model, 4-bit weights)   [integration seam: RuntimeAdapter]",
        "lpurple", "purple", 9.5, True)
    arrow(ax, 50, 62, 50, 58, color="gray"); arrow(ax, 50, 49, 50, 45, "KV hooks: write_token / read_block / migrate / attention weights", "purple")
    box(ax, 2, 28, 96, 17, "", "lyellow", "yellow")
    ax.text(4, 42.5, "Software control plane (CVA6 scalar core or host CPU)", fontsize=10, fontweight="bold", color=C["yellow"])
    box(ax, 4, 30, 17.5, 9, "Paged block table\n16-token blocks\nkey min/max summaries", "white", "yellow", 8)
    box(ax, 23.5, 30, 17.5, 9, "H2O-style importance\ndecayed attention score\nsink + recent guard", "white", "yellow", 8)
    box(ax, 43, 30, 17.5, 9, "Utility planner\n(tier, codec) per block\nhysteresis, greedy knapsack", "white", "yellow", 8)
    box(ax, 62.5, 30, 17.5, 9, "Calibration at seal\nrel. error + size for\nevery codec", "white", "yellow", 8)
    box(ax, 82, 30, 14, 9, "Block-sparse\nselection\n(min/max bound)", "white", "yellow", 8)
    for x in (30, 50, 70):
        arrow(ax, x, 28, x, 22, color="orange")
    box(ax, 2, 8, 30, 14, "Ara RVV 1.0 vector unit\nfused QK on packed 4-bit keys\nsoftmax row-sums for H2O\nVRF slice per lane", "lgreen", "green", 8.5, True)
    box(ax, 35, 8, 30, 14, "KV codec engine (RTL)\nquant4 / dequant4 on AXI-stream\n(or any external codec engine)\nsweep scorer + block table", "lorange", "orange", 8.5, True)
    box(ax, 68, 8, 30, 14, "DMA / block mover\nscatter-gather\ntier transitions + prefetch", "lblue", "blue", 8.5, True)
    arrow(ax, 32, 15, 35, 15, style="<|-|>", color="gray"); arrow(ax, 65, 15, 68, 15, style="<|-|>", color="gray")
    for i, (x, t) in enumerate(((4, "T0  SRAM / VRF\nprotected set"), (27, "T1  DRAM\nany raw codec"),
                                (50, "T2  DRAM\nkivi4 / kivi2"), (73, "T3  flash\nzlib-coded"))):
        box(ax, x, 0.5, 21, 5.5, t, "lgray", "gray", 8)
    arrow(ax, 50, 8, 50, 6.3, color="gray")
    ax.set_title("UAMM-2 system architecture", fontsize=12, fontweight="bold")
    save(fig, "d_system_architecture.png")


def d_tiers():
    fig, ax = canvas(10, 5.6, ylim=(0, 56))
    cols = [("T0", "SRAM / VRF window", "protected: sink + recent\nfp16, never moved", "lgreen", "green", "24 KB", "0.0005 ms"),
            ("T1", "DRAM (hot)", "any raw codec\n(fp16..kivi2)", "lblue", "blue", "48 KB", "0.002 ms"),
            ("T2", "DRAM (warm)", "kivi4 | kivi2", "lpurple", "purple", "32 KB", "0.002 ms"),
            ("T3", "Flash", "int8z | kivi4z | kivi2z\n(zlib on the blob)", "lorange", "orange", "64 KB", "0.1 ms")]
    for i, (t, n, c, fc, ec, cap, lat) in enumerate(cols):
        x = 2 + i * 24.5
        box(ax, x, 30, 22, 20, f"{t}\n{n}", fc, ec, 10, True)
        box(ax, x, 15, 22, 12, "codecs:\n" + c, "white", ec, 8.5)
        box(ax, x, 4, 22, 8, f"capacity (default): {cap}\nbase latency (assumed): {lat}", "lgray", "gray", 8)
        if i < 3:
            arrow(ax, x + 22.3, 40, x + 24.2, 40, color="gray")
    ax.text(50, 53.5, "Tier / codec map (a placement is a (tier, codec) pair)", ha="center", fontsize=12, fontweight="bold")
    ax.text(50, 0.6, "Left to right: faster, smaller, more precise  ->  slower, larger, more compressed",
            ha="center", fontsize=9, color=C["gray"])
    save(fig, "d_tier_codec_map.png")


def d_planner():
    fig, ax = canvas(10, 8.2, ylim=(0, 82))
    box(ax, 30, 74, 40, 6, "Every plan_interval decode steps", "lgray", "gray", 10, True)
    box(ax, 30, 64, 40, 6, "Refresh H2O block importance\n(normalize by 95th percentile)", "lyellow", "yellow", 9)
    box(ax, 30, 54, 40, 6, "Protected blocks: fixed in T0 (sink + recent)\ncharge their bytes to T0", "lgreen", "green", 9)
    box(ax, 30, 44, 40, 6, "Sort movable blocks by heat\nheat = 0.5*hit_ema + 0.5*importance", "lblue", "blue", 9)
    box(ax, 4, 28, 40, 12, "For each block, for each allowed (tier, codec):\n  skip if codec precision < block precision (ratchet)\n  skip if tier has no room (with headroom)\n  cost = p*read + lambda*heat*err + congestion", "white", "blue", 8.5)
    box(ax, 56, 28, 40, 12, "Hysteresis:\nkeep current if residency < min  OR\ncost(current) <= cost(best) + margin*cost(best)", "white", "purple", 8.5)
    box(ax, 4, 12, 40, 10, "Nothing fits?\n  allow_drop & importance < threshold -> DROP\n  else stay put (overflow is reported)", "lred", "red", 8.5)
    box(ax, 56, 12, 40, 10, "Apply plan: re-encode if codec changed\n(lossless blob transform if same precision)\nmove tier; charge migration cost", "lorange", "orange", 8.5)
    box(ax, 30, 1, 40, 6, "Updated block table / tier occupancy", "lgray", "gray", 9, True)
    arrow(ax, 50, 74, 50, 70.5); arrow(ax, 50, 64, 50, 60.5); arrow(ax, 50, 54, 50, 50.5)
    arrow(ax, 40, 44, 24, 40.5); arrow(ax, 44, 34, 56, 34, "best vs current"); arrow(ax, 76, 28, 76, 22.5)
    arrow(ax, 24, 28, 24, 22.5); arrow(ax, 76, 12, 62, 7.5); arrow(ax, 24, 12, 38, 7.5)
    ax.set_title("UAMM-2 planner (one planning pass)", fontsize=12, fontweight="bold")
    save(fig, "d_planner_flow.png")


def d_decode_step():
    fig, ax = canvas(11, 5.2, ylim=(0, 52))
    steps = [("1  append\ntoken K,V", "lgray", "gray"), ("2  protect\nsink + recent", "lgreen", "green"),
             ("3  select blocks\nmin/max bound\n(top-B)", "lblue", "blue"), ("4  fetch + decode\nper tier/codec", "lorange", "orange"),
             ("5  attention\nsoftmax(QK)V", "lpurple", "purple"), ("6  H2O update\ndecay + add\nattn weights", "lyellow", "yellow")]
    for i, (t, fc, ec) in enumerate(steps):
        x = 1.5 + i * 16.2
        box(ax, x, 22, 14, 14, t, fc, ec, 8.5, True)
        if i < 5:
            arrow(ax, x + 14.3, 29, x + 15.9, 29)
    box(ax, 2, 4, 44, 10, "Skipped blocks are never fetched or decoded:\nthis is where the tiering pays off", "white", "blue", 9)
    box(ax, 52, 4, 46, 10, "Every plan_interval steps:\nrefresh importance -> plan -> migrate / re-encode", "white", "orange", 9)
    arrow(ax, 48.5, 20, 48.5, 14.5, color="blue"); arrow(ax, 88, 22, 78, 14.5, color="orange")
    ax.set_title("One decode step through UAMM-2", fontsize=12, fontweight="bold")
    save(fig, "d_decode_step.png")


def d_lifecycle():
    fig, ax = canvas(10.5, 4.8, ylim=(0, 48))
    box(ax, 2, 18, 15, 12, "OPEN\nstaging fp16\nT0, protected", "lgreen", "green", 8.5, True)
    box(ax, 22, 18, 15, 12, "SEALED\nfp16 payload\nT0", "lblue", "blue", 8.5, True)
    box(ax, 42, 18, 15, 12, "CALIBRATED\nerr_table +\nsize_table", "lyellow", "yellow", 8.5, True)
    box(ax, 62, 18, 15, 12, "PLACED\n(tier, codec)\nby planner", "lpurple", "purple", 8.5, True)
    box(ax, 82, 18, 15, 12, "DROPPED\n(opt-in)\nH2O-style", "lred", "red", 8.5, True)
    for x in (17.3, 37.3, 57.3, 77.3):
        arrow(ax, x, 24, x + 4.4, 24)
    ax.text(19.5, 27.5, "16th token", fontsize=7.5); ax.text(39, 27.5, "encode x7 codecs", fontsize=7.5)
    ax.text(59, 27.5, "plan pass", fontsize=7.5); ax.text(78, 27.5, "no fit", fontsize=7.5)
    arrow(ax, 69.5, 18, 69.5, 9, rad=0.0); arrow(ax, 69.5, 9, 69.5, 18, rad=0.0)
    ax.text(70.5, 8, "re-plan: move tier or lower precision\n(ratchet: precision never increases)", fontsize=8, color=C["purple"])
    ax.set_title("Block lifecycle", fontsize=12, fontweight="bold")
    save(fig, "d_block_lifecycle.png")


def d_quant_axes():
    import numpy as np
    rng = np.random.default_rng(3)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    K = rng.normal(size=(16, 24)) * 0.5
    K[:, 5] *= 8; K[:, 17] *= 6
    axs[0].imshow(abs(K), cmap="Greys", aspect="auto"); axs[0].set_title("Key block |K| (tokens x channels)\nchannels 5 and 17 are outliers", fontsize=9)
    axs[0].set_xlabel("channel"); axs[0].set_ylabel("token")
    for t in range(16):
        axs[1].add_patch(Rectangle((0, t), 24, 1, fc="none", ec=C["red"], lw=0.8))
    axs[1].imshow(abs(K), cmap="Greys", aspect="auto"); axs[1].set_title("Per-TOKEN groups (rows)\nthe outlier channel inflates every row's range", fontsize=9)
    axs[1].set_xlabel("channel")
    for c in range(24):
        axs[2].add_patch(Rectangle((c, 0), 1, 16, fc="none", ec=C["green"], lw=0.8))
    axs[2].imshow(abs(K), cmap="Greys", aspect="auto"); axs[2].set_title("Per-CHANNEL groups (columns), used for keys\noutliers get their own scale", fontsize=9)
    axs[2].set_xlabel("channel")
    for a in axs:
        a.set_xlim(0, 24); a.set_ylim(16, 0)
    save(fig, "d_quant_axes.png")


def d_pack():
    fig, ax = canvas(10, 3.6, ylim=(0, 36))
    ax.text(50, 33, "4-bit packing (matches codecs.py, the RTL and the Ara kernel)", ha="center", fontsize=11, fontweight="bold")
    for b in range(4):
        x = 4 + b * 23.5
        ax.text(x + 10, 27, f"byte {b}", ha="center", fontsize=9)
        box(ax, x, 15, 10.5, 9, f"hi nibble\nelement {2*b+1}", "lorange", "orange", 8)
        box(ax, x + 11, 15, 10.5, 9, f"lo nibble\nelement {2*b}", "lblue", "blue", 8)
    ax.text(50, 8, "row-major (T, d); element i = t*d + c lives in byte i//2; even i in the LOW nibble.   d must be even.",
            ha="center", fontsize=9)
    ax.text(50, 3, "bits [7:4] | [3:0]   ->   hardware lane i occupies bits [4i+3 : 4i] of the packed word",
            ha="center", fontsize=9, color=C["gray"])
    save(fig, "d_packing.png")


def d_rtl_scorer():
    fig, ax = canvas(11, 4.6, ylim=(0, 46))
    box(ax, 1, 26, 14, 14, "Metadata SRAM\n64 x 25 bit\n{prot,freq,rec,imp}", "lgray", "gray", 8.5, True)
    box(ax, 19, 26, 14, 14, "issue counter\nrd_cnt (AW+1)\nsync read", "lblue", "blue", 8.5, True)
    box(ax, 37, 26, 14, 14, "stage 0 regs\ns0_v, s0_idx,\ns0_data", "lpurple", "purple", 8.5, True)
    box(ax, 55, 26, 22, 14, "score = (imp*154 + rec*64 +\nfreq*38) >> 8\ntier thresholds 174 / 89\nevict: p>=243 & score<89\n& !protected", "lorange", "orange", 8, True)
    box(ax, 81, 26, 17, 14, "stage 1 regs\nout_valid, idx,\nscore, tier, evict", "lgreen", "green", 8.5, True)
    for x in (15.3, 33.3, 51.3, 77.3):
        arrow(ax, x, 33, x + 3.4, 33)
    box(ax, 1, 6, 30, 12, "write port (we, waddr, wdata)\nfirmware loads block metadata", "white", "gray", 8.5)
    box(ax, 38, 6, 30, 12, "control FSM: start (ignored while busy),\npressure latched, busy/done", "white", "blue", 8.5)
    box(ax, 72, 6, 26, 12, "1 block / cycle\nsweep latency = NBLK + 3 cycles", "white", "green", 8.5)
    arrow(ax, 16, 18, 8, 25.5); arrow(ax, 53, 18, 26, 25.5)
    ax.set_title("uamm_sweep_scorer: 2-stage pipeline", fontsize=12, fontweight="bold")
    save(fig, "d_rtl_scorer.png")


def d_rtl_codec():
    fig, ax = canvas(11, 4.4, ylim=(0, 44))
    box(ax, 1, 20, 17, 14, "x_flat\n16 x 8-bit\n(INT8 hot data)", "lgray", "gray", 8.5, True)
    box(ax, 22, 20, 21, 14, "uamm_kv_quant4\ndx = max(x - zp, 0)\nq = min(15,\n(dx*inv_q12 + 2048) >> 12)", "lorange", "orange", 8, True)
    box(ax, 47, 20, 17, 14, "q_flat\n16 x 4-bit\n= 64-bit word", "lgreen", "green", 8.5, True)
    box(ax, 68, 20, 22, 14, "uamm_kv_dequant4\nx' = min(255,\nzp + ((q*step_q8 + 128) >> 8))", "lorange", "orange", 8, True)
    box(ax, 93, 20, 6.5, 14, "x'", "lgray", "gray", 9, True)
    for x in (18.3, 43.3, 64.3, 90.3):
        arrow(ax, x, 27, x + 3.2 if x < 90 else x + 2.3, 27)
    box(ax, 22, 4, 42, 10, "per-group parameters from firmware:\nzp = min,  inv_q12 = round(15*4096/range),  step_q8 = round(range*256/15)", "white", "blue", 8.5)
    arrow(ax, 43, 14, 33, 19.5)
    ax.set_title("uamm_kv_codec: quantize / dequantize datapath (1 register stage each)", fontsize=12, fontweight="bold")
    save(fig, "d_rtl_codec.png")


def d_rtl_table():
    fig, ax = canvas(10, 3.8, ylim=(0, 38))
    box(ax, 2, 18, 22, 14, "lookup port\nrd_en, raddr", "lblue", "blue", 9, True)
    box(ax, 32, 14, 36, 22, "uamm_block_table\n256 entries x (tier2, codec2, phys12)\n+ valid bit vector (reset to 0)\ndirect-mapped, no TLB", "lorange", "orange", 9, True)
    box(ax, 76, 18, 22, 14, "rd_valid, hit,\ntier, codec, phys\n(1 cycle)", "lgreen", "green", 9, True)
    box(ax, 32, 1, 36, 8, "update port: we, waddr, w_valid, tier, codec, phys", "lgray", "gray", 8.5)
    arrow(ax, 24.3, 25, 31.7, 25); arrow(ax, 68.3, 25, 75.7, 25); arrow(ax, 50, 9.3, 50, 13.7)
    ax.set_title("uamm_block_table: logical block -> (tier, codec, physical slot)", fontsize=12, fontweight="bold")
    save(fig, "d_rtl_table.png")


def d_ara():
    fig, ax = canvas(10.5, 6.2, ylim=(0, 62))
    box(ax, 2, 40, 28, 18, "CVA6 scalar core\nplanner firmware\nblock table\nAXI master", "lyellow", "yellow", 9, True)
    box(ax, 36, 40, 62, 18, "", "lgreen", "green")
    ax.text(38, 55.5, "Ara vector unit (RVV 1.0)", fontsize=10, fontweight="bold", color=C["green"])
    box(ax, 38, 43, 14, 9, "dispatcher\nsequencer", "white", "green", 8)
    for i in range(4):
        box(ax, 54 + i * 10, 43, 9, 9, f"lane {i}\nVRF slice\nALU/FPU", "white", "green", 7)
    box(ax, 38, 28, 28, 8, "VLSU (AXI AR/AW)", "lblue", "blue", 9, True)
    box(ax, 70, 28, 28, 8, "kv_qk_dot_kivi4_rvv\nfused QK on packed nibbles", "lorange", "orange", 8.5, True)
    box(ax, 2, 12, 28, 10, "KV codec engine (RTL)\nAXI-stream dequant in load path", "lorange", "orange", 8.5, True)
    box(ax, 36, 12, 28, 10, "DRAM controller\nLPDDR", "lgray", "gray", 9, True)
    box(ax, 70, 12, 28, 10, "Flash / NVMe\nzlib-coded blocks", "lgray", "gray", 9, True)
    arrow(ax, 30, 49, 36, 49, "acc_req / acc_resp", tdy=2)
    arrow(ax, 52, 28, 50, 22.5); arrow(ax, 30, 17, 36, 17, style="<|-|>"); arrow(ax, 64, 17, 70, 17, style="<|-|>")
    arrow(ax, 52, 40, 52, 36.5)
    ax.set_title("Placement on a CVA6 + Ara SoC", fontsize=12, fontweight="bold")
    save(fig, "d_ara_integration.png")


def d_memory_hier():
    fig, ax = canvas(9, 6, ylim=(0, 60))
    levels = [("Registers / VRF slice", "KB", "~TB/s", "lgreen", 8), ("On-chip SRAM / cache", "100s KB - few MB", "100s GB/s", "lblue", 18),
              ("DRAM (LPDDR)", "GBs", "10s GB/s", "lpurple", 28), ("Flash / NVMe", "10s-100s GB", "~1 GB/s - few GB/s", "lorange", 38)]
    for i, (n, cap, bw, fc, _) in enumerate(levels):
        w = 30 + i * 18
        x = 50 - w / 2
        y = 46 - i * 11.5
        box(ax, x, y, w, 9, f"{n}     capacity: {cap}     bandwidth: {bw}", fc, "gray", 8.5, True)
    ax.text(50, 57, "Typical edge-SoC memory hierarchy (orders of magnitude, illustrative)", ha="center", fontsize=11, fontweight="bold")
    ax.text(50, 3, "Every step down: ~10x capacity, ~10x lower bandwidth, ~10-100x higher access latency", ha="center", fontsize=9, color=C["gray"])
    save(fig, "d_memory_hierarchy.png")


def d_integration_levels():
    fig, ax = canvas(10, 5, ylim=(0, 50))
    box(ax, 2, 28, 28, 16, "Policy layer\n(this work)\nwhich block, which tier,\nwhich precision", "lyellow", "yellow", 9, True)
    box(ax, 36, 28, 28, 16, "Codec engine\n(external engine or\nuamm_kv_codec)\nhow to shrink / expand", "lorange", "orange", 9, True)
    box(ax, 70, 28, 28, 16, "Compute\n(Ara / RVV kernels)\nattention over\npacked data", "lgreen", "green", 9, True)
    box(ax, 2, 6, 96, 14, "Compiler-based runtime owns model graph, weights, scheduling  -  UAMM-2 plugs in through RuntimeAdapter", "lpurple", "purple", 9, True)
    for x in (16, 50, 84):
        arrow(ax, x, 28, x, 20.5, style="<|-|>")
    ax.set_title("Separation of concerns", fontsize=12, fontweight="bold")
    save(fig, "d_separation_of_concerns.png")


def _rect(ax, x, y, w, h, text, fs=13, fc="white", bold=False, lw=1.8):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="#222222", lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.3)


def _diamond(ax, cx, cy, w, h, text, fs=12.5):
    ax.add_patch(plt.Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)], fc="#f3f3f3", ec="#222222", lw=1.8))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, linespacing=1.3)


def _arr(ax, x1, y1, x2, y2, label=None, fs=12, dx=0, dy=0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1.8, mutation_scale=16))
    if label:
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, label, fontsize=fs, ha="center", va="center", style="italic")


def d_planner():
    fig, ax = canvas(8.4, 10.4, xlim=(0, 84), ylim=(0, 104))
    _rect(ax, 22, 94, 40, 8, "Every 8 decode steps\n(plan interval)", 13, "#e8e8e8", True)
    _rect(ax, 22, 80, 40, 9, "Update the importance of\nevery block (H2O scores)", 13)
    _rect(ax, 22, 66, 40, 9, "Keep the sink and recent\nblocks fixed in T0, FP16", 13)
    _rect(ax, 22, 52, 40, 9, "Sort the other blocks,\nhottest first", 13)
    _rect(ax, 22, 36, 40, 11, "For each block, try every\n(tier, codec) that has room\nand does not raise precision", 13)
    _rect(ax, 22, 20, 40, 11, "Cost = read + decode time\n+ error penalty + congestion\nPick the cheapest", 13, "#f3f3f3")
    _diamond(ax, 42, 9.5, 36, 14, "Clearly better than\nthe current spot?")
    _arr(ax, 42, 94, 42, 89.3); _arr(ax, 42, 80, 42, 75.3); _arr(ax, 42, 66, 42, 61.3)
    _arr(ax, 42, 52, 42, 47.3); _arr(ax, 42, 36, 42, 31.3); _arr(ax, 42, 20, 42, 16.7)
    _rect(ax, 62, 3, 21, 13, "Yes: re-encode\nif needed, move,\ncharge the cost", 12, "#e8e8e8")
    _rect(ax, 1, 3, 20, 13, "No: leave the\nblock where\nit is", 12, "#e8e8e8")
    _arr(ax, 60, 9.5, 62, 9.5, "yes", 12, 0, 2.2); _arr(ax, 24, 9.5, 21, 9.5, "no", 12, 0, 2.2)
    save(fig, "d_planner_flow.png")


def d_decode_step():
    fig, ax = canvas(11, 6.4, xlim=(0, 110), ylim=(0, 64))
    steps = ["1  Append the\nnew token's\nK and V", "2  Mark sink and\nrecent blocks\nas protected", "3  Rank blocks\nwith the min/max\nbound, keep top 24",
             "4  Fetch and decode\nthe chosen blocks", "5  Attention:\nsoftmax(QK) V", "6  Update H2O\nscores"]
    for i, t in enumerate(steps):
        x = 2 + (i % 3) * 36
        y = 38 if i < 3 else 8
        _rect(ax, x, y, 30, 18, t, 12.5)
        if i in (0, 1, 3, 4):
            _arr(ax, x + 30, y + 9, x + 36, y + 9)
    _arr(ax, 90, 38, 90, 33.5); ax.plot([90, 17], [33.5, 33.5], color="#222222", lw=1.8); _arr(ax, 17, 33.5, 17, 26.3)
    ax.text(55, 61, "One decode step in UAMM-2", fontsize=14, fontweight="bold", ha="center")
    ax.text(55, 2.2, "Blocks that are not chosen in step 3 are never fetched or decoded. Every 8 steps the planner runs.", fontsize=11, ha="center", style="italic")
    save(fig, "d_decode_step.png")


def d_memory_hier():
    fig, ax = canvas(9.4, 6.4, xlim=(0, 94), ylim=(0, 64))
    levels = [("Registers / vector registers", "a few KB", "~TB/s"), ("On-chip SRAM", "100 KB to a few MB", "100s of GB/s"),
              ("DRAM (LPDDR)", "a few GB", "10s of GB/s"), ("Flash", "tens to hundreds of GB", "about 1 GB/s")]
    for i, (n, cap, bw) in enumerate(levels):
        w = 34 + i * 17; x = 47 - w / 2; y = 49 - i * 14
        _rect(ax, x, y, w, 11, f"{n}\n{cap}  |  {bw}", 12.5, ["#e8e8e8", "#f3f3f3", "#ffffff", "#f3f3f3"][i], True if i == 0 else False)
    ax.text(47, 62, "Typical edge memory hierarchy (schematic)", ha="center", fontsize=13.5, fontweight="bold")
    ax.text(47, 1.5, "Each step down: about 10x more capacity, 10x less bandwidth, much higher latency", ha="center", fontsize=11, style="italic")
    save(fig, "d_memory_hierarchy.png")


if __name__ == "__main__":
    for f in (d_system, d_tiers, d_planner, d_decode_step, d_lifecycle, d_quant_axes, d_pack, d_rtl_scorer,
              d_rtl_codec, d_rtl_table, d_ara, d_memory_hier, d_integration_levels):
        f()
