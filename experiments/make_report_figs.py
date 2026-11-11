"""Figures that frame the report: UAMM-1 vs UAMM-2 comparison, project mind map, work timeline."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs" / "figs"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def mean(rs, k): return float(np.mean([r[k] for r in rs]))


def compare():
    e1 = json.loads((ROOT / "experiments/results/e1_ablation.json").read_text())
    items = [("UAMM-1", "v1_threshold_fp16", "#111111"),
             ("4-bit\nonly", "static_kivi4_no_planner", "#777777"),
             ("UAMM-2\nINT8+4", "v2_planner_int8_kivi4", "#cfcfcf"),
             ("UAMM-2\nall", "v2_planner_all_codecs", "#bbbbbb")]
    panels = [("Tokens deleted (of 1,536)", lambda d: mean(d, "dropped_tokens"), "{:.0f}"),
              ("Resident size / FP16", lambda d: mean(d, "resident_vs_fp16"), "{:.2f}"),
              ("Attention-output error", lambda d: mean(d, "attn_rel_l2_mean"), "{:.2f}"),
              ("Fetch time, ms / token (model)", lambda d: mean(d, "sim_fetch_ms_per_token"), "{:.2f}")]
    fig, axs = plt.subplots(1, 4, figsize=(14, 4.4))
    for ax, (title, f, fmt) in zip(axs, panels):
        vals = [f(e1[k]) for _, k, _ in items]
        bars = ax.bar(range(4), vals, color=[c for *_, c in items], width=0.68, edgecolor='black', linewidth=1.2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=11)
        ax.set_xticks(range(4)); ax.set_xticklabels([l for l, *_ in items], fontsize=10)
        ax.set_title(title, fontsize=12); ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout()
    fig.savefig(FIG / "f24_v1_v2_compare.png", dpi=200); plt.close(fig)


def box(ax, x, y, w, h, text, fc, tc="black", fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#222222", lw=0.9))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal", linespacing=1.25)


def mindmap():
    steps = [
        ("1", "Problem", "Section 1", ["Decoding on a small RISC-V", "core is memory-bound", "KV cache grows with context", "Fast memory is only KBs"]),
        ("2", "Literature", "Section 2", ["PagedAttention, H2O, Quest,", "KIVI, EvicPress, ...", "Each paper pulls one lever", "None joins placement + precision"]),
        ("3", "UAMM-1", "Section 4", ["Paged 16-token blocks", "H2O-style importance score", "Three levels, two thresholds", "Verilog score unit"]),
        ("4", "Gaps in UAMM-1", "Sections 4 and 5", ["Deletes tokens when full", "Importance saturates", "No real tensors, no codec", "Migrations cost nothing"]),
        ("5", "UAMM-2", "Section 5", ["Placement = (tier, codec)", "One cost-based planner", "Hysteresis, precision ratchet", "Reads only blocks that matter"]),
        ("6", "Results", "Section 6", ["Ten simulated experiments", "Fair baselines added", "Three RTL blocks verified", "Ara kernel compiled only"]),
        ("7", "Future scope", "Section 7", ["Real model and real traces", "Planner cost in hardware", "Run the kernel on Ara", "Learned and adaptive policies"]),
        ("8", "Record", "Sections 8 and 9", ["Timeline 10 Aug to Nov", "Repository and git set-up", "Summary of claims", "Appendices"]),
    ]
    fig, ax = plt.subplots(figsize=(16, 7.4))
    ax.set_xlim(0, 16); ax.set_ylim(0, 7.4); ax.axis("off")
    W, H = 3.5, 2.95; gap = 0.45
    pos = []
    for i, st in enumerate(steps):
        r, c = divmod(i, 4)
        x = 0.25 + c * (W + gap); y = 4.2 - r * 3.8
        pos.append((x, y))
        ax.add_patch(Rectangle((x, y), W, H, fc="white", ec="black", lw=1.8))
        ax.add_patch(Rectangle((x, y + H - 0.62), W, 0.62, fc="#222222", ec="black", lw=1.8))
        ax.text(x + 0.12, y + H - 0.31, "Step " + st[0], color="white", fontsize=12, fontweight="bold", va="center")
        ax.text(x + W - 0.12, y + H - 0.31, st[2], color="white", fontsize=10.5, va="center", ha="right")
        ax.text(x + W / 2, y + H - 1.0, st[1], fontsize=14, fontweight="bold", ha="center", va="center")
        for k, line in enumerate(st[3]):
            ax.text(x + 0.18, y + H - 1.5 - k * 0.38, line, fontsize=10.5, va="center")
    for i in range(7):
        (x1, y1), (x2, y2) = pos[i], pos[i + 1]
        if i == 3:
            ax.plot([pos[3][0] + W / 2, pos[3][0] + W / 2], [pos[3][1], 3.78], color="black", lw=2)
            ax.plot([pos[3][0] + W / 2, pos[4][0] + W / 2], [3.78, 3.78], color="black", lw=2)
            ax.annotate("", xy=(pos[4][0] + W / 2, pos[4][1] + H + 0.04), xytext=(pos[4][0] + W / 2, 3.78), arrowprops=dict(arrowstyle="-|>", lw=2, color="black"))
        else:
            ax.annotate("", xy=(x2 - 0.02, y2 + H / 2), xytext=(x1 + W + 0.02, y1 + H / 2), arrowprops=dict(arrowstyle="-|>", lw=2, color="black"))
    fig.savefig(FIG / "d_mindmap.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def timeline():
    phases = [("Looking at AGUs and RISC-V extensions", "2026-08-10", "2026-08-15", "#777777"),
              ("Finding the real bottleneck: memory", "2026-08-15", "2026-08-20", "#444444"),
              ("Writing the first draft", "2026-08-20", "2026-08-24", "#999999"),
              ("Reading papers", "2026-08-25", "2026-09-20", "#bbbbbb"),
              ("Building UAMM-1", "2026-09-08", "2026-09-30", "#111111"),
              ("Reviewing UAMM-1, listing gaps", "2026-09-28", "2026-10-10", "#666666"),
              ("Building UAMM-2 (engine)", "2026-10-06", "2026-10-28", "#222222"),
              ("RTL, kernel, waveforms", "2026-10-20", "2026-11-02", "#bbbbbb"),
              ("Experiments, review, corrections", "2026-10-28", "2026-11-09", "#444444"),
              ("Final report and Turnitin", "2026-11-05", "2026-11-12", "#999999")]
    import datetime as dt
    d = lambda s: dt.date.fromisoformat(s)
    fig, ax = plt.subplots(figsize=(13, 5.2))
    for i, (n, a, b, c) in enumerate(phases):
        ax.barh(i, (d(b) - d(a)).days, left=d(a).toordinal(), color=c, height=0.6)
    ax.axvline(d("2026-08-24").toordinal(), color="#222222", ls="--", lw=1)
    ax.text(d("2026-08-24").toordinal() + 1, -0.9, "first draft submitted, 24 Aug", fontsize=12)
    ticks = [d(x).toordinal() for x in ("2026-08-10", "2026-09-01", "2026-09-15", "2026-10-01", "2026-10-15", "2026-11-01", "2026-11-12")]
    ax.set_xticks(ticks); ax.set_xticklabels(["10 Aug", "1 Sep", "15 Sep", "1 Oct", "15 Oct", "1 Nov", "12 Nov"], fontsize=12)
    ax.set_yticks(range(len(phases))); ax.set_yticklabels([p[0] for p in phases], fontsize=12); ax.tick_params(axis="y", length=0); ax.invert_yaxis(); ax.spines["left"].set_visible(False)
    ax.set_xlim(d("2026-08-06").toordinal(), d("2026-11-14").toordinal())
    fig.tight_layout(); fig.savefig(FIG / "d_timeline.png", dpi=200, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    compare(); mindmap(); timeline()
