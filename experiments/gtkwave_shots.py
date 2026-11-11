"""Drive the real GTKWave (under Xvfb) on the Verilator VCD dumps and capture screenshots."""
import pathlib; ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[1])
import subprocess, sys, time, os, vcdvcd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
W = ROOT / "docs" / "wave"; OUT = ROOT / "docs" / "gtkwave"; OUT.mkdir(exist_ok=True)

def first(v, suffix, val="1"):
    for s in v.signals:
        if s.endswith(suffix):
            return [t for t, x in v[s].tv if x == val]
    raise KeyError(suffix)

def shot(vcd, name, sigs, t0, t1, marker, dec=()):
    """sigs: full signal names. dec: names to show in decimal; others hex."""
    tcl = OUT / f"{name}.tcl"
    lines = ["set sigs [list " + " ".join("{%s}" % s for s in sigs) + "]", "gtkwave::addSignalsFromList $sigs",
             "gtkwave::/Edit/UnHighlight_All"]
    for s in sigs:
        if "[" in s:
            lines += ["gtkwave::highlightSignalsFromList {%s}" % s,
                      "gtkwave::/Edit/Data_Format/%s" % ("Decimal" if s in dec else "Hex"), "gtkwave::/Edit/UnHighlight_All"]
    lines += [f"gtkwave::setZoomRangeTimes {t0} {t1}", f"gtkwave::setMarker {marker}"]
    tcl.write_text("\n".join(lines) + "\n")
    png = OUT / f"{name}.png"
    H = 190 + 24 * len(sigs)
    rel = lambda q: os.path.relpath(q, ROOT_DIR)
    cmd = (f"cd {ROOT_DIR} && gtkwave -S {rel(tcl)} {rel(vcd)} >/dev/null 2>&1 & GP=$!; sleep 4; "
           f"WID=$(xdotool search --name GTKWave | head -1); xdotool windowmove $WID 0 0; xdotool windowsize $WID 1500 {H}; sleep 4; "
           f"import -window root {png}; kill $GP")
    subprocess.run(["xvfb-run", "-a", "-s", f"-screen 0 1500x{H}x24", "bash", "-c", cmd], check=False)
    print("shot", png, png.exists())

# sweep scorer
v = vcdvcd.VCDVCD(str(W / "trace_sweep_scorer.vcd"), store_tvs=True)
P = "tb_sweep_scorer.dut."
base = [P + x for x in ("clk", "rst_n", "start", "busy", "rd_cnt[6:0]", "s0_v", "s0_data[24:0]", "out_valid", "out_idx[5:0]", "out_score[7:0]", "out_tier[1:0]", "out_evict", "done")]
st = first(v, "dut.start")[0]
dec = {P + "rd_cnt[6:0]", P + "out_idx[5:0]", P + "out_score[7:0]", P + "out_tier[1:0]", P + "pres[7:0]"}
shot(W / "trace_sweep_scorer.vcd", "g01_scorer_start", base, st - 25000, st + 135000, st + 20000, dec)
dn = first(v, "dut.done")[0]
shot(W / "trace_sweep_scorer.vcd", "g02_scorer_done", base, dn - 90000, dn + 40000, dn, dec)
ev = first(v, "dut.out_evict")[0]
shot(W / "trace_sweep_scorer.vcd", "g03_scorer_evict", [P + x for x in ("clk", "pres[7:0]", "out_valid", "out_idx[5:0]", "out_score[7:0]", "out_tier[1:0]", "out_evict")], ev - 60000, ev + 100000, ev, dec)
falls = [t for t, x in v[[s for s in v.signals if s.endswith("dut.rst_n")][0]].tv if x == "0" and t > 100000]
shot(W / "trace_sweep_scorer.vcd", "g04_scorer_reset", [P + x for x in ("clk", "rst_n", "busy", "rd_cnt[6:0]", "s0_v", "out_valid", "done")], falls[-1] - 100000, falls[-1] + 90000, falls[-1], dec)
# codec
v2 = vcdvcd.VCDVCD(str(W / "trace_kv_codec.vcd"), store_tvs=True); Q = "tb_kv_codec."
iv = first(v2, "uq.in_valid")[0]
names = [Q + x for x in ("uq.clk", "uq.in_valid", "uq.zp[7:0]", "uq.inv_q12[15:0]", "uq.x_flat[127:0]", "uq.out_valid", "uq.q_flat[63:0]", "ud.step_q8[15:0]", "ud.out_valid", "ud.x_flat[127:0]")]
shot(W / "trace_kv_codec.vcd", "g05_codec_pipeline", names, iv - 15000, iv + 85000, iv + 20000, {Q + "uq.zp[7:0]", Q + "uq.inv_q12[15:0]", Q + "ud.step_q8[15:0]"})
# block table
v3 = vcdvcd.VCDVCD(str(W / "trace_block_table.vcd"), store_tvs=True); B = "tb_block_table.dut."
h = first(v3, "dut.hit")[0] + 50000
names = [B + x for x in ("clk", "we", "waddr[7:0]", "w_valid", "w_tier[1:0]", "w_codec[1:0]", "w_phys[11:0]", "rd_en", "raddr[7:0]", "rd_valid", "hit", "tier[1:0]", "codec[1:0]", "phys[11:0]")]
shot(W / "trace_block_table.vcd", "g06_block_table", names, h - 80000, h + 100000, h, {B + x for x in ("waddr[7:0]", "raddr[7:0]", "w_tier[1:0]", "w_codec[1:0]", "tier[1:0]", "codec[1:0]")})
