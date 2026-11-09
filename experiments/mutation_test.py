"""RTL mutation testing: inject one bug at a time into a copy of hw/, run the regression, record whether it FAILs."""
import re, shutil, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
M = [  # (file, module-target, description, old, new)
 ("uamm_sweep_scorer.v","sweep_scorer","W_IMP 154 -> 153","8'd154","8'd153"),
 ("uamm_sweep_scorer.v","sweep_scorer","W_FRQ 38 -> 40","8'd38","8'd40"),
 ("uamm_sweep_scorer.v","sweep_scorer","TH_HOT 174 -> 175","8'd174","8'd175"),
 ("uamm_sweep_scorer.v","sweep_scorer","TH_WARM 89 -> 88","8'd89","8'd88"),
 ("uamm_sweep_scorer.v","sweep_scorer","EVICT_P 243 -> 244","8'd243","8'd244"),
 ("uamm_sweep_scorer.v","sweep_scorer","tier ignores protected","prot ? 2'b00 :","1'b0 ? 2'b00 :"),
 ("uamm_sweep_scorer.v","sweep_scorer","evict ignores protected","(!prot) && (pres","(1'b1) && (pres"),
 ("uamm_kv_codec.v","kv_codec","quant: drop +2048 rounding","(prod + 32'd2048) >> 12","(prod) >> 12"),
 ("uamm_kv_codec.v","kv_codec","quant: remove clamp at 15","(r > 32'd15) ? 4'd15 : r[3:0]","r[3:0]"),
 ("uamm_kv_codec.v","kv_codec","dequant: drop +128 rounding","(prod + 32'd128) >> 8","(prod) >> 8"),
 ("uamm_kv_codec.v","kv_codec","dequant: remove saturation at 255","(sum > 32'd255) ? 8'd255 : sum[7:0]","sum[7:0]"),
 ("uamm_block_table.v","block_table","valid bit write disabled","if (we) vbits[waddr] <= w_valid;","if (we) vbits[waddr] <= 1'b1;"),
]
def main():
    rows=[]
    for f,mod,desc,old,new in M:
        with tempfile.TemporaryDirectory() as td:
            hw=Path(td)/"hw"; shutil.copytree(ROOT/"hw",hw,ignore=shutil.ignore_patterns("build"))
            shutil.copytree(ROOT/"uamm2",Path(td)/"uamm2"); 
            p=hw/"rtl"/f; s=p.read_text()
            if old not in s: rows.append((desc,"PATTERN NOT FOUND")); continue
            p.write_text(s.replace(old,new,1))
            r=subprocess.run(["make","-C",str(hw),"test"],capture_output=True,text=True)
            out=r.stdout+r.stderr
            rows.append((desc,"KILLED (testbench FAIL)" if (r.returncode!=0 or "FAIL" in out) else "SURVIVED"))
    w=max(len(d) for d,_ in rows)
    lines=[f"{'mutation':<{w}}  result"]+[f"{d:<{w}}  {r}" for d,r in rows]
    lines.append(f"\n{sum('KILLED' in r for _,r in rows)}/{len(rows)} mutants killed")
    txt="\n".join(lines); print(txt)
    (ROOT/"docs/captures/mutation.txt").write_text(txt+"\n")
main()
