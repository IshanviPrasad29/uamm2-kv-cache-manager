import sys, json, subprocess, re
sys.path.insert(0, "/home/claude/uamm2_project/report")
from lib import *
import ch01, ch02, ch03, ch04, ch_gaps, ch05, ch_code, ch06, ch07, ch08, ch09, ch11, ch10, ch12, ch13, appx

def make(pages, toc_entries=None, lof=None):
    R = Report(pages)
    d = R.d
    # cover
    for _ in range(9): d.add_paragraph()
    t = d.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("Termpaper"); r.font.size = Pt(36); r.bold = True
    t = d.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("ELL305"); r.font.size = Pt(30); r.bold = True; r.font.color.rgb = RGBColor(0,0,0)
    for _ in range(4): d.add_paragraph()
    t = d.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("ISHANVI PRASAD (2024EE10406)"); r.font.size = Pt(18); r.bold = True
    t = d.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("Sweety (2024MT10535)"); r.font.size = Pt(18); r.bold = True
    d.add_page_break()
    for _ in range(8): d.add_paragraph()
    t = d.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("Unified Adaptive Memory Manager"); r.font.size = Pt(30); r.bold = True; r.font.color.rgb = RGBColor(0,0,0)
    t = d.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("for Key–Value Cache Management in Large Language Model Inference on Edge Processors"); r.font.size = Pt(16)
    d.add_page_break()
    R.toc.append((1, "Contents")); d.add_heading("Contents", 1)
    R.toc_block(toc_entries or [])
    for m in (ch01, ch02, ch03, ch04, ch_gaps, ch05, ch_code, ch06, ch07, ch08, ch09, ch11, ch10, ch12, ch13, appx):
        m.build(R)
    return R

if __name__ == "__main__":
    pages = json.load(open("/tmp/pages.json")) if len(sys.argv) > 1 else {}
    toc_entries = json.load(open("/tmp/toc.json")) if len(sys.argv) > 1 else None
    R = make(pages, toc_entries)
    json.dump(R.toc, open("/tmp/toc_all.json", "w"))
    out = "/home/claude/uamm2_project/UAMM2_Report.docx"
    R.save(out); print("saved", out, "figs", R.nfig, "tables", R.ntab)
