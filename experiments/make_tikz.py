"""Re-run the diagram scripts and export each diagram to TikZ (docs/tikz)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.figure as mf
import tikzexport
_orig = mf.Figure.savefig
def _sv(self, fname, *a, **k):
    n = str(fname).split("/")[-1]
    if n.startswith("d_") and tikzexport.export(self, n): print("tikz", n)
    return _orig(self, fname, *a, **k)
mf.Figure.savefig = _sv
import make_diagrams as D, make_report_figs as M
for f in (D.d_system, D.d_tiers, D.d_planner, D.d_decode_step, D.d_lifecycle, D.d_quant_axes, D.d_pack, D.d_rtl_scorer, D.d_rtl_codec, D.d_rtl_table, D.d_ara, D.d_memory_hier, D.d_integration_levels):
    f()
M.mindmap()
