"""Export simple matplotlib diagrams (patches, lines, text, arrows) to TikZ.

Used for the diagrams in the report so that every diagram is drawn in TikZ.
"""
import re
from pathlib import Path
import matplotlib
from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon, FancyArrowPatch
from matplotlib.text import Annotation
from matplotlib.colors import to_rgb

OUT = Path(__file__).resolve().parent.parent / "docs" / "tikz"
OUT.mkdir(parents=True, exist_ok=True)


def col(c):
    if c is None: return None
    try: r, g, b = to_rgb(c)
    except Exception: return None
    return "{rgb,255:red,%d;green,%d;blue,%d}" % (round(r * 255), round(g * 255), round(b * 255))


def esc(s):
    s = s.replace("\\", "\\textbackslash{}")
    for a, b in (("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}")):
        s = s.replace(a, b)
    s = s.replace("\n", " \\\\ ")
    return s


def export(fig, name, target_w=6.2, land_w=9.5, minfs=7.5):
    axes = [a for a in fig.axes]
    if len(axes) != 1: return False
    ax = axes[0]
    if ax.images or ax.axison: return False
    W, H = fig.get_size_inches()
    pos = ax.get_position()
    aw, ah = pos.width * W, pos.height * H
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    texts = [t for t in ax.texts if t.get_text().strip() or (isinstance(t, Annotation) and t.arrow_patch is not None)]
    if ax.get_title(): texts.append(ax.title)
    fss = [t.get_fontsize() for t in texts if t.get_text().strip()] or [10]
    s = target_w / aw; land = False
    if s * min(fss) < minfs:
        s = min(land_w / aw, 1.0); land = True
        if aw * s > land_w: s = land_w / aw
    else:
        s = min(s, 1.0)
    sx = aw * s / (x1 - x0); sy = ah * s / (y1 - y0)
    X = lambda x: "%.3fin" % ((x - x0) * sx)
    Y = lambda y: "%.3fin" % ((y - y0) * sy)
    P = lambda x, y: "(%s,%s)" % (X(x), Y(y))
    out = []

    def arrowopts(style):
        st = style or "-"
        l = "latex" if "<" in st.split("-")[0] else ""
        r = "latex" if ">" in st.split("-")[-1] else ""
        if "<" in st and ">" not in st.replace("<|", ""): pass
        return "%s-%s" % (l, r)

    def lw(w): return "line width=%.2fpt" % (w * s * 0.9 + 0.2)
    def dash(ls): return ",dashed" if ls in ("--", "dashed") else (",dotted" if ls in (":", "dotted") else "")

    for p in ax.patches:
        fc = col(p.get_facecolor()) if p.get_fill() else None
        ec = col(p.get_edgecolor())
        if isinstance(p, FancyBboxPatch):
            bs = p.get_boxstyle(); pad = getattr(bs, "pad", 0); rr = getattr(bs, "rounding_size", 0) or 0
            x, y, w, h = p.get_x() - pad, p.get_y() - pad, p.get_width() + 2 * pad, p.get_height() + 2 * pad
            r = "rounded corners=%.1fpt," % (rr * sx * 72 * 0.6) if rr else ""
            out.append("\\draw[%s%s%s%s] %s rectangle %s;" % (r, lw(p.get_linewidth()), ",draw=" + ec if ec else "", (",fill=" + fc) if fc else "", P(x, y), P(x + w, y + h)) + dash(p.get_linestyle()).replace(",", "", 0))
        elif isinstance(p, Rectangle):
            x, y, w, h = p.get_x(), p.get_y(), p.get_width(), p.get_height()
            out.append("\\draw[%s,draw=%s%s] %s rectangle %s;" % (lw(p.get_linewidth()), ec or "black", (",fill=" + fc) if fc else "", P(x, y), P(x + w, y + h)))
        elif isinstance(p, Polygon):
            pts = p.get_xy()
            out.append("\\draw[%s,draw=%s%s] %s -- cycle;" % (lw(p.get_linewidth()), ec or "black", (",fill=" + fc) if fc else "", " -- ".join(P(a, b) for a, b in pts[:-1] if True)))
        elif isinstance(p, FancyArrowPatch):
            (xa, ya), (xb, yb) = p._posA_posB
            st = p.get_arrowstyle().__class__.__name__
            opts = "latex-" if False else "-latex"
            if "BracketA" in st or st == "CurveAB": opts = "latex-latex"
            rad = 0
            cs = p.get_connectionstyle()
            rad = getattr(cs, "rad", 0) or 0
            bend = ("to[bend left=%d]" % int(rad * 60)) if rad > 0 else (("to[bend right=%d]" % int(-rad * 60)) if rad < 0 else "--")
            out.append("\\draw[%s,%s,draw=%s%s] %s %s %s;" % (opts, lw(p.get_linewidth()), ec or "black", dash(p.get_linestyle()), P(xa, ya), bend, P(xb, yb)))
    for l in ax.lines:
        xs, ys = l.get_xdata(), l.get_ydata()
        out.append("\\draw[%s,draw=%s%s] %s;" % (lw(l.get_linewidth()), col(l.get_color()) or "black", dash(l.get_linestyle()), " -- ".join(P(a, b) for a, b in zip(xs, ys))))
    for t in texts:
        if isinstance(t, Annotation) and t.arrow_patch is not None:
            (xa, ya) = t.xyann; (xb, yb) = t.xy
            ap = t.arrow_patch
            out.append("\\draw[-latex,%s,draw=%s] %s -- %s;" % (lw(ap.get_linewidth()), col(ap.get_edgecolor()) or "black", P(xa, ya), P(xb, yb)))
            if not t.get_text().strip(): continue
        txt = t.get_text()
        if not txt.strip(): continue
        x, y = (t.get_position() if t is not ax.title else ((x0 + x1) / 2, y1))
        if t is ax.title: y = y1 + (y1 - y0) * 0.03
        fs = t.get_fontsize() * s
        style = ""
        if t.get_fontweight() in ("bold", 700): style += "\\bfseries"
        if t.get_style() == "italic": style += "\\itshape"
        ha = t.get_ha(); va = t.get_va()
        anch = {"center": "", "left": "west", "right": "east"}.get(ha, "")
        vv = {"center": "", "top": "north", "bottom": "south", "baseline": "south"}.get(va, "")
        a = (vv + " " + anch).strip()
        a = ("anchor=" + a + ",") if a else ""
        tc = col(t.get_color()) or "black"
        out.append("\\node[%salign=%s,text=%s,font=\\fontsize{%.1fpt}{%.1fpt}\\selectfont%s,inner sep=0pt] at %s {%s};" % (
            a, "left" if ha == "left" else ("right" if ha == "right" else "center"), tc, fs, fs * 1.18, style, P(x, y), esc(txt)))
    body = "\n".join(out)
    tex = "\\begin{tikzpicture}[line cap=round,line join=round]\n" + body + "\n\\end{tikzpicture}\n"
    (OUT / (name.replace(".png", "") + ".tex")).write_text(tex)
    (OUT / (name.replace(".png", "") + ".meta")).write_text("land" if land else "port")
    return True
