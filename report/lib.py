import json, re, textwrap
from pathlib import Path
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor, Emu

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs" / "figs"
NAVY = RGBColor(0x11, 0x11, 0x11)
GREY = RGBColor(0x55, 0x55, 0x55)


def shade(cell, hexcol):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hexcol)
    tcPr.append(shd)


def set_cell_margins(table, top=40, bottom=40, left=70, right=70):
    tblPr = table._tbl.tblPr
    m = OxmlElement("w:tblCellMar")
    for k, v in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        e = OxmlElement(f"w:{k}"); e.set(qn("w:w"), str(v)); e.set(qn("w:type"), "dxa"); m.append(e)
    tblPr.append(m)


def add_field(run, instr):
    for t, txt in (("begin", None), (None, instr), ("end", None)):
        if t:
            e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), t); run._r.append(e)
        else:
            e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt; run._r.append(e)


SKIP = {"f02_kv_bandwidth_bound.png","f03_quant_effect_memory_wall.png","f04_coverage_matrix.png","f07_budget_sweep.png","f08_lambda_sweep.png","f09_lambda_pareto.png"}


class Report:
    def __init__(self, pages=None):
        self.d = Document()
        self.pages = pages or {}
        self.toc = []          # (level, text)
        self.nfig = 0; self.ntab = 0
        self.fig_list = []; self.tab_list = []
        self._setup()

    # ---------------------------------------------------------------- setup
    def _setup(self):
        d = self.d
        s = d.sections[0]
        s.page_width, s.page_height = Inches(8.27), Inches(11.69)   # A4
        s.left_margin = s.right_margin = Inches(1.25)
        s.top_margin = Inches(0.95); s.bottom_margin = Inches(0.9)
        st = d.styles["Normal"]; st.font.name = "Latin Modern Roman"; st.font.size = Pt(11)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Latin Modern Roman")
        st.paragraph_format.space_after = Pt(6); st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; st.paragraph_format.line_spacing = 1.08
        for name, size, before, after in (("Heading 1", 17, 0, 12), ("Heading 2", 13, 14, 5), ("Heading 3", 11, 10, 3)):
            h = d.styles[name]; h.font.name = "Latin Modern Roman"; h.font.size = Pt(size); h.font.bold = True
            h.font.color.rgb = RGBColor(0x10,0x10,0x10)
            h.element.rPr.rFonts.set(qn("w:eastAsia"), "Latin Modern Roman"); h.element.rPr.rFonts.set(qn("w:ascii"), "Latin Modern Roman"); h.element.rPr.rFonts.set(qn("w:hAnsi"), "Latin Modern Roman")
            rf = h.element.rPr.rFonts
            for a in ("w:asciiTheme","w:hAnsiTheme","w:eastAsiaTheme","w:cstheme"):
                if rf.get(qn(a)) is not None: del rf.attrib[qn(a)]
            h.paragraph_format.space_before = Pt(before); h.paragraph_format.space_after = Pt(after)
            h.paragraph_format.keep_with_next = True
        # footer page number
        fp = s.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = fp.add_run(); r.font.size = Pt(9); r.font.color.rgb = GREY; add_field(r, "PAGE")
        hp = s.header.paragraphs[0]; hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r = hp.add_run("ELL305 Termpaper — Ishanvi Prasad (2024EE10406), Sweety (2024MT10535)"); r.font.size = Pt(8.5); r.font.color.rgb = GREY

    # ---------------------------------------------------------------- text
    def h1(self, text, newpage=True):
        if newpage:
            self.d.add_page_break()
        self.toc.append((1, text)); return self.d.add_heading(text, 1)

    def h2(self, text):
        self.toc.append((2, text)); return self.d.add_heading(text, 2)

    def h3(self, text):
        return self.d.add_heading(text, 3)

    def p(self, text, bold_lead=None, italic=False, align=None, size=None, color=None, after=None):
        para = self.d.add_paragraph()
        if bold_lead:
            r = para.add_run(bold_lead + " "); r.bold = True
        self._runs(para, text, italic, size, color)
        if align == "j": para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if align == "c": para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if after is not None: para.paragraph_format.space_after = Pt(after)
        return para

    def _runs(self, para, text, italic=False, size=None, color=None):
        # **bold** and `code` inline markup
        for tok in re.split(r"(\*\*.+?\*\*|`.+?`)", text):
            if not tok: continue
            if tok.startswith("**"): r = para.add_run(tok[2:-2]); r.bold = True
            elif tok.startswith("`"):
                r = para.add_run(tok[1:-1]); r.font.name = "Latin Modern Mono"; r.font.size = Pt(9.5)
                r._r.rPr.rFonts.set(qn("w:eastAsia"), "Latin Modern Mono")
            else: r = para.add_run(tok)
            if italic: r.italic = True
            if size and not tok.startswith("`"): r.font.size = Pt(size)
            if color: r.font.color.rgb = color

    def bullets(self, items, style="List Bullet"):
        for it in items:
            para = self.d.add_paragraph(style=style)
            para.paragraph_format.space_after = Pt(3)
            if isinstance(it, tuple):
                r = para.add_run(it[0] + " "); r.bold = True; self._runs(para, it[1])
            else: self._runs(para, it)

    def numbered(self, items):
        for n, it in enumerate(items, 1):
            para = self.d.add_paragraph()
            pf = para.paragraph_format; pf.left_indent = Inches(0.3); pf.first_line_indent = Inches(-0.3); pf.space_after = Pt(3)
            r = para.add_run(f"{n}.\t"); r.bold = True
            pf.tab_stops.add_tab_stop(Inches(0.3))
            self._runs(para, it)

    def callout(self, title, text, fill="F2F2F2"):
        t = self.d.add_table(rows=1, cols=1); t.alignment = WD_TABLE_ALIGNMENT.CENTER
        c = t.rows[0].cells[0]; shade(c, fill); set_cell_margins(t, 90, 90, 140, 140)
        pp = c.paragraphs[0]; r = pp.add_run(title); r.bold = True; r.font.color.rgb = NAVY
        for para_text in (text if isinstance(text, list) else [text]):
            q = c.add_paragraph(); self._runs(q, para_text, size=10); q.paragraph_format.space_after = Pt(3)
        self.d.add_paragraph().paragraph_format.space_after = Pt(2)

    def eq(self, text):
        para = self.d.add_paragraph(); para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = para.add_run(text); r.font.name = "Cambria Math"; r.font.size = Pt(11); r.italic = True
        para.paragraph_format.space_after = Pt(8)

    # ---------------------------------------------------------------- figures / tables
    def fig(self, name, caption, width=6.2):
        if name in SKIP: return
        path = FIG / name
        assert path.exists(), name
        para = self.d.add_paragraph(); para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.keep_with_next = True; para.paragraph_format.space_after = Pt(2)
        para.add_run().add_picture(str(path), width=Inches(width))
        self.nfig += 1
        cp = self.d.add_paragraph(); cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cp.add_run(f"Figure {self.nfig}. "); r.bold = True; r.font.size = Pt(9.5)
        r2 = cp.add_run(caption.replace("`","")); r2.font.size = Pt(9.5); r2.italic = True
        cp.paragraph_format.space_after = Pt(10)
        self.fig_list.append((self.nfig, caption))

    def table(self, header, rows, caption=None, widths=None, size=8.5, hdr_fill=None, zebra=True):
        if caption:
            self.ntab += 1
            cp = self.d.add_paragraph(); cp.paragraph_format.keep_with_next = True; cp.paragraph_format.space_after = Pt(3)
            r = cp.add_run(f"Table {self.ntab}. "); r.bold = True; r.font.size = Pt(9.5)
            r2 = cp.add_run(caption); r2.italic = True; r2.font.size = Pt(9.5)
            self.tab_list.append((self.ntab, caption))
        t = self.d.add_table(rows=1, cols=len(header)); t.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_cell_margins(t)
        for i, h in enumerate(header):
            c = t.rows[0].cells[i]
            pp = c.paragraphs[0]; r = pp.add_run(str(h)); r.bold = True; r.font.size = Pt(size)
            pp.paragraph_format.space_after = Pt(0)
        for ri, row in enumerate(rows):
            cells = t.add_row().cells
            for i, v in enumerate(row):
                pp = cells[i].paragraphs[0]; pp.paragraph_format.space_after = Pt(0); pp.paragraph_format.line_spacing = 1.0
                self._runs(pp, str(v), size=size)
                if zebra and ri % 2 == 1: shade(cells[i], "FFFFFF")
        # booktabs rules
        def _b(cell, **kw):
            tcPr = cell._tc.get_or_add_tcPr(); b = OxmlElement("w:tcBorders")
            for edge, sz in kw.items():
                e = OxmlElement("w:" + edge); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(sz)); e.set(qn("w:color"), "000000"); b.append(e)
            tcPr.append(b)
        for c in t.rows[0].cells: _b(c, top=12, bottom=6)
        for c in t.rows[-1].cells: _b(c, bottom=12)
        # repeat header row
        trPr = t.rows[0]._tr.get_or_add_trPr(); th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true"); trPr.append(th)
        if widths:
            t.autofit = False
            tot = sum(widths); widths = [w * 5.77 / tot for w in widths]
            for i, w in enumerate(widths): t.columns[i].width = Inches(w)
            for row in t.rows:
                for i, w in enumerate(widths): row.cells[i].width = Inches(w)
        self.d.add_paragraph().paragraph_format.space_after = Pt(4)

    def code(self, text, size=7.2, maxw=112, title=None):
        if title:
            tp = self.d.add_paragraph(); r = tp.add_run(title); r.bold = True; r.font.size = Pt(9.5); r.font.color.rgb = NAVY
            tp.paragraph_format.keep_with_next = True; tp.paragraph_format.space_after = Pt(2)
        out = []
        for ln in text.expandtabs(4).rstrip("\n").split("\n"):
            if len(ln) <= maxw: out.append(ln)
            else:
                ind = len(ln) - len(ln.lstrip())
                w = textwrap.wrap(ln, maxw, subsequent_indent=" " * (ind + 4), break_long_words=True, break_on_hyphens=False, drop_whitespace=False)
                out.extend(w)
        # chunk into paragraphs of 60 lines so pagination is smooth
        for i in range(0, len(out), 60):
            para = self.d.add_paragraph(); pf = para.paragraph_format
            pf.space_after = Pt(0); pf.space_before = Pt(0); pf.line_spacing = 1.0
            r = para.add_run("\n".join(out[i:i + 60])); r.font.name = "Latin Modern Mono"; r.font.size = Pt(size)
            r._r.rPr.rFonts.set(qn("w:eastAsia"), "Latin Modern Mono")
            ppr = para._p.get_or_add_pPr(); shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "F4F4F4"); ppr.append(shd)
        self.d.add_paragraph().paragraph_format.space_after = Pt(4)

    def file_listing(self, rel, title=None, size=7.2):
        p = ROOT / rel
        self.code(p.read_text(), size=size, title=title or rel)

    # ---------------------------------------------------------------- TOC placeholder
    def toc_block(self, entries):
        for lvl, text in entries:
            para = self.d.add_paragraph()
            pf = para.paragraph_format; pf.space_after = Pt(1 if lvl == 2 else 3); pf.space_before = Pt(5 if lvl == 1 else 0)
            pf.left_indent = Inches(0 if lvl == 1 else 0.3)
            pf.tab_stops.add_tab_stop(Inches(5.77), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
            r = para.add_run(text + "\t" + str(self.pages.get(text, "")))
            r.font.size = Pt(10.5 if lvl == 1 else 9.5); r.bold = (lvl == 1)

    def save(self, path): self.d.save(path)
