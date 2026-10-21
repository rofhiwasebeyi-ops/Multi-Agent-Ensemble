"""Reusable Tk building blocks for the results view: a scrollable page, cards,
a confusion-matrix grid, bar/heat-coloured tables and bullet lists.

Everything here is plain tkinter/ttk (no extra dependency). Pages are rebuilt
from scratch on every display, which keeps the code simple and means a stale
widget can never leak into a later run.
"""
import tkinter as tk
from tkinter import ttk

FONT = "Segoe UI"
MONO = "Consolas"

# palette
BG = "#ffffff"
PANEL = "#f8fafc"
BORDER = "#e2e8f0"
TEXT = "#0f172a"
MUTED = "#64748b"
ACCENT = "#1d4ed8"
GOOD, GOOD_BG = "#15803d", "#dcfce7"
WARN, WARN_BG = "#b45309", "#fef3c7"
BAD, BAD_BG = "#b91c1c", "#fee2e2"
INFO, INFO_BG = "#1d4ed8", "#dbeafe"
BAR = "#60a5fa"
BAR_NEG = "#f87171"

TONES = {"good": (GOOD, GOOD_BG), "warn": (WARN, WARN_BG), "bad": (BAD, BAD_BG),
         "info": (INFO, INFO_BG), "muted": (MUTED, PANEL)}


def blend(c1, c2, t):
    """Linear blend of two #rrggbb colours (t=0 -> c1, t=1 -> c2)."""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(round(x + (y - x) * max(0, min(1, t)))) for x, y in zip(a, b))


def heat(value, lo=0.0, hi=1.0, low="#ffffff", high="#3b82f6"):
    """Colour for `value` on a white -> blue ramp (used for matrices)."""
    if hi == lo:
        return low
    return blend(low, high, (value - lo) / (hi - lo))


class ScrollPage(ttk.Frame):
    """A vertically scrollable page. Put content in `.body`. Labels registered
    through `wrap()` re-wrap automatically when the pane is resized."""

    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, background=BG)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.body = tk.Frame(self.canvas, background=BG)
        self._win = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self._wraps = []
        self.body.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self._on_resize)
        for w in (self.canvas, self.body):
            w.bind("<Enter>", self._bind_wheel)
            w.bind("<Leave>", self._unbind_wheel)

    def _on_resize(self, event):
        self.canvas.itemconfigure(self._win, width=event.width)
        for lbl, margin in self._wraps:
            try:
                lbl.configure(wraplength=max(160, event.width - margin))
            except tk.TclError:
                pass

    def wrap(self, label, margin=48):
        self._wraps.append((label, margin))
        width = self.canvas.winfo_width()
        label.configure(wraplength=max(160, (width if width > 1 else 560) - margin))
        return label

    # mouse wheel only while the pointer is over this page
    def _bind_wheel(self, _e):
        self.canvas.bind_all("<MouseWheel>", self._wheel)
        self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-3, "units"))
        self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(3, "units"))

    def _unbind_wheel(self, _e):
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.unbind_all(seq)

    def _wheel(self, e):
        self.canvas.yview_scroll(int(-e.delta / 120) if abs(e.delta) >= 120 else (-1 if e.delta > 0 else 1), "units")

    def clear(self):
        for child in self.body.winfo_children():
            child.destroy()
        self._wraps = []
        self.canvas.yview_moveto(0)


# ---------------------------------------------------------------- primitives
def label(parent, text, size=10, bold=False, color=TEXT, bg=BG, anchor="w", justify="left", mono=False, **kw):
    return tk.Label(parent, text=text, font=(MONO if mono else FONT, size, "bold" if bold else "normal"),
                    fg=color, bg=bg, anchor=anchor, justify=justify, **kw)


def heading(page, text, subtitle=None):
    f = tk.Frame(page.body, bg=BG)
    f.pack(fill="x", padx=16, pady=(16, 4))
    label(f, text, 12, True, TEXT).pack(anchor="w")
    if subtitle:
        page.wrap(label(f, subtitle, 9, color=MUTED), 40).pack(anchor="w", fill="x")
    tk.Frame(f, bg=BORDER, height=1).pack(fill="x", pady=(4, 0))
    return f


def paragraph(page, text, color=TEXT, size=10, padx=16, pady=(4, 2), bold=False):
    lbl = label(page.body, text, size, bold, color)
    page.wrap(lbl, 2 * padx + 8)
    lbl.pack(fill="x", padx=padx, pady=pady)
    return lbl


def banner(page, text, tone="info", title=None):
    """A tinted call-out box with an optional bold title line."""
    fg, bg = TONES[tone]
    box = tk.Frame(page.body, bg=bg, highlightbackground=blend(fg, "#ffffff", 0.6), highlightthickness=1)
    box.pack(fill="x", padx=16, pady=(8, 4))
    if title:
        label(box, title, 10, True, fg, bg).pack(anchor="w", padx=12, pady=(8, 0))
    lbl = label(box, text, 10, False, TEXT, bg)
    page.wrap(lbl, 56)
    lbl.pack(fill="x", padx=12, pady=(4 if title else 8, 8))
    return box


def bullets(page, items, padx=16):
    """items: str or (tone, str). A coloured dot marks each item."""
    for it in items:
        tone, text = it if isinstance(it, tuple) else ("muted", it)
        fg, _ = TONES[tone]
        row = tk.Frame(page.body, bg=BG)
        row.pack(fill="x", padx=padx, pady=2)
        tk.Label(row, text="●", font=(FONT, 8), fg=fg, bg=BG).pack(side="left", anchor="n", padx=(2, 8), pady=(3, 0))
        lbl = label(row, text, 10, False, TEXT)
        page.wrap(lbl, 2 * padx + 40)
        lbl.pack(side="left", fill="x", expand=True)


def empty_state(page, title, text):
    f = tk.Frame(page.body, bg=BG)
    f.pack(fill="both", expand=True, padx=24, pady=60)
    label(f, title, 12, True, MUTED, anchor="center", justify="center").pack(fill="x")
    lbl = label(f, text, 10, False, MUTED, anchor="center", justify="center")
    page.wrap(lbl, 96)
    lbl.pack(fill="x", pady=(6, 0))


def metric_cards(page, cards):
    """cards: [(title, value, description)] laid out as an even row."""
    row = tk.Frame(page.body, bg=BG)
    row.pack(fill="x", padx=16, pady=(8, 4))
    for i, (title, value, desc) in enumerate(cards):
        row.columnconfigure(i, weight=1, uniform="cards")
        c = tk.Frame(row, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        c.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 6, 0))
        label(c, title.upper(), 8, True, MUTED, PANEL).pack(anchor="w", padx=10, pady=(8, 0))
        label(c, value, 20, True, TEXT, PANEL).pack(anchor="w", padx=10)
        d = label(c, desc, 8, False, MUTED, PANEL)
        d.pack(anchor="w", fill="x", padx=10, pady=(0, 8))
        d.configure(wraplength=120)


def confusion_grid(page, tp, fn, fp, tn):
    """2x2 grid; colour = good/bad outcome, number = students, % = share of the row."""
    box = tk.Frame(page.body, bg=BG)
    box.pack(fill="x", padx=16, pady=(8, 4))
    for c in (1, 2):
        box.columnconfigure(c, weight=1, uniform="cm")
    label(box, "", 9, bg=BG).grid(row=0, column=0)
    label(box, "Flagged at-risk", 9, True, MUTED, anchor="center").grid(row=0, column=1, sticky="ew")
    label(box, "Not flagged", 9, True, MUTED, anchor="center").grid(row=0, column=2, sticky="ew")

    def cell(r, c, n, row_total, caption, tone):
        fg, bg = TONES[tone]
        f = tk.Frame(box, bg=bg, highlightbackground=blend(fg, "#ffffff", 0.55), highlightthickness=1)
        f.grid(row=r, column=c, sticky="nsew", padx=3, pady=3)
        label(f, f"{n:,}", 16, True, fg, bg, anchor="center").pack(fill="x", pady=(6, 0))
        share = f"{n / row_total:.0%} of row" if row_total else ""
        label(f, f"{caption}", 9, True, fg, bg, anchor="center").pack(fill="x")
        label(f, share, 8, False, MUTED, bg, anchor="center").pack(fill="x", pady=(0, 6))

    label(box, "Actually\nat-risk", 9, True, MUTED, anchor="e", justify="right").grid(row=1, column=0, sticky="e", padx=(0, 6))
    label(box, "Actually\nnot at-risk", 9, True, MUTED, anchor="e", justify="right").grid(row=2, column=0, sticky="e", padx=(0, 6))
    cell(1, 1, tp, tp + fn, "caught", "good")
    cell(1, 2, fn, tp + fn, "missed", "bad")
    cell(2, 1, fp, fp + tn, "false alarm", "warn")
    cell(2, 2, tn, fp + tn, "correct", "good")


class Table(tk.Frame):
    """A grid of labels. Per column you can ask for:
         kind="text"   plain text (left aligned for column 0, centred otherwise)
         kind="bar"    value in [0,1] drawn as a horizontal bar with the text on top
         kind="delta"  signed number, green/red text
         kind="heat"   cell background shaded by value (lo..hi)
         kind="tag"    coloured pill; `rows` cell is (text, tone)
    Cell values are either str or (str, number) when a number drives bar/heat/delta.
    """

    def __init__(self, parent, cols, rows, kinds=None, widths=None, highlight=None, heat_range=(0, 1),
                 bar_max=1.0, first_col_w=None, zebra=True, bold_first=True, wraps=None):
        wraps = wraps or {}
        super().__init__(parent, bg=BG, highlightbackground=BORDER, highlightthickness=1)
        kinds = kinds or ["text"] * len(cols)
        highlight = highlight or {}
        for c in range(len(cols)):
            self.columnconfigure(c, weight=(3 if c == 0 else 2) if not widths else widths[c])
        for c, name in enumerate(cols):
            h = tk.Label(self, text=name, font=(FONT, 9, "bold"), fg=MUTED, bg=PANEL,
                         anchor="w" if c == 0 else "center", padx=8, pady=6, wraplength=130, justify="center" if c else "left")
            h.grid(row=0, column=c, sticky="nsew")
        for r, row in enumerate(rows, start=1):
            bg = highlight.get(r - 1) or (PANEL if zebra and r % 2 == 0 else BG)
            for c, cell in enumerate(row):
                kind = kinds[c]
                text, num = cell if isinstance(cell, tuple) else (cell, None)
                if kind == "bar" and num is not None:
                    self._bar(r, c, text, num, bg, bar_max)
                    continue
                fg, cbg, bold = TEXT, bg, (c == 0 and bold_first)
                if kind == "delta" and num is not None:
                    fg = GOOD if num > 0.0005 else BAD if num < -0.0005 else MUTED
                    bold = True
                elif kind == "heat" and num is not None:
                    cbg = heat(num, *heat_range)
                    fg = "#ffffff" if (num - heat_range[0]) / ((heat_range[1] - heat_range[0]) or 1) > 0.65 else TEXT
                elif kind == "tag" and num is not None:
                    tfg, tbg = TONES[num]
                    pill = tk.Frame(self, bg=bg)
                    pill.grid(row=r, column=c, sticky="nsew")
                    tk.Label(pill, text=text, font=(FONT, 9, "bold"), fg=tfg, bg=tbg, padx=8, pady=1).pack(pady=5)
                    continue
                lbl = tk.Label(self, text=text, font=(FONT, 10, "bold" if bold else "normal"), fg=fg, bg=cbg,
                               anchor="w" if (c == 0 or c in wraps) else "center", padx=8, pady=5, justify="left",
                               wraplength=wraps.get(c, (first_col_w or 260) if c == 0 else 0))
                lbl.grid(row=r, column=c, sticky="nsew")

    def _bar(self, r, c, text, num, bg, bar_max):
        cell = tk.Canvas(self, height=26, width=60, bg=bg, highlightthickness=0)
        cell.grid(row=r, column=c, sticky="nsew")

        def draw(_e=None, cell=cell):
            cell.delete("all")
            w = max(cell.winfo_width(), 60)
            frac = max(0.0, min(1.0, abs(num) / (bar_max or 1)))
            cell.create_rectangle(6, 6, w - 6, 20, fill=BORDER, width=0)
            if frac > 0:
                cell.create_rectangle(6, 6, 6 + (w - 12) * frac, 20, fill=BAR if num >= 0 else BAR_NEG, width=0)
            cell.create_text(w / 2, 13, text=text, font=(FONT, 9, "bold"), fill=TEXT)
        cell.bind("<Configure>", draw)
        draw()


def table(page, cols, rows, padx=16, **kw):
    t = Table(page.body, cols, rows, **kw)
    t.pack(fill="x", padx=padx, pady=(6, 6))
    return t


def caption(page, text):
    return paragraph(page, text, MUTED, 9, pady=(0, 6))
