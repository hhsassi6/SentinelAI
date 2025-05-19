import fitz  # PyMuPDF
import regex as re2
from statistics import quantiles
from pathlib import Path

# Expressions régulières pour la détection des titres
HEAD_RE_CAPS = re2.compile(r"^[\p{Lu}0-9 &/–-]{6,}$")
HEAD_RE_NUM = re2.compile(r"^\d+(\.\d+)*\s+\S+")

def collect_big_fonts(page):
    sizes = [s["size"]
             for blk in page.get_text("dict")["blocks"]
             for ln in blk.get("lines", [])
             for s in ln.get("spans", [])]
    if not sizes: return set()
    thresh = quantiles(sizes, n=10)[-1]
    return {s for s in sizes if s >= thresh}

def lines_with_fonts(page):
    for blk in page.get_text("dict")["blocks"]:
        for ln in blk.get("lines", []):
            txt = "".join(sp["text"] for sp in ln["spans"]).strip()
            if txt:
                yield txt, max(sp["size"] for sp in ln["spans"])

def split_by_titles(pdf_path):
    doc = fitz.open(pdf_path)
    sections, buf, head = [], [], "INTRODUCTION"
    for page in doc:
        big = collect_big_fonts(page)
        for txt, sz in lines_with_fonts(page):
            looks_head = (
                sz in big or txt.endswith(":") or
                HEAD_RE_CAPS.match(txt) or HEAD_RE_NUM.match(txt)
            ) and len(txt.split()) <= 15
            if looks_head:
                if buf:
                    sections.append((head, "\n".join(buf).strip()))
                    buf = []
                head = txt.rstrip(":")
            else:
                buf.append(txt)
    if buf:
        sections.append((head, "\n".join(buf).strip()))
    return sections

def process_pdf(pdf_path):
    """Traite un PDF et retourne ses sections"""
    return split_by_titles(pdf_path) 