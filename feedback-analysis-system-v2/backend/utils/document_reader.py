# backend/utils/document_reader.py
# ─────────────────────────────────────────────────────────────────────────────
# Document Reader — Unified File Loader
#
# This module extracts text and structured data from the file types a user
# might upload as context documents or survey data.
#
# Supported types:
#   .pdf   — extracts text page by page using PyMuPDF (fitz)
#   .docx  — extracts text and tables using python-docx
#   .txt   — reads plain text directly
#   .csv   — loads as a pandas DataFrame
#   .xlsx  — loads as a pandas DataFrame
#
# Install dependencies:
#   pip install pymupdf python-docx pandas openpyxl
# ─────────────────────────────────────────────────────────────────────────────

import io
import pandas as pd
from pathlib import Path


# ── PDF extraction ─────────────────────────────────────────────────────────────

# What this function does:
#   Opens a PDF file and extracts all text from it, page by page.
#   Each page's text is prefixed with a [Page N] marker so the LLM can
#   reference specific pages if needed.
#
# Parameters:
#   file  (BytesIO | file-like object) — the PDF file to read.
#                                        In Streamlit, st.file_uploader returns
#                                        a file-like object that works here.
#
# Returns:
#   str — all extracted text, with pages separated by blank lines

def extract_text_from_pdf(file) -> str:
    try:
        import fitz  # PyMuPDF — pip install pymupdf
    except ImportError:
        raise ImportError(
            "PyMuPDF is required to read PDF files.\n"
            "Install it with:  pip install pymupdf"
        )

    # Ensure the file pointer is at the start.
    # .seek(0) moves to byte 0 (the beginning) of the file-like object.
    file.seek(0)

    # fitz.open() with stream= and filetype= accepts a BytesIO object.
    # It reads the raw bytes and parses the PDF structure.
    doc = fitz.open(stream=file.read(), filetype="pdf")

    pages_text = []
    for page_number, page in enumerate(doc, start=1):
        # .get_text("text") extracts the text in reading order (left → right, top → bottom).
        page_text = page.get_text("text")
        if page_text.strip():  # Skip blank or image-only pages.
            pages_text.append(f"[Page {page_number}]\n{page_text}")

    doc.close()

    # "\n\n".join() puts a blank line between each page block.
    return "\n\n".join(pages_text)


# ── DOCX extraction ────────────────────────────────────────────────────────────

# What this function does:
#   Reads a Word .docx file and extracts all paragraph text.
#   Heading paragraphs are marked with "## " so the LLM can understand
#   the document's section structure.
#
# Parameters:
#   file  (BytesIO | file-like object) — the DOCX file to read
#
# Returns:
#   str — all extracted text with heading markers

def extract_text_from_docx(file) -> str:
    try:
        from docx import Document  # python-docx — pip install python-docx
    except ImportError:
        raise ImportError(
            "python-docx is required to read .docx files.\n"
            "Install it with:  pip install python-docx"
        )

    file.seek(0)
    doc = Document(file)

    parts = []
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue  # Skip empty paragraphs.

        # paragraph.style.name returns things like "Heading 1", "Heading 2", "Normal".
        style = paragraph.style.name
        if style.startswith("Heading"):
            parts.append(f"\n## {text}")
        else:
            parts.append(text)

    return "\n".join(parts)


# ── Plain text extraction ──────────────────────────────────────────────────────

def extract_text_from_txt(file) -> str:
    """Read a plain .txt file and return its contents as a string."""
    file.seek(0)
    raw_bytes = file.read()
    # Try UTF-8 first; fall back to latin-1 which accepts all byte values.
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("latin-1")


# ── Unified loader ─────────────────────────────────────────────────────────────

# What this function does:
#   Detects the file type from the filename extension and routes to the
#   appropriate extractor function above.
#
# Parameters:
#   file      (file-like object) — the uploaded file from Streamlit's uploader
#   filename  (str)              — the original filename (e.g. "brief.pdf")
#                                  used to detect the extension
#
# Returns:
#   dict with keys:
#     raw_text   (str | None)          — full extracted text for PDF/DOCX/TXT
#     dataframe  (pd.DataFrame | None) — parsed DataFrame for CSV/XLSX
#     file_type  (str)                 — "pdf", "docx", "txt", "csv", or "xlsx"
#     page_count (int | None)          — number of PDF pages, or None

def load_file(file, filename: str) -> dict:
    # Path(filename).suffix returns the file extension, e.g. ".pdf" or ".CSV".
    # .lower() normalizes it so ".PDF" and ".pdf" both match.
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        text = extract_text_from_pdf(file)
        return {
            "raw_text":   text,
            "dataframe":  None,
            "file_type":  "pdf",
            # Count [Page N] markers to determine page count without re-opening.
            "page_count": text.count("[Page "),
        }

    elif ext == ".docx":
        text = extract_text_from_docx(file)
        return {
            "raw_text":   text,
            "dataframe":  None,
            "file_type":  "docx",
            "page_count": None,
        }

    elif ext == ".txt":
        text = extract_text_from_txt(file)
        return {
            "raw_text":   text,
            "dataframe":  None,
            "file_type":  "txt",
            "page_count": None,
        }

    elif ext == ".csv":
        file.seek(0)
        try:
            df = pd.read_csv(file, encoding="utf-8")
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding="latin-1")
        return {
            "raw_text":   None,
            "dataframe":  df,
            "file_type":  "csv",
            "page_count": None,
        }

    elif ext in (".xlsx", ".xls"):
        file.seek(0)
        df = pd.read_excel(file)
        return {
            "raw_text":   None,
            "dataframe":  df,
            "file_type":  "xlsx",
            "page_count": None,
        }

    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'.\n"
            "Supported formats: PDF, DOCX, TXT, CSV, XLSX"
        )
