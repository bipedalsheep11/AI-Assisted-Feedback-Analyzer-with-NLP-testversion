# backend/utils/document_reader.py
# ─────────────────────────────────────────────────────────────────
# Unified document loader. Supports PDF, DOCX, CSV, and XLSX.
# Returns a dict with 'raw_text' and/or 'dataframe' depending
# on the file type.
# ─────────────────────────────────────────────────────────────────

from pathlib import Path
import pandas as pd
import io


def extract_text_from_pdf(file) -> str:
    """
    Extract all text from a PDF file object.

    Parameters
    ----------
    file : file-like object or bytes — the PDF content

    Returns
    -------
    str — extracted text with page markers
    """
    try:
        import fitz  # PyMuPDF
        if hasattr(file, "read"):
            file.seek(0)
            doc = fitz.open(stream=file.read(), filetype="pdf")
        else:
            doc = fitz.open(stream=file, filetype="pdf")
        pages = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(f"[Page {i}]\n{text}")
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        return "[PyMuPDF not installed — install with: pip install pymupdf]"
    except Exception as e:
        return f"[PDF extraction error: {e}]"


def extract_text_from_docx(file) -> str:
    """
    Extract all text from a Word .docx file, preserving headings.
    """
    try:
        from docx import Document
        doc   = Document(file)
        parts = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            if para.style.name.startswith("Heading"):
                parts.append(f"\n## {text}")
            else:
                parts.append(text)
        return "\n".join(parts)
    except ImportError:
        return "[python-docx not installed — install with: pip install python-docx]"
    except Exception as e:
        return f"[DOCX extraction error: {e}]"


def load_file(file, filename: str) -> dict:
    """
    Unified entry point — detects file type from extension and
    routes to the correct extractor.

    Parameters
    ----------
    file     : file-like object (from Streamlit st.file_uploader)
    filename : str — original filename, used to detect extension

    Returns
    -------
    dict with keys:
      'raw_text'   : str | None
      'dataframe'  : pd.DataFrame | None
      'file_type'  : str — 'pdf', 'docx', 'csv', 'xlsx'
      'page_count' : int | None (PDF only)
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        text = extract_text_from_pdf(file)
        return {
            "raw_text":   text,
            "dataframe":  None,
            "file_type":  "pdf",
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

    elif ext == ".csv":
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
        df = pd.read_excel(file)
        return {
            "raw_text":   None,
            "dataframe":  df,
            "file_type":  "xlsx",
            "page_count": None,
        }

    elif ext == ".txt":
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        return {
            "raw_text":   content,
            "dataframe":  None,
            "file_type":  "txt",
            "page_count": None,
        }

    else:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            "Supported formats: PDF, DOCX, TXT, CSV, XLSX"
        )
