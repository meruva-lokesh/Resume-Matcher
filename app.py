"""
Gradio Resume <-> Job Description Matcher (single-file)

- Upload PDF / DOCX (processed in-memory)
- Two embedding modes: sbert (fast) and bert (CLS)
- Keyword analysis + section-aware suggestions
- Avoids blocking downloads at import/startup
"""

import io
import os
import re
import tempfile
import traceback
from typing import Tuple, Dict

import fitz            # PyMuPDF
import docx            # python-docx

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import gradio as gr

# Lazy imports for heavy ML libs
_TRANSFORMS_LOADED = False
_sentence_transformer = None
_bert_tokenizer = None
_bert_model = None

# --------------------------
# Built-in stopwords (avoid nltk.download at startup)
# --------------------------
# Compact English stopword set sufficient for resume preprocessing.
EN_STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and","any","are","as",
    "at","be","because","been","before","being","below","between","both","but","by",
    "could","did","do","does","doing","down","during","each","few","for","from","further",
    "had","has","have","having","he","her","here","hers","herself","him","himself","his",
    "how","i","if","in","into","is","it","its","itself","just","me","more","most","my",
    "myself","no","nor","not","now","of","off","on","once","only","or","other","ought","our",
    "ours","ourselves","out","over","own","same","she","should","so","some","such","than",
    "that","the","their","theirs","them","themselves","then","there","these","they","this",
    "those","through","to","too","under","until","up","very","was","we","were","what","when",
    "where","which","while","who","whom","why","with","would","you","your","yours","yourself",
    "yourselves"
}

# --------------------------
# Utilities: text extraction
# --------------------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = [p.get_text("text") for p in doc]
        doc.close()
        return "\n".join(p for p in pages if p)
    except Exception as e:
        return f"[Error reading PDF: {e}]"

def extract_text_from_docx_bytes(docx_bytes: bytes) -> str:
    try:
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp.write(docx_bytes)
            tmp.flush()
            tmp_name = tmp.name
        doc = docx.Document(tmp_name)
        paragraphs = [p.text for p in doc.paragraphs if p.text]
        # remove the temporary file
        try:
            os.remove(tmp_name)
        except Exception:
            pass
        return "\n".join(paragraphs)
    except Exception as e:
        return f"[Error reading DOCX: {e}]"

def extract_text_from_fileobj(file_obj) -> Tuple[str, str]:
    """
    Robustly handle multiple shapes Gradio might give:
      - a path string (e.g. "C:\\...\\tempfile.pdf")
      - a file-like object with .read()
      - a dict-like object with keys 'name' and 'data' (or 'file_name'/'data')
      - raw bytes

    Returns: (extracted_text, filename)
    On error, returns an error string as the text (so the caller can show it).
    """
    fname = "uploaded_file"
    raw_bytes = None

    try:
        # 1) If Gradio gives a plain path string
        if isinstance(file_obj, str):
            fname = os.path.basename(file_obj)
            with open(file_obj, "rb") as f:
                raw_bytes = f.read()

        # 2) If it's bytes already
        elif isinstance(file_obj, (bytes, bytearray)):
            raw_bytes = bytes(file_obj)

        # 3) If it's a dict-like (common in some Gradio versions)
        elif isinstance(file_obj, dict):
            if "name" in file_obj:
                fname = file_obj.get("name") or fname
            elif "file_name" in file_obj:
                fname = file_obj.get("file_name") or fname

            data = file_obj.get("data") or file_obj.get("content") or None
            if isinstance(data, str):
                raw_bytes = data.encode()
            elif isinstance(data, (bytes, bytearray)):
                raw_bytes = bytes(data)
            else:
                for v in file_obj.values():
                    if isinstance(v, (bytes, bytearray)):
                        raw_bytes = bytes(v)
                        break

        # 4) If it's a file-like object (has .read())
        elif hasattr(file_obj, "read"):
            try:
                if hasattr(file_obj, "seek"):
                    try:
                        file_obj.seek(0)
                    except Exception:
                        pass
                raw_bytes = file_obj.read()
                if isinstance(raw_bytes, str):
                    raw_bytes = raw_bytes.encode()
                fname = getattr(file_obj, "name", fname)
            except Exception:
                raw_bytes = None

        else:
            try:
                raw_bytes = bytes(file_obj)
            except Exception:
                raw_bytes = None

        if raw_bytes is None:
            return (f"[Error reading uploaded file: Could not extract bytes from object of type {type(file_obj)}]", fname)

        ext = (fname.split(".")[-1].lower() if "." in fname else "")

        # dispatch based on extension or magic bytes
        if ext == "pdf" or (len(raw_bytes) >= 4 and raw_bytes[:4] == b"%PDF"):
            text = extract_text_from_pdf_bytes(raw_bytes)
            return (text, fname)
        elif ext in ("docx", "doc"):
            text = extract_text_from_docx_bytes(raw_bytes)
            return (text, fname)
        else:
            if len(raw_bytes) >= 4 and raw_bytes[:4] == b"%PDF":
                return (extract_text_from_pdf_bytes(raw_bytes), fname)
            if len(raw_bytes) >= 2 and raw_bytes[:2] == b"PK":
                try:
                    return (extract_text_from_docx_bytes(raw_bytes), fname)
                except Exception:
                    pass
            try:
                return (raw_bytes.decode("utf-8", errors="ignore"), fname)
            except Exception as e:
                return (f"[Error decoding uploaded file bytes: {e}]", fname)

    except Exception as exc:
        return (f"[Error reading uploaded file: {exc}\n{traceback.format_exc()}]", fname)


# --------------------------
# Text preprocessing
# --------------------------
def preprocess_text(text: str, remove_stopwords: bool = True) -> str:
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    words = t.split()
    if remove_stopwords:
        words = [w for w in words if w not in EN_STOPWORDS]
    return " ".join(words)

# --------------------------
# Embedding helpers (lazy)
# --------------------------
def _lazy_load_transformers():
    global _TRANSFORMS_LOADED, _sentence_transformer, _bert_tokenizer, _bert_model
    if _TRANSFORMS_LOADED:
        return
    try:
        # import here to avoid heavy imports at startup if not used
        from sentence_transformers import SentenceTransformer
        from transformers import BertTokenizer, BertModel
        import torch

        _sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
        _bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        _bert_model = BertModel.from_pretrained("bert-base-uncased")
        _bert_model.eval()
        _TRANSFORMS_LOADED = True
    except Exception as e:
        # propagate load error later when user selects that mode
        raise RuntimeError(f"Failed to load transformer models: {e}")

def get_sentence_embedding(text: str, mode: str = "sbert") -> np.ndarray:
    if mode == "sbert":
        if not _TRANSFORMS_LOADED or _sentence_transformer is None:
            _lazy_load_transformers()
        return _sentence_transformer.encode([text], show_progress_bar=False)
    elif mode == "bert":
        if not _TRANSFORMS_LOADED or _bert_tokenizer is None:
            _lazy_load_transformers()
        import torch
        tokens = _bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            out = _bert_model(**tokens)
            cls = out.last_hidden_state[:, 0, :].numpy()  # (1, dim)
        return cls
    else:
        raise ValueError("Unsupported mode")

def calculate_similarity(resume_text: str, job_text: str, mode: str = "sbert") -> float:
    r_emb = get_sentence_embedding(resume_text, mode=mode)
    j_emb = get_sentence_embedding(job_text, mode=mode)
    sim = cosine_similarity(r_emb, j_emb)[0][0]
    return float(np.round(sim * 100, 2))

# --------------------------
# Keyword analysis
# --------------------------
DEFAULT_KEYWORDS = {
    "skills": {"python", "nlp", "java", "sql", "tensorflow", "pytorch", "docker", "git"},
    "concepts": {"machine", "learning", "data", "analysis", "nlp", "vision"},
    "roles": {"software", "engineer", "developer", "manager", "scientist", "analyst"},
}

def analyze_resume_keywords(resume_text: str, job_description: str, keywords: Dict = None):
    if keywords is None:
        keywords = DEFAULT_KEYWORDS
    clean_resume = preprocess_text(resume_text)
    clean_job = preprocess_text(job_description)
    resume_words = set(clean_resume.split())
    job_words = set(clean_job.split())
    missing_words = job_words - resume_words
    missing = {}
    for cat, kws in keywords.items():
        missing[cat] = sorted([w for w in kws if w in missing_words])
    # detect headings heuristically
    low = (resume_text or "").lower()
    sections = {
        "skills": "skills" in low,
        "experience": "experience" in low,
        "summary": ("summary" in low) or ("objective" in low),
    }
    suggestions = []
    if any(missing.values()):
        for cat, kws in missing.items():
            for kw in kws:
                if cat == "skills":
                    suggestions.append(f"Add '{kw}' to your Skills section." if sections["skills"] else f"Create a Skills section and include '{kw}'.")
                elif cat == "concepts":
                    suggestions.append(f"Show '{kw}' in your Experience or Projects.")
                elif cat == "roles":
                    suggestions.append(f"Mention the role/title (e.g. '{kw}') in Summary or Experience.")
    else:
        suggestions.append("Great job! Your resume already contains many job-relevant keywords.")
    return missing, suggestions

# --------------------------
# Gradio app logic
# --------------------------
def analyze_resume(file, job_description: str, mode: str, show_cleaned: bool):
    if file is None:
        return 0.0, "No file uploaded.", "", {}, "Please upload a PDF or DOCX resume.", ""

    try:
        resume_text, fname = extract_text_from_fileobj(file)
        if resume_text.strip().startswith("[Error"):
            raise RuntimeError(resume_text)

        cleaned = preprocess_text(resume_text)
        # compute similarity (use cleaned resume and cleaned job)
        sim_pct = calculate_similarity(cleaned, preprocess_text(job_description), mode=mode)

        # verdict
        if sim_pct >= 80:
            verdict = "Excellent match"
        elif sim_pct >= 60:
            verdict = "Good match"
        elif sim_pct >= 40:
            verdict = "Fair — can improve"
        else:
            verdict = "Low match — consider revising"

        missing, suggestions = analyze_resume_keywords(resume_text, job_description)
        suggestions_text = "\n".join(f"- {s}" for s in suggestions)
        cleaned_preview = cleaned if show_cleaned else ""
        raw_preview = "\n".join([ln.strip() for ln in resume_text.splitlines() if ln.strip()][:20])

        return float(sim_pct), f"{sim_pct:.2f}% — {verdict}", cleaned_preview, missing, suggestions_text, raw_preview

    except Exception as e:
        tb = traceback.format_exc()
        return 0.0, f"Error: {e}", "", {}, "An error occurred. Check server logs.", tb

# --------------------------
# Build Gradio UI
# --------------------------
def build_ui():
    with gr.Blocks(title="Resume ↔ Job Matcher") as demo:
        gr.Markdown("# Resume — Job Description Matcher")
        gr.Markdown("Upload a PDF or DOCX resume. Choose embedding mode (fast vs BERT). Files are processed in-memory (no permanent storage).")

        with gr.Row():
            with gr.Column(scale=1):
                file_in = gr.File(label="Upload resume (PDF or DOCX)", file_count="single")
                mode = gr.Radio(choices=["sbert", "bert"], value="sbert", label="Embedding mode")
                job_desc = gr.Textbox(lines=4, label="Job description / Target role", value="Looking for a software engineer skilled in Python, machine learning, and NLP.")
                show_cleaned = gr.Checkbox(label="Show cleaned (preprocessed) resume preview", value=True)
                run_btn = gr.Button("Analyze Resume", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### Match score")
                score_slider = gr.Slider(value=0, minimum=0, maximum=100, step=0.01, interactive=False, label="Similarity (%)")
                score_text = gr.Textbox(label="Score & verdict", interactive=False)

                gr.Markdown("### Cleaned Resume Preview")
                cleaned_preview = gr.Textbox(label="Cleaned preview", interactive=False, lines=8)

                gr.Markdown("### Missing Keywords")
                missing_out = gr.JSON(label="Missing keywords")

                gr.Markdown("### Suggestions")
                suggestions_out = gr.Textbox(label="Suggestions", interactive=False, lines=8)

                gr.Markdown("### Extracted resume (first lines)")
                raw_preview = gr.Textbox(label="Resume preview", interactive=False, lines=8)

        run_btn.click(
            analyze_resume,
            inputs=[file_in, job_desc, mode, show_cleaned],
            outputs=[score_slider, score_text, cleaned_preview, missing_out, suggestions_out, raw_preview],
        )

        gr.Markdown("---\nBuilt with Gradio. First run may download transformer models (sbert/bert).")

    return demo

if __name__ == "__main__":
    demo = build_ui()
    # For local dev: set share=True for a temporary public URL
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
