import io
import os
import re
import tempfile
import traceback
from typing import Tuple, Dict

import fitz  # PyMuPDF
import docx  # python-docx

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# --------------------------
# Pre-load all heavy libraries and models at startup.
# --------------------------
print("Starting up: Loading transformer models...")
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
import torch

# Load models into memory once when the application starts
sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()
print("Transformer models loaded successfully.")

# --------------------------
# Built-in stopwords
# --------------------------
EN_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as",
    "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself",
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
        docx_io = io.BytesIO(docx_bytes)
        doc = docx.Document(docx_io)
        paragraphs = [p.text for p in doc.paragraphs if p.text]
        return "\n".join(paragraphs)
    except Exception as e:
        return f"[Error reading DOCX: {e}]"


def extract_text_from_fileobj(file_obj) -> Tuple[str, str]:
    fname = "uploaded_file"
    try:
        fname = os.path.basename(file_obj.name)
        with open(file_obj.name, "rb") as f:
            raw_bytes = f.read()

        ext = fname.split('.')[-1].lower()

        if ext == "pdf":
            return (extract_text_from_pdf_bytes(raw_bytes), fname)
        elif ext == "docx":
            return (extract_text_from_docx_bytes(raw_bytes), fname)
        else:  # Fallback for .txt or other text-based files
            return (raw_bytes.decode("utf-8", errors="ignore"), fname)

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
# Embedding helpers
# --------------------------
def get_sentence_embedding(text: str, mode: str = "sbert") -> np.ndarray:
    if mode == "sbert":
        return sentence_transformer.encode([text], show_progress_bar=False)
    elif mode == "bert":
        tokens = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            out = bert_model(**tokens)
            cls = out.last_hidden_state[:, 0, :].numpy()
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
    "skills": {"python", "nlp", "java", "sql", "tensorflow", "pytorch", "docker", "git", "react", "cloud", "aws",
               "azure"},
    "concepts": {"machine", "learning", "data", "analysis", "nlp", "vision", "agile", "scrum"},
    "roles": {"software", "engineer", "developer", "manager", "scientist", "analyst", "architect"},
}


def analyze_resume_keywords(resume_text: str, job_description: str, keywords: Dict = None):
    if keywords is None:
        keywords = DEFAULT_KEYWORDS
    clean_resume = preprocess_text(resume_text)
    clean_job = preprocess_text(job_description)
    resume_words = set(clean_resume.split())
    job_words = set(clean_job.split())

    missing = {}
    for cat, kws in keywords.items():
        missing_from_cat = [kw for kw in kws if kw in job_words and kw not in resume_words]
        if missing_from_cat:
            missing[cat] = sorted(missing_from_cat)

    low_resume = (resume_text or "").lower()
    sections_present = {
        "skills": "skills" in low_resume,
        "experience": "experience" in low_resume or "employment" in low_resume,
        "summary": "summary" in low_resume or "objective" in low_resume,
    }

    suggestions = []
    if any(missing.values()):
        for cat, kws in missing.items():
            for kw in kws:
                if cat == "skills":
                    suggestions.append(f"Add keyword '{kw}' to your Skills section." if sections_present[
                        "skills"] else f"Consider creating a Skills section to include '{kw}'.")
                elif cat == "concepts":
                    suggestions.append(
                        f"Try to demonstrate your knowledge of '{kw}' in your Experience or Projects section.")
                elif cat == "roles":
                    suggestions.append(f"Align your Summary/Objective to mention the title '{kw}'.")
    else:
        suggestions.append("Great job! Your resume contains many of the keywords found in the job description.")

    return missing, "\n".join(f"- {s}" for s in suggestions)


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

        cleaned_resume = preprocess_text(resume_text)
        cleaned_job = preprocess_text(job_description)

        sim_pct = calculate_similarity(cleaned_resume, cleaned_job, mode=mode)

        if sim_pct >= 80:
            verdict = "Excellent match"
        elif sim_pct >= 60:
            verdict = "Good match"
        elif sim_pct >= 40:
            verdict = "Fair match — can be improved"
        else:
            verdict = "Low match — consider major revisions"

        missing, suggestions_text = analyze_resume_keywords(resume_text, job_description)

        cleaned_preview = cleaned_resume if show_cleaned else "Preview disabled."
        raw_preview = "\n".join([ln.strip() for ln in resume_text.splitlines() if ln.strip()][:15])

        return float(sim_pct), f"{sim_pct:.2f}% — {verdict}", cleaned_preview, missing, suggestions_text, raw_preview

    except Exception as e:
        tb = traceback.format_exc()
        return 0.0, f"Error: {e}", "", {}, "An error occurred. Check server logs for details.", tb


# --------------------------
# Build Gradio UI
# --------------------------
def build_ui():
    # The 'theme' parameter is removed to restore the default Gradio look
    with gr.Blocks(title="Resume ↔ Job Matcher") as demo:
        gr.Markdown("# Resume — Job Description Matcher")
        gr.Markdown(
            "Upload a PDF or DOCX resume, paste a job description, and get an instant analysis of how well they match.")

        with gr.Row():
            with gr.Column(scale=1):
                file_in = gr.File(label="Upload resume (PDF or DOCX)", file_count="single")
                mode = gr.Radio(choices=["sbert", "bert"], value="sbert", label="Analysis Mode",
                                info="SBERT is faster, BERT is more detailed.")
                job_desc = gr.Textbox(lines=8, label="Job Description",
                                      placeholder="Paste the full job description here...")
                show_cleaned = gr.Checkbox(label="Show cleaned (preprocessed) resume preview", value=False)
                run_btn = gr.Button("Analyze Resume", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### Match Score")
                score_slider = gr.Slider(value=0, minimum=0, maximum=100, step=0.01, interactive=False,
                                         label="Similarity (%)")
                score_text = gr.Textbox(label="Score & Verdict", interactive=False)

                gr.Markdown("### Keyword Analysis & Suggestions")
                suggestions_out = gr.Textbox(label="Suggestions for Improvement", interactive=False, lines=5)
                missing_out = gr.JSON(label="Missing Keywords from Job Description")

                with gr.Accordion("Show Previews...", open=False):
                    cleaned_preview = gr.Textbox(label="Cleaned Resume Preview", interactive=False, lines=8)
                    raw_preview = gr.Textbox(label="Raw Extracted Resume (First 15 lines)", interactive=False, lines=8)

        run_btn.click(
            analyze_resume,
            inputs=[file_in, job_desc, mode, show_cleaned],
            outputs=[score_slider, score_text, cleaned_preview, missing_out, suggestions_out, raw_preview],
        )

        gr.Markdown("---")
        gr.Markdown("Built with Gradio. Transformer models are pre-loaded at startup.")

    return demo


if __name__ == "__main__":
    demo = build_ui()
    # "0.0.0.0" is required for deployment on platforms like Hugging Face Spaces.
    demo.launch(server_name="0.0.0.0")