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
from sklearn.feature_extraction.text import TfidfVectorizer
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
    "yourselves", "resume", "job", "description", "work", "experience", "skill", "skills", "applicant", "application"
}

# --------------------------
# Job Suggestions Database
# --------------------------
JOB_SUGGESTIONS_DB = {
    "Data Scientist": {"python", "sql", "machine", "learning", "tensorflow", "pytorch", "analysis"},
    "Data Analyst": {"sql", "python", "excel", "tableau", "analysis", "statistics"},
    "Backend Developer": {"python", "java", "sql", "docker", "aws", "api", "git"},
    "Frontend Developer": {"react", "javascript", "html", "css", "git", "ui", "ux"},
    "Full-Stack Developer": {"python", "javascript", "react", "sql", "docker", "git"},
    "Machine Learning Engineer": {"python", "tensorflow", "pytorch", "machine", "learning", "docker", "cloud"},
    "Project Manager": {"agile", "scrum", "project", "management", "jira"}
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
        else:
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


def analyze_resume_keywords(resume_text: str, job_description: str):
    clean_resume = preprocess_text(resume_text)
    clean_job = preprocess_text(job_description)
    resume_words = set(clean_resume.split())
    job_words = set(clean_job.split())
    missing = {}
    for cat, kws in DEFAULT_KEYWORDS.items():
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
# Project Section Analysis
# --------------------------
def extract_projects_section(resume_text: str) -> str:
    project_headings = ["projects", "personal projects", "academic projects", "portfolio"]
    end_headings = [
        "skills", "technical skills", "experience", "work experience",
        "education", "awards", "certifications", "languages", "references"
    ]
    lines = resume_text.split('\n')
    start_index = -1
    end_index = len(lines)
    for i, line in enumerate(lines):
        cleaned_line = line.strip().lower()
        if cleaned_line in project_headings:
            start_index = i
            break
    if start_index == -1:
        return "Could not automatically identify a 'Projects' section in this resume."
    for i in range(start_index + 1, len(lines)):
        cleaned_line = line.strip().lower()
        if len(cleaned_line.split()) < 4 and cleaned_line in end_headings:
            end_index = i
            break
    project_section_lines = lines[start_index:end_index]
    return "\n".join(project_section_lines)


def analyze_projects_fit(projects_text: str, job_description_text: str, mode: str) -> str:
    if projects_text.startswith("Could not"):
        return "Cannot analyze project fit as no projects section was found."

    cleaned_projects = preprocess_text(projects_text)
    cleaned_job = preprocess_text(job_description_text)

    if not cleaned_projects:
        return "Projects section is empty or contains no relevant text to analyze."

    project_fit_score = calculate_similarity(cleaned_projects, cleaned_job, mode=mode)

    if project_fit_score >= 75:
        verdict = f"<p style='color:green;'>✅ **Highly Relevant ({project_fit_score:.2f}%):** The projects listed are an excellent match for this job's requirements.</p>"
    elif project_fit_score >= 50:
        verdict = f"<p style='color:limegreen;'>👍 **Relevant ({project_fit_score:.2f}%):** The projects show relevant skills and experience for this role.</p>"
    else:
        verdict = f"<p style='color:orange;'>⚠️ **Moderately Relevant ({project_fit_score:.2f}%):** The projects may not directly align with the key requirements. Consider highlighting different aspects of your work.</p>"

    return verdict


# --------------------------
# Formatting and Suggestion Functions
# --------------------------
def format_missing_keywords(missing: Dict) -> str:
    if not any(missing.values()):
        return "✅ No critical keywords seem to be missing. Great job!"
    output = "### 🔑 Keywords Missing From Your Resume\n"
    for category, keywords in missing.items():
        if keywords:
            output += f"**Missing {category.capitalize()}:** {', '.join(keywords)}\n"
    return output


def suggest_jobs(resume_text: str) -> str:
    resume_words = set(preprocess_text(resume_text).split())
    suggestions = []
    for job_title, required_skills in JOB_SUGGESTIONS_DB.items():
        matched_skills = resume_words.intersection(required_skills)
        if len(matched_skills) >= 3:
            suggestions.append(job_title)
    if not suggestions:
        return "Could not determine strong job matches from the resume. Try adding more specific skills and technologies."
    output = "### 🚀 Job Titles You May Be a Good Fit For\n"
    for job in suggestions:
        output += f"- {job}\n"
    return output


def extract_top_keywords(text: str, top_n: int = 15) -> str:
    if not text.strip():
        return "Not enough text provided."
    try:
        vectorizer = TfidfVectorizer(stop_words=list(EN_STOPWORDS))
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = np.array(vectorizer.get_feature_names_out())
        scores = tfidf_matrix.toarray().flatten()
        top_indices = scores.argsort()[-top_n:][::-1]
        top_keywords = feature_names[top_indices]
        return ", ".join(top_keywords)
    except ValueError:
        return "Could not extract keywords (text may be too short)."


# --------------------------
# Main Gradio app logic
# --------------------------
def analyze_resume(file, job_description: str, mode: str):
    if file is None or not job_description.strip():
        return 0.0, "Please upload a resume and paste a job description.", "", "", "", "", "", "", ""

    try:
        resume_text, fname = extract_text_from_fileobj(file)
        if resume_text.strip().startswith("[Error"):
            raise RuntimeError(resume_text)

        cleaned_resume = preprocess_text(resume_text)
        cleaned_job = preprocess_text(job_description)

        sim_pct = calculate_similarity(cleaned_resume, cleaned_job, mode=mode)

        if sim_pct >= 80:
            verdict = f"<h3 style='color:green;'>✅ Excellent Match ({sim_pct:.2f}%)</h3>"
        elif sim_pct >= 60:
            verdict = f"<h3 style='color:limegreen;'>👍 Good Match ({sim_pct:.2f}%)</h3>"
        elif sim_pct >= 40:
            verdict = f"<h3 style='color:orange;'>⚠️ Fair Match ({sim_pct:.2f}%)</h3>"
        else:
            verdict = f"<h3 style='color:red;'>❌ Low Match ({sim_pct:.2f}%)</h3>"

        missing_dict, suggestions_text = analyze_resume_keywords(resume_text, job_description)

        missing_formatted = format_missing_keywords(missing_dict)
        job_suggestions = suggest_jobs(resume_text)

        projects_section = extract_projects_section(resume_text)
        project_fit_verdict = analyze_projects_fit(projects_section, job_description, mode)

        resume_keywords_text = extract_top_keywords(cleaned_resume)
        jd_keywords_text = extract_top_keywords(cleaned_job)

        return (
            float(sim_pct), verdict, missing_formatted, suggestions_text,
            job_suggestions, projects_section, project_fit_verdict, resume_keywords_text, jd_keywords_text
        )

    except Exception as e:
        tb = traceback.format_exc()
        return 0.0, f"### An Error Occurred\n`{e}`", "", "", "", "", "", "", ""


# --------------------------
# Clear Button Logic
# --------------------------
def clear_inputs():
    return None, "", "sbert", None, None, None, None, None, None, None, None


# --------------------------
# Build Gradio UI
# --------------------------
def build_ui():
    with gr.Blocks(theme=gr.themes.Default(), title="Resume ↔ Job Matcher") as demo:
        gr.Markdown("# 📄 Resume & Job Description Analyzer 🎯")
        gr.Markdown(
            "Upload a resume, paste a job description, and get an instant analysis, keyword suggestions, and potential job matches.")

        with gr.Row():
            with gr.Column(scale=2):
                file_in = gr.File(label="Upload resume (PDF or DOCX)", file_count="single",
                                  file_types=[".pdf", ".docx"])
                job_desc = gr.Textbox(lines=10, label="Job Description",
                                      placeholder="Paste the full job description here...")
                mode = gr.Radio(choices=["sbert", "bert"], value="sbert", label="Analysis Mode",
                                info="SBERT is faster, BERT is more detailed.")
                with gr.Row():
                    clear_btn = gr.Button("Clear")
                    run_btn = gr.Button("Analyze Resume", variant="primary")

            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.TabItem("📊 Analysis & Suggestions"):
                        score_slider = gr.Slider(value=0, minimum=0, maximum=100, step=0.01, interactive=False,
                                                 label="Similarity Score")
                        score_text = gr.Markdown()
                        suggestions_out = gr.Textbox(label="Suggestions to Improve Your Resume", interactive=False,
                                                     lines=5)
                        missing_out = gr.Markdown(label="Keywords Check")

                    with gr.TabItem("🛠️ Project Analysis"):
                        project_fit_out = gr.Markdown(label="Project Fit Verdict")
                        projects_out = gr.Textbox(label="Extracted Projects Section", interactive=False, lines=12)

                    with gr.TabItem("🚀 Job Suggestions"):
                        job_suggestions_out = gr.Markdown(label="Potential Job Roles")

                    with gr.TabItem("🔑 Top Keywords"):
                        resume_keywords_out = gr.Textbox(label="Top Resume Keywords")
                        jd_keywords_out = gr.Textbox(label="Top Job Description Keywords")

        run_btn.click(
            analyze_resume,
            inputs=[file_in, job_desc, mode],
            outputs=[score_slider, score_text, missing_out, suggestions_out, job_suggestions_out, projects_out,
                     project_fit_out, resume_keywords_out,
                     jd_keywords_out],
            show_progress='full'
        )

        clear_btn.click(
            clear_inputs,
            inputs=[],
            outputs=[file_in, job_desc, mode, score_slider, score_text, missing_out, suggestions_out,
                     job_suggestions_out, projects_out, project_fit_out, resume_keywords_out, jd_keywords_out]
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
    #demo.launch(server_name="0.0.0.0")