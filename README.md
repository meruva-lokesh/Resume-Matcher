# Resume ↔ Job Description Matcher (Gradio)

A Gradio app that compares an uploaded resume (PDF/DOCX) with a job description using semantic embeddings and simple keyword analysis.

## Features
- Upload PDF or DOCX resumes (processed in-memory)
- Fast semantic similarity (sentence-transformers / all-MiniLM-L6-v2)
- Optional BERT (CLS) mode (slower / heavier)
- Section-aware missing keyword suggestions

## Usage
- Upload your resume file, paste job description, click **Analyze Resume**
- Prefer the default **sbert** mode for faster response and lower memory usage.

## Deployment
This app runs on Hugging Face Spaces (Gradio). See repository `requirements.txt`. No private keys required.

