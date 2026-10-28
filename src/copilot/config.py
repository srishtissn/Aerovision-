"""
Central configuration for the AeroVision Maintenance Copilot.

All secrets come from environment variables / a local `.env` file
(never hard-code keys). See `.env.example`.
"""
from __future__ import annotations

import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

try:  # python-dotenv is optional; env vars work without it
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:  # pragma: no cover
    pass

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")

# Generator / agent model (Groq, OpenAI-compatible tool calling)
GROQ_MODEL = os.environ.get("AEROVISION_GROQ_MODEL", "llama-3.3-70b-versatile")
# Vision second-opinion + (default) LLM-as-judge model
GEMINI_MODEL = os.environ.get("AEROVISION_GEMINI_MODEL", "gemini-2.5-flash")

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")
EVAL_OUT_DIR = os.path.join(ROOT, "output", "eval")
SAMPLE_DIR = os.path.join(ROOT, "data", "sample_images")
CMAPSS_DIR = os.path.join(ROOT, "data", "cmapss")

# Sentence-transformers model for RAG embeddings (falls back to TF-IDF
# automatically if sentence-transformers / faiss are not installed).
EMBED_MODEL = os.environ.get("AEROVISION_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

MAX_REPAIR_ATTEMPTS = 1  # LLM re-tries (with verifier feedback) before template fallback
