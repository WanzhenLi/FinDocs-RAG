"""
Lightweight evaluation harness for the Streamlit RAG app.

Responsibilities:
- Ensure a dedicated evaluation Chroma collection exists (lazy build from two PDFs).
- Run a small set of pre-defined test cases and compute simple, reproducible metrics.

Inputs expected (checked into repo by user):
- evaluation/docs/*.pdf            # two small PDFs
- evaluation/test_cases.json       # five test cases with acceptable answers
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHROMA_EVAL_COLLECTION,
    CHROMA_EVAL_DIR,
)
from multimodal_loader import MultiFormatDocumentLoader


# ---------------------------
# Index construction helpers
# ---------------------------

def _eval_collection_has_data() -> bool:
    """Return True if the evaluation collection exists and has vectors."""
    try:
        chroma = Chroma(
            collection_name=CHROMA_EVAL_COLLECTION,
            persist_directory=CHROMA_EVAL_DIR,
            embedding_function=OpenAIEmbeddings(),
        )
        # Count via internal handle when available
        if getattr(chroma, "_collection", None) is not None:
            try:
                return chroma._collection.count() > 0  # type: ignore[attr-defined]
            except Exception:
                pass
        # Fallback: attempt a trivial retrieval (may raise if empty)
        try:
            _ = chroma.similarity_search("test", k=1)
            return True
        except Exception:
            return False
    except Exception:
        return False


def ensure_eval_index() -> Any:
    """Ensure eval index is built; return a retriever over the eval collection.

    Raises helpful exceptions when PDFs or test cases are missing.
    """
    if not _eval_collection_has_data():
        # Load PDFs from evaluation/docs
        docs_dir = os.path.join("evaluation", "docs")
        if not os.path.isdir(docs_dir):
            raise FileNotFoundError(f"Evaluation docs directory not found: {docs_dir}")

        # Collect supported files (pdfs expected)
        loader = MultiFormatDocumentLoader()
        file_paths: List[str] = []
        for name in os.listdir(docs_dir):
            p = os.path.join(docs_dir, name)
            if os.path.isfile(p) and loader.is_supported_format(p):
                file_paths.append(p)

        if not file_paths:
            raise FileNotFoundError("No supported evaluation documents found in evaluation/docs/")

        # Load and split
        documents: List[Document] = loader.load_multiple_documents(file_paths)
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        doc_splits = splitter.split_documents(documents)

        # Build Chroma collection for eval
        chroma = Chroma.from_documents(
            documents=doc_splits,
            collection_name=CHROMA_EVAL_COLLECTION,
            embedding=OpenAIEmbeddings(),
            persist_directory=CHROMA_EVAL_DIR,
        )
        # Persist (best-effort)
        try:
            chroma.persist()
        except Exception:
            pass

    # Open collection and return retriever
    chroma = Chroma(
        collection_name=CHROMA_EVAL_COLLECTION,
        persist_directory=CHROMA_EVAL_DIR,
        embedding_function=OpenAIEmbeddings(),
    )
    return chroma.as_retriever()


# ---------------------------
# Metrics helpers
# ---------------------------

def _strip_punct(text: str) -> str:
    # Python's re does not support \p{P} by default; use a simpler approach
    return re.sub(r"[^\w\s]", "", text)


def _normalize_units(text: str) -> str:
    # Simple unit normalization: billion -> b, millions -> m, etc.
    t = text
    t = re.sub(r"\bbillions?\b", "b", t)
    t = re.sub(r"\bmillions?\b", "m", t)
    t = re.sub(r"\bthousands?\b", "k", t)
    return t


def normalize_text(text: str) -> str:
    t = (text or "").strip().lower()
    t = _strip_punct(t)
    t = re.sub(r"\s+", " ", t)
    t = _normalize_units(t)
    return t


def compute_token_f1(prediction: str, gold: str) -> float:
    """Kept for possible future use; not used in MVP metrics."""
    pred_tokens = set(normalize_text(prediction).split())
    gold_tokens = set(normalize_text(gold).split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _matches_soft(pred_norm: str, expected_norm_list: List[str], regex_list: List[str]) -> bool:
    # substring match
    for ans in expected_norm_list:
        if ans and ans in pred_norm:
            return True
    # regex match
    for pattern in regex_list:
        try:
            if re.search(pattern, pred_norm):
                return True
        except re.error:
            continue
    return False


def _doc_hit(retrieved_docs: List[Document], expected_docs: List[str]) -> bool:
    if not retrieved_docs or not expected_docs:
        return False
    expected_lower = [e.lower() for e in expected_docs]
    for d in retrieved_docs:
        meta = getattr(d, "metadata", {}) or {}
        fname = (meta.get("original_filename") or meta.get("file_name") or meta.get("source") or "").lower()
        for e in expected_lower:
            if e and e in fname:
                return True
    return False


# ---------------------------
# Public API
# ---------------------------

def run_eval(rag_workflow) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Execute evaluation over test cases and compute summary + details.

    Returns:
        summary: dict with average metrics
        details: list of per-case dicts
    """
    # Ensure index and get retriever
    eval_retriever = ensure_eval_index()

    # Load test cases
    tc_path = os.path.join("evaluation", "test_cases.json")
    if not os.path.exists(tc_path):
        raise FileNotFoundError(f"Test cases not found: {tc_path}")
    with open(tc_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    # Per-case evaluation
    details: List[Dict[str, Any]] = []
    soft_sum = 0
    doc_hit_sum = 0

    for case in test_cases:
        question = case.get("question", "")
        expected_answers: List[str] = case.get("expected_answers", [])
        expected_docs: List[str] = case.get("expected_docs", [])
        keywords: List[str] = case.get("keywords", [])
        regex_any_of: List[str] = case.get("regex_any_of", [])

        # Switch to eval retriever
        rag_workflow.set_retriever(eval_retriever)
        result = rag_workflow.process_question(question)

        actual = (result or {}).get("solution", "")
        retrieved_docs: List[Document] = (result or {}).get("documents", []) or []

        # Normalize answers
        pred_norm = normalize_text(actual)
        expected_norm_list = [normalize_text(a) for a in expected_answers]

        # Normalized EM: substring/regex/keywords any-of
        soft_em = _matches_soft(pred_norm, expected_norm_list, regex_any_of)
        if not soft_em and keywords:
            # any keyword presence (normalized)
            kw_norm = [normalize_text(k) for k in keywords]
            soft_em = any(k and k in pred_norm for k in kw_norm)

        # Doc hit and readable sources (dedup by filename, with snippet)
        doc_hit = _doc_hit(retrieved_docs, expected_docs)
        sources: List[Dict[str, str]] = []
        seen = set()
        for d in retrieved_docs:
            meta = getattr(d, "metadata", {}) or {}
            fname = meta.get("original_filename") or meta.get("file_name") or meta.get("source") or ""
            base = os.path.basename(fname) if fname else ""
            key = base or fname
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            content = (getattr(d, "page_content", "") or "").strip()
            preview = content[:400] + ("..." if len(content) > 400 else "")
            sources.append({"name": key, "snippet": preview})

        soft_sum += 1 if soft_em else 0
        doc_hit_sum += 1 if doc_hit else 0

        details.append(
            {
                "question": question,
                "actual": actual,
                "expected_answers": expected_answers,
                "expected_docs": expected_docs,
                "soft_em": soft_em,
                "doc_hit": doc_hit,
                "sources": sources,
                # Reference scores if present
                "document_relevance_score": (result or {}).get("document_relevance_score"),
                "question_relevance_score": (result or {}).get("question_relevance_score"),
            }
        )

    n = max(1, len(details))
    summary = {
        "soft_em_avg": soft_sum / n,
        "retrieval_hit_rate": doc_hit_sum / n,
    }

    return summary, details
