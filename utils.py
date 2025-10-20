"""
Utility functions for the FinDoc Demo RAG application
"""
import shutil
import os
import hashlib
import uuid
import streamlit as st
from config import CHROMA_PERSIST_DIR


def clear_chroma_db():
    """Clear ChromaDB data directory for fresh start"""
    if os.path.exists(CHROMA_PERSIST_DIR):
        shutil.rmtree(CHROMA_PERSIST_DIR)
        print("Cleared existing ChromaDB data for fresh start")


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'processed_file' not in st.session_state:
        st.session_state.processed_file = None
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'graph_instance' not in st.session_state:
        st.session_state.graph_instance = None
    if 'db_cleared' not in st.session_state:
        st.session_state.db_cleared = False
    # Multi-file support
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'processed_files_key' not in st.session_state:
        st.session_state.processed_files_key = None
    if 'processed_file_ids' not in st.session_state:
        st.session_state.processed_file_ids = set()
    # UI interaction state
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'pending_question' not in st.session_state:
        st.session_state.pending_question = None
    if 'eval_requested' not in st.session_state:
        st.session_state.eval_requested = False
    if 'eval_status' not in st.session_state:
        st.session_state.eval_status = ''
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'home'
    if 'eval_results' not in st.session_state:
        st.session_state.eval_results = None
    if 'eval_completed_at' not in st.session_state:
        st.session_state.eval_completed_at = None


def get_file_key(uploaded_file):
    """Generate unique key for uploaded file"""
    if uploaded_file is None:
        return None
    return f"{uploaded_file.name}_{uploaded_file.size}"


def compute_file_hash(uploaded_file) -> str:
    """Compute a stable content hash for an uploaded file (SHA256)."""
    data = uploaded_file.getvalue() if hasattr(uploaded_file, 'getvalue') else uploaded_file.read()
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def get_files_key(uploaded_files) -> str:
    """Compute a combined key for multiple files based on content hashes (stable ordering)."""
    if not uploaded_files:
        return ""
    hashes = [compute_file_hash(f) for f in uploaded_files]
    hashes.sort()
    return "|".join(hashes)


def get_session_collection_name(base_name: str) -> str:
    """Build a session-scoped collection name for Chroma."""
    sid = st.session_state.get('session_id') or "default"
    return f"{base_name}-{sid}"


def format_file_size(size_bytes):
    """Format file size in human-readable format"""
    if size_bytes >= 1024 * 1024:
        size_mb = size_bytes / (1024 * 1024)
        return f"{size_mb:.2f} MB"
    elif size_bytes >= 1024:
        size_kb = size_bytes / 1024
        return f"{size_kb:.1f} KB"
    else:
        return f"{size_bytes} bytes"
