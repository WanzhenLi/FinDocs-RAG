"""
UI components for the FinDocs RAG application
"""
import streamlit as st
import re
import time
from config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT, SIDEBAR_STATE, 
    UPLOAD_PLACEHOLDER_TITLE, UPLOAD_PLACEHOLDER_TEXT
)
from utils import format_file_size


def setup_page_config():
    """Sets up Streamlit page settings"""
    st.set_page_config(
        page_title=PAGE_TITLE, 
        page_icon=PAGE_ICON,
        layout=LAYOUT,
        initial_sidebar_state=SIDEBAR_STATE
    )


def render_header():
    """Shows the main header section"""
    st.title("FinDocs RAG")
    st.markdown("RAG system for financial document analysis, powered by LangGraph")
    st.divider()


def render_sidebar(document_loader):
    """Shows the sidebar with logo/text, navigation, and status."""
    with st.sidebar:
        # Simple title (logo-like) and plain description
        st.markdown("### FinDocs RAG")
        st.write(
            "Advanced retrieval augmented generation system for financial documents including "
            "annual reports, fund documents, and other structured financial materials."
        )

        # Navigation
        st.markdown("---")
        st.markdown("### Navigation")
        # Home on top inside Navigation
        if st.button("Home", use_container_width=True):
            st.session_state.current_view = "home"
            st.session_state.eval_status = ""
            st.session_state.eval_completed_at = None
            st.rerun()

        # Run / View eval actions
        if st.button("Run Evaluation", use_container_width=True):
            st.session_state.eval_requested = True
            st.session_state.eval_status = "running"
            st.session_state.current_view = "eval"
            st.rerun()
        if st.session_state.get("eval_results"):
            if st.button(" View Last Evaluation", use_container_width=True):
                st.session_state.current_view = "eval"
                st.rerun()

        # Status indicator with auto-hide after 5s when completed
        status = st.session_state.get("eval_status", "idle")
        completed_at = st.session_state.get("eval_completed_at")
        if status == "running":
            st.info("Evaluation running...")
        elif status == "done":
            if completed_at and (time.time() - completed_at > 5):
                st.session_state.eval_status = ""
                st.session_state.eval_completed_at = None
            else:
                st.success("✅ Evaluation completed")
        elif status == "error":
            st.error("❌ Evaluation failed")


def render_upload_section(document_loader):
    """Shows the document upload section (multi-file)."""
    st.markdown("### Upload Document")
    
    st.info("Upload financial documents for analysis.")
    
    # File uploader (multi-file)
    uploaded_files = st.file_uploader(
        "Choose files",
        type=document_loader.get_supported_extensions(),
        help="Upload one or more financial documents for analysis.",
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    return uploaded_files


def render_file_analysis(file_info):
    """Shows file analysis metrics"""
    st.markdown("### File Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Filename**")
        st.write(file_info['filename'])
    
    with col2:
        st.markdown("**Size**")
        size_display = format_file_size(file_info['size'])
        st.write(size_display)
    
    with col3:
        st.markdown("**Type**")
        st.write(f".{file_info['extension'].upper()}")
    
    with col4:
        st.markdown("**Status**")
        status_text = "Supported" if file_info['is_supported'] else "Unsupported"
        st.write(status_text)


def render_upload_placeholder():
    """Shows placeholder when no file is uploaded"""
    st.markdown(f"""
    <div style="text-align: center; padding: 3rem; background: #f8fafc; border-radius: 10px; margin: 2rem 0;">
        <h3>{UPLOAD_PLACEHOLDER_TITLE}</h3>
        <p>{UPLOAD_PLACEHOLDER_TEXT}</p>
    </div>
    """, unsafe_allow_html=True)


def render_question_section(uploaded_files):
    """Shows the question input section for multi-file context"""
    st.markdown("---")
    st.markdown("### Query Your Documents")
    
    # Display current files info
    if uploaded_files:
        st.markdown(f"**Current Documents ({len(uploaded_files)}):**")
        names = [f.name for f in uploaded_files]
        st.write("\n".join([f"- {n}" for n in names]))
    else:
        st.markdown("**Current Documents (0):**")
    
    # Question input (form enables Enter-to-submit)
    is_processing = st.session_state.get('processing', False)
    has_docs = bool(st.session_state.get('retriever'))
    with st.form("question_form", clear_on_submit=False):
        col1, col2 = st.columns([4, 1])
        with col1:
            question = st.text_input(
                'Enter your question:', 
                placeholder="What were the key financial highlights in this period?",
                disabled=(not has_docs) or is_processing,
                help="Ask questions about financial metrics, performance, strategies, or any content in your documents",
                key='question_input'
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            submitted = st.form_submit_button(
                "Ask",
                use_container_width=True,
                disabled=(not has_docs) or is_processing,
            )
        if submitted:
            st.session_state.pending_question = question or st.session_state.get('question_input', '')
            st.session_state.processing = True
            # Immediately rerun so the app processes in the same interaction
            st.rerun()
    


def render_answer_section(result):
    """Shows the answer section"""
    st.markdown("### Answer")
    answer_text = result.get('solution', '')
    st.success(answer_text)

    # Extract citation numbers like [1], [2] from the answer
    citation_nums = []
    try:
        nums = re.findall(r"\[(\d+)\]", answer_text)
        citation_nums = sorted(set(int(n) for n in nums))
    except Exception:
        citation_nums = []

    documents = result.get('documents') or []
    if documents and citation_nums:
        st.markdown("---")
        st.markdown("### Sources")

        for n in citation_nums:
            if 1 <= n <= len(documents):
                doc = documents[n - 1]
                meta = getattr(doc, 'metadata', {}) or {}
                name = meta.get('original_filename') or meta.get('file_name') or meta.get('source') or 'Document'
                page = meta.get('page')
                chunk_id = meta.get('chunk_id')
                # Remove numeric label; show file name only
                header = f"Source — {name}"
                details = []
                if isinstance(page, int):
                    details.append(f"page {page + 1}")
                if chunk_id is not None:
                    details.append(f"chunk {chunk_id}")
                if details:
                    header += " (" + ", ".join(details) + ")"

                st.markdown(f"**{header}**")
                # Show a preview of the content
                content = (doc.page_content or '').strip()
                preview_len = 600
                if len(content) > preview_len:
                    content_preview = content[:preview_len] + '...'
                else:
                    content_preview = content
                st.write(content_preview)
            else:
                # Ignore invalid citation numbers silently
                pass
