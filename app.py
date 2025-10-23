
"""
FinDoc Demo with LangGraph

This is the main Streamlit application that demonstrates how to build a RAG system
using LangGraph for workflow management. The application handles document uploads,
processes questions, and generates answers using a LangGraph-orchestrated pipeline.

Key components:
- Document processing and chunking
- LangGraph workflow for RAG operations
- Question answering with fallback to online search
- Evaluation and quality assessment
- Real-time user interface

This implementation shows practical patterns for building RAG applications with
LangGraph, including state management, conditional routing, and error handling.
Good for understanding how LangGraph works with RAG systems.
"""
import streamlit as st

# Local imports
from config import QUESTION_PLACEHOLDER
from utils import clear_chroma_db, initialize_session_state
from ui_components import (
    setup_page_config, render_header, render_sidebar,
    render_upload_section, render_upload_placeholder,
    render_question_section
)
from document_loader import MultiModalDocumentLoader
from document_processor import DocumentProcessor
from rag_workflow import RAGWorkflow
from evaluation.harness import ensure_eval_index, run_eval
import os, json, time

# Initialize components
document_loader = MultiModalDocumentLoader()
document_processor = DocumentProcessor(document_loader)
rag_workflow = RAGWorkflow()


def _render_metrics_summary_table(values_by_label):
    """Render Metrics summary table with exactly two rows; Value may be empty."""
    import pandas as pd
    labels = [
        "Document Relevance",
        "Avg. Doc Relevance",
    ]
    rows = [[label, values_by_label.get(label, "")] for label in labels]
    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    st.table(df)


def _render_answer_sources_metrics(result, answer_placeholder, sources_placeholder, metrics_placeholder):
    """Render Answer, Sources, and Metrics from a result into provided placeholders."""
    # Answer
    with answer_placeholder.container():
        st.markdown("### Answer")
        answer_text = result.get('solution', '') if result else ''
        if answer_text:
            st.success(answer_text)
        else:
            st.caption("No answer yet.")

    # Sources
    with sources_placeholder.container():
        st.markdown("### Sources")
        documents = result.get('documents') if result else None
        answer_text = result.get('solution', '') if result else ''
        import re as _re
        citation_nums = []
        try:
            nums = _re.findall(r"\[(\d+)\]", answer_text or '')
            citation_nums = sorted(set(int(n) for n in nums))
        except Exception:
            citation_nums = []
        if documents and citation_nums:
            for n in citation_nums:
                if 1 <= n <= len(documents):
                    doc = documents[n - 1]
                    meta = getattr(doc, 'metadata', {}) or {}
                    name = meta.get('original_filename') or meta.get('file_name') or meta.get('source') or 'Document'
                    page = meta.get('page')
                    chunk_id = meta.get('chunk_id')
                    header = f"Source — {name}"
                    details = []
                    if isinstance(page, int):
                        details.append(f"page {page + 1}")
                    if chunk_id is not None:
                        details.append(f"chunk {chunk_id}")
                    if details:
                        header += " (" + ", ".join(details) + ")"
                    st.markdown(f"**{header}**")
                    content = (doc.page_content or '').strip()
                    preview_len = 600
                    st.write(content[:preview_len] + ('...' if len(content) > preview_len else ''))
        else:
            st.caption("No sources yet.")

    # Metrics
    with metrics_placeholder.container():
        st.markdown("### Metrics")
        # Build values for summary table
        values = {}
        if result:
            if 'document_evaluations' in result and result['document_evaluations']:
                evaluations = result['document_evaluations']
                try:
                    relevant_count = sum(1 for eval in evaluations if getattr(eval, 'score', '').lower() == 'yes')
                    total_count = len(evaluations)
                    values["Document Relevance"] = f"{relevant_count}/{total_count} relevant"
                    if hasattr(evaluations[0], 'relevance_score'):
                        avg_score = sum(eval.relevance_score for eval in evaluations) / len(evaluations)
                        values["Avg. Doc Relevance"] = f"{avg_score:.2f}"
                except Exception:
                    pass
        _render_metrics_summary_table(values)
        if result:
            import pandas as pd
            with st.expander("Detailed Evaluation Results"):
                if 'document_evaluations' in result and result['document_evaluations']:
                    st.markdown("**Document Evaluation Details:**")
                    eval_data = []
                    for i, eval in enumerate(result['document_evaluations']):
                        row = [f"Document {i+1}", getattr(eval, 'score', '')]
                        row.append(f"{getattr(eval, 'relevance_score', float('nan')):.2f}" if hasattr(eval, 'relevance_score') else "N/A")
                        cov = getattr(eval, 'coverage_assessment', None)
                        row.append(cov[:50] + "..." if isinstance(cov, str) and len(cov) > 50 else (cov or "N/A"))
                        eval_data.append(row)
                    if eval_data:
                        eval_df = pd.DataFrame(eval_data, columns=["Document", "Score", "Relevance", "Coverage"])
                        st.dataframe(eval_df, use_container_width=True)
                reasoning_data = []
                if 'question_relevance_score' in result and hasattr(result['question_relevance_score'], 'reasoning'):
                    reasoning_data.append(["Question Relevance", result['question_relevance_score'].reasoning])
                if 'document_relevance_score' in result and hasattr(result['document_relevance_score'], 'reasoning'):
                    reasoning_data.append(["Document Relevance", result['document_relevance_score'].reasoning])
                if reasoning_data:
                    st.markdown("**Evaluation Reasoning:**")
                    reasoning_df = pd.DataFrame(reasoning_data, columns=["Evaluation Type", "Reasoning"])
                    st.dataframe(reasoning_df, use_container_width=True)


def handle_question_processing(question, answer_placeholder, sources_placeholder, metrics_placeholder):
    """Process a question, store result to session, and render into placeholders with spinner in Answer area."""
    print(f"Processing question: {question}")
    with answer_placeholder.container():
        st.markdown("### Answer")
        with st.spinner('Analyzing your question and retrieving relevant information...'):
            result = rag_workflow.process_question(question)
    # Persist and render
    st.session_state.last_result = result
    _render_answer_sources_metrics(result, answer_placeholder, sources_placeholder, metrics_placeholder)


def handle_user_interaction(uploaded_files, answer_placeholder, sources_placeholder, metrics_placeholder):
    """Handle user interactions for Q&A (multi-file), avoiding duplicate Query rendering."""
    if not uploaded_files:
        return

    if st.session_state.get('processing'):
        # During processing, do NOT render the question again to avoid duplicate inputs.
        pending_q = (st.session_state.get('pending_question') or '').strip()
        try:
            if pending_q:
                handle_question_processing(pending_q, answer_placeholder, sources_placeholder, metrics_placeholder)
            else:
                with answer_placeholder.container():
                    st.markdown("### Answer")
                    st.warning("Please enter a question before clicking Ask.")
        finally:
            # Reset processing state and allow new input
            st.session_state.processing = False
            st.session_state.pending_question = None
            try:
                st.session_state.question_input = ""
            except Exception:
                pass
            # Immediately rerun so Query section re-renders enabled in the same interaction
            st.rerun()


def render_evaluation_view():
    """Render evaluation dataset preview, run if needed, and display cached results."""
    st.markdown("## Evaluation Dataset")
    # Files
    docs_dir = os.path.join("evaluation", "docs")
    files = []
    try:
        for name in sorted(os.listdir(docs_dir)):
            p = os.path.join(docs_dir, name)
            if os.path.isfile(p):
                size_kb = os.path.getsize(p) / 1024.0
                files.append((name, size_kb))
    except Exception:
        pass
    if files:
        st.markdown("**Files loaded:**")
        link_map = {
            "NVDA-Q4FY25-CFO-Commentary.pdf": "https://drive.google.com/file/d/1X819VtvKVFfCwZcF5_UcHO7kNnzHMET4/view?usp=drive_link",
            "chase-report.pdf": "https://drive.google.com/file/d/1ew_o6xf8TZXq8cYSkfVYjYsd8dZZfGK2/view?usp=drive_link",
        }
        st.markdown("\n".join([f"- [{n}]({link_map.get(n, '#')}) — {sz:.1f} KB" for n, sz in files]))

    # Questions
    tc_path = os.path.join("evaluation", "test_cases.json")
    try:
        with open(tc_path, "r", encoding="utf-8") as f:
            tcs = json.load(f) or []
    except Exception:
        tcs = []
    if tcs:
        st.markdown("**Query set:**")
        st.write("\n".join([f"{i+1}. {tc.get('question','').strip()}" for i, tc in enumerate(tcs)]))

    st.markdown("---")
    st.markdown("## Evaluation Results")

    status = st.session_state.get('eval_status', 'idle')
    eval_results = st.session_state.get('eval_results')
    if st.session_state.get('eval_requested') or (status == 'running' and not eval_results):
        with st.spinner('Running evaluation on test cases...'):
            saved_retriever = st.session_state.get('retriever')
            try:
                eval_retriever = ensure_eval_index()
                rag_workflow.set_retriever(eval_retriever)
                summary, details = run_eval(rag_workflow)
                st.session_state.eval_results = {'summary': summary, 'details': details}
                st.session_state.eval_status = 'done'
                st.session_state.eval_completed_at = time.time()
                st.session_state.eval_requested = False
            except Exception as e:
                st.session_state.eval_status = 'error'
                st.session_state.eval_requested = False
                st.error(f"Evaluation failed: {e}")
            finally:
                rag_workflow.set_retriever(saved_retriever)
                st.session_state.retriever = saved_retriever
            st.rerun()

    eval_results = st.session_state.get('eval_results')
    if not eval_results:
        st.info("No evaluation results yet.")
        return

    summary = eval_results.get('summary') or {}
    details = eval_results.get('details') or []
    soft = float(summary.get('soft_em_avg', 0.0))
    hit = float(summary.get('retrieval_hit_rate', 0.0))
    c1, c2 = st.columns(2)
    c1.metric("Normalized EM", f"{soft*100:.1f}%")
    c2.metric("Hit@1  (doc-level)", f"{hit*100:.1f}%")

    if details:
        st.markdown("### Per-case Details")
        for i, d in enumerate(details, start=1):
            soft_ok = "✅" if d.get('soft_em') else "❌"
            hit_ok = "✅" if d.get('doc_hit') else "❌"
            with st.expander(f"Case {i}: Normalized EM {soft_ok} · Hit@1 (doc-level) {hit_ok}"):
                st.markdown("**Question**")
                st.write(d.get('question', ''))
                st.markdown("**Actual (Model Answer)**")
                st.write(d.get('actual', ''))
                st.markdown("**Expected (Answer Key) — accepted variants**")
                exp = d.get('expected_answers') or []
                st.write("\n".join([f"- {x}" for x in exp]) if exp else "(none)")
                st.markdown("**Sources**")
                sources = d.get('sources') or []
                if sources:
                    for s in sources:
                        name = s.get('name') or 'Document'
                        snippet = s.get('snippet') or ''
                        st.markdown(f"- {name}")
                        st.write(snippet)
                else:
                    st.write("(none)")
                st.markdown("**Expected Docs**")
                expected_docs = [os.path.basename(x) for x in (d.get('expected_docs') or [])]
                if expected_docs:
                    st.write("\n".join([f"- {x}" for x in expected_docs]))
                else:
                    st.write("(none)")
                st.markdown("**Document Matching**")
                retrieved_names = [os.path.basename(s.get('name','')) for s in (sources or []) if s.get('name')]
                matched = sorted(set(expected_docs) & set(retrieved_names))
                missing = sorted(set(expected_docs) - set(retrieved_names))
                st.write(f"Matched: {matched if matched else '[]'} | Missing: {missing if missing else '[]'}")

def main():
    """Main application function"""
    # Initialize session state and clear DB only once
    initialize_session_state()
    
    # Clear ChromaDB only on first run
    if 'db_cleared' not in st.session_state:
        clear_chroma_db()
        st.session_state.db_cleared = True
        print("ChromaDB cleared on app startup")
    
    # Setup page and render UI
    setup_page_config()
    render_sidebar(document_loader)

    # Route to eval view
    if st.session_state.get('current_view', 'home') == 'eval':
        render_evaluation_view()
        return
    
    # Handle file upload (multi-file)
    uploaded_files = render_upload_section(document_loader)

    # Process uploaded files
    if uploaded_files:
        retriever = document_processor.process_files(uploaded_files)
        if retriever:
            st.session_state.retriever = retriever
            print("Files processed, retriever stored in session state")
        else:
            print("File processing failed - no retriever created")

    # Always render Query section once (avoid duplicates during processing)
    render_question_section(uploaded_files)

    # Prepare placeholders for Answer / Sources / Metrics and render skeletons
    answer_placeholder = st.empty()
    sources_placeholder = st.empty()
    metrics_placeholder = st.empty()

    with answer_placeholder.container():
        st.markdown("### Answer")
        st.caption("No answer yet.")
    with sources_placeholder.container():
        st.markdown("### Sources")
        st.caption("No sources yet.")
    with metrics_placeholder.container():
        st.markdown("### Metrics")
        _render_metrics_summary_table(values_by_label={})

    # Handle user interactions (updates placeholders as needed)
    handle_user_interaction(uploaded_files, answer_placeholder, sources_placeholder, metrics_placeholder)

    # After interactions (post-rerun), render last result if available
    last_result = st.session_state.get('last_result')
    if last_result:
        _render_answer_sources_metrics(last_result, answer_placeholder, sources_placeholder, metrics_placeholder)


if __name__ == "__main__":
    main()
