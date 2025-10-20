"""
Configuration settings for the FinDoc Demo RAG application
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Disable LangSmith tracing if not explicitly enabled
if 'LANGCHAIN_TRACING_V2' not in os.environ:
    os.environ['LANGCHAIN_TRACING_V2'] = 'false'

# RAG Configuration
ENABLE_ONLINE_SEARCH = False  # Disable online search (Tavily) - only use document-based answers

# UI Configuration
PAGE_TITLE = "FinDocs RAG"
PAGE_ICON = "FD"
LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

# File Processing Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
CHROMA_COLLECTION_NAME = "rag-chroma"
CHROMA_PERSIST_DIR = "./.chroma"

# Evaluation Store (dedicated; do not clear on app startup)
CHROMA_EVAL_COLLECTION = "rag-chroma-eval"
CHROMA_EVAL_DIR = "./.chroma_eval"

# Model Configuration
LLM_TEMPERATURE = 0
TAVILY_SEARCH_RESULTS = 2

# Supported File Types
SUPPORTED_EXTENSIONS = [
    "pdf", "docx", "doc", "csv", "xlsx", "xls", 
    "txt", "md", "py", "js", "html", "xml"
]

# UI Messages
UPLOAD_PLACEHOLDER_TITLE = "Upload Financial Documents"
UPLOAD_PLACEHOLDER_TEXT = "Upload annual reports, fund documents, or other financial materials to analyze and query."
QUESTION_PLACEHOLDER = "What were the key financial highlights in this period?"
