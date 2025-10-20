"""
State management for LangGraph RAG workflow

This module defines the state structure used throughout the LangGraph RAG
workflow. The GraphState tracks all the information needed as the system
processes questions and generates answers.

The state includes:
- Question and answer data
- Document retrieval results
- Search method tracking
- Evaluation scores and metrics
- Error handling information

This state management approach is essential for LangGraph RAG implementations,
providing clear data flow and enabling complex workflow orchestration.
"""
from typing import List, TypedDict, Optional, Dict, Any

class GraphState(TypedDict):
    """
    State structure for LangGraph RAG workflow
    
    This defines all the data that flows through the LangGraph RAG pipeline.
    Each step in the workflow can read from and write to this state, allowing
    for complex decision-making and proper error handling.
    
    Used throughout the RAG workflow to maintain context and enable
    conditional logic based on processing results.
    """
    
    question: str
    solution: str
    online_search: bool
    documents: List[str]
    search_method: Optional[str]  # 'documents' or 'online'
    document_evaluations: Optional[List[Dict[str, Any]]]  # Store document evaluation results
    document_relevance_score: Optional[Dict[str, Any]]  # Store document relevance check
    question_relevance_score: Optional[Dict[str, Any]]  # Store question relevance check