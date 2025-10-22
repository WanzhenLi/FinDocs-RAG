"""
Document evaluation chain for LangGraph RAG workflows

This module handles document relevance evaluation as part of the LangGraph RAG
pipeline. It determines whether retrieved documents contain enough relevant
information to answer a user's question effectively.

The evaluation chain is a key component in LangGraph RAG systems, providing
quality gates that prevent irrelevant documents from being used for answer
generation. This improves the overall quality of RAG responses.

Used within the LangGraph workflow to make routing decisions about whether
to proceed with document-based answers or fall back to online search.
"""
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0)

class EvaluateDocs(BaseModel):
    """
    Document evaluation results for LangGraph RAG workflows
    
    This model structures the evaluation results when assessing whether
    retrieved documents are sufficient for answering a question. Used
    throughout the LangGraph RAG workflow for routing decisions.
    """
    
    score: str = Field(
        description="Whether documents are relevant to the question - 'yes' if sufficient, 'no' if insufficient"
    )
    
    relevance_score: float = Field(
        default=0.5,
        description="Relevance score between 0.0 and 1.0 indicating how well documents match the query",
        ge=0.0,
        le=1.0
    )
    
    coverage_assessment: str = Field(
        default="",
        description="Assessment of how well the documents cover the query requirements"
    )
    
    missing_information: str = Field(
        default="",
        description="Description of key information missing from documents (if any)"
    )


structured_output = llm.with_structured_output(EvaluateDocs)

system = """You are an expert document relevance evaluator for a RAG (Retrieval-Augmented Generation) system. Your role is to assess whether retrieved documents contain sufficient information to answer a user's query effectively.

EVALUATION FRAMEWORK:

1. TOPICAL RELEVANCE:
   - Do the documents directly address the main subject of the query?
   - Are the key concepts and themes aligned with what the user is asking?

2. INFORMATION SUFFICIENCY:
   - Is there enough detail to provide a comprehensive answer?
   - Are specific facts, data, or examples present when needed?
   - Can the query be answered without requiring external knowledge?

3. INFORMATION QUALITY:
   - Is the information accurate and credible?
   - Are there conflicting statements within the documents?
   - Is the information current and relevant to the query context?

4. COMPLETENESS ASSESSMENT:
   - Does the document set cover all aspects of the query?
   - Are there obvious gaps in information that would prevent a complete answer?

SCORING CRITERIA:
- Score 'yes' if documents provide sufficient, relevant information to answer the query satisfactorily
- Score 'no' if documents lack key information, are off-topic, or insufficient for a complete answer

ADDITIONAL REQUIREMENTS:
- Provide a relevance score (0.0-1.0) indicating match quality
- Assess coverage of query requirements
- Identify any missing critical information

Be thorough but efficient in your evaluation. Focus on practical utility for answer generation."""

evaluate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", """Please evaluate whether the retrieved documents are sufficient to answer the user's query.

USER QUERY:
{question}

RETRIEVED DOCUMENTS:
{document}

EVALUATION REQUIRED:
1. Primary Score: 'yes' if documents are sufficient, 'no' if insufficient
2. Relevance Score: 0.0-1.0 rating of how well documents match the query
3. Coverage Assessment: How well do the documents address the query requirements?
4. Missing Information: What key information (if any) is missing for a complete answer?

Provide your comprehensive evaluation based on the framework above."""),
    ]
)

evaluate_docs = evaluate_prompt | structured_output