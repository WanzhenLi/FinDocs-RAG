from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

class QuestionRelevance(BaseModel):
    """Model for question-answer relevance evaluation results"""
    
    binary_score: bool = Field(
        description="Whether the answer adequately addresses the question - true if relevant, false if not relevant"
    )
    
    relevance_score: float = Field(
        default=0.5,
        description="Relevance score between 0.0 and 1.0 indicating how well the answer addresses the question",
        ge=0.0,
        le=1.0
    )
    
    completeness: str = Field(
        default="partial",
        description="Assessment of answer completeness: 'complete', 'partial', or 'minimal'"
    )
    
    reasoning: str = Field(
        default="",
        description="Brief explanation of the relevance assessment and what makes the answer relevant or irrelevant"
    )
    
    missing_aspects: str = Field(
        default="",
        description="Key aspects of the question that are not addressed in the answer (if any)"
    )


llm = ChatOpenAI(temperature=0)
structured_output = llm.with_structured_output(QuestionRelevance)

system = """You are an expert question-answer relevance evaluator for a conversational AI system. Your role is to assess whether a generated answer properly addresses and resolves the user's question.

EVALUATION CRITERIA:

1. DIRECT RELEVANCE:
   - Does the answer directly address the core question being asked?
   - Are the main points of the question specifically addressed?
   - Is the answer focused on what the user actually wants to know?

2. COMPLETENESS ASSESSMENT:
   - Does the answer cover all important aspects of the question?
   - Are there significant parts of the question left unanswered?
   - Is the level of detail appropriate for the question type?

3. ACCURACY AND APPROPRIATENESS:
   - Is the answer factually consistent with what was asked?
   - Does the answer stay within the scope of the question?
   - Are there any contradictions or off-topic elements?

4. USEFULNESS FOR USER:
   - Would this answer satisfy the user's information need?
   - Is the answer actionable or informative as requested?
   - Does it provide the type of response the question implies?

SCORING GUIDELINES:
- Score 'true' if the answer adequately addresses the question and would satisfy the user
- Score 'false' if the answer is off-topic, incomplete, or fails to address the core question

ADDITIONAL ASSESSMENTS:
- Provide a relevance score (0.0-1.0) indicating answer quality
- Assess completeness level: complete, partial, or minimal
- Explain your reasoning for the evaluation
- Identify any missing key aspects if the answer is incomplete

Focus on practical utility - would this answer help the user achieve their goal?"""
relevance_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", """Please evaluate whether the generated answer properly addresses the user's question.

USER QUESTION:
{question}

GENERATED ANSWER:
{solution}

EVALUATION REQUIRED:
1. Binary Score: true if answer addresses question adequately, false if not
2. Relevance Score: 0.0-1.0 rating of how well answer addresses the question
3. Completeness: 'complete', 'partial', or 'minimal' coverage of question aspects
4. Reasoning: Brief explanation of your assessment
5. Missing Aspects: Key parts of question not addressed (if any)

Provide your comprehensive evaluation based on the criteria above."""),
    ]
)

question_relevance: RunnableSequence = relevance_prompt | structured_output