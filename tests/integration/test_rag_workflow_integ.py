from types import SimpleNamespace

from langchain_core.documents import Document

import rag_workflow as rag_module
from rag_workflow import RAGWorkflow


class SessionState(dict):
    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents

    def invoke(self, question):
        return self.documents


class FakeEvalChain:
    def invoke(self, payload):
        return SimpleNamespace(
            score="yes",
            relevance_score=0.9,
            coverage_assessment="covers question",
            missing_information="",
        )


class FakeGenerateChain:
    def invoke(self, payload):
        return f"Answer to {payload['question']}[1]"


class FakeDocRelChain:
    def __init__(self):
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        return SimpleNamespace(binary_score=True, reasoning="grounded")


class FakeQuestionRelChain:
    def __init__(self):
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        return SimpleNamespace(reasoning="answers question")


def test_rag_workflow_process_question_returns_metrics(monkeypatch):
    fake_st = SimpleNamespace(session_state=SessionState())
    monkeypatch.setattr(rag_module, "st", fake_st)
    monkeypatch.setattr(rag_module, "evaluate_docs", FakeEvalChain())
    monkeypatch.setattr(rag_module, "generate_chain", FakeGenerateChain())
    fake_doc_chain = FakeDocRelChain()
    fake_q_chain = FakeQuestionRelChain()
    monkeypatch.setattr(rag_module, "document_relevance", fake_doc_chain)
    monkeypatch.setattr(rag_module, "question_relevance", fake_q_chain)

    workflow = RAGWorkflow()
    documents = [Document(page_content="Revenue grew by 10%", metadata={"original_filename": "report.pdf"})]
    workflow.set_retriever(FakeRetriever(documents))

    result = workflow.process_question("What was revenue growth?")

    assert result["solution"].startswith("Answer to")
    assert len(result["documents"]) == 1
    assert len(result["document_evaluations"]) == 1
    assert fake_doc_chain.calls == 1
    assert fake_q_chain.calls == 1
