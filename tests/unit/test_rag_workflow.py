from types import SimpleNamespace

from langchain_core.documents import Document

import rag_workflow as rag_module
from rag_workflow import RAGWorkflow


class FakeRetriever:
    def __init__(self, documents):
        self._documents = list(documents)
        self.calls = 0

    def invoke(self, question):
        self.calls += 1
        return list(self._documents)


def _setup_workflow(monkeypatch):
    fake_streamlit = SimpleNamespace(session_state={})
    monkeypatch.setattr(rag_module, "st", fake_streamlit)
    return RAGWorkflow(), fake_streamlit


def test_retrieve_without_retriever_triggers_online_search(monkeypatch):
    workflow, _ = _setup_workflow(monkeypatch)
    workflow.retriever = None

    result = workflow._retrieve({"question": "What is revenue?"})

    assert result["documents"] == []
    assert result["online_search"] is True


def test_retrieve_with_retriever_returns_documents(monkeypatch):
    workflow, _ = _setup_workflow(monkeypatch)
    docs = [Document(page_content="Alpha", metadata={})]
    workflow.set_retriever(FakeRetriever(docs))

    result = workflow._retrieve({"question": "What is alpha?"})

    assert result["documents"] == docs
    assert "online_search" not in result


def test_build_numbered_context_includes_metadata_and_truncates(monkeypatch):
    workflow, _ = _setup_workflow(monkeypatch)
    documents = [
        Document(
            page_content="Short context",
            metadata={"original_filename": "fileA.pdf", "page": 1, "chunk_id": 0},
        ),
        Document(
            page_content="B" * 1300,
            metadata={"original_filename": "fileB.pdf", "page": 2, "chunk_id": 3},
        ),
    ]

    context = workflow._build_numbered_context(documents)

    assert "[1] fileA.pdf | page 2 | chunk 0" in context
    assert "[2] fileB.pdf | page 3 | chunk 3" in context
    assert "B" * 1250 not in context  # confirms truncation occurred
    assert context.count("[1]") == 1 and context.count("[2]") == 1


def test_evaluate_filters_irrelevant_docs_and_flags_online(monkeypatch):
    workflow, _ = _setup_workflow(monkeypatch)
    documents = [
        Document(page_content="Relevant", metadata={}),
        Document(page_content="Irrelevant", metadata={}),
    ]

    class FakeEvalChain:
        def __init__(self):
            self.calls = 0

        def invoke(self, payload):
            self.calls += 1
            score = "yes" if self.calls == 1 else "no"
            return SimpleNamespace(
                score=score,
                relevance_score=0.9,
                coverage_assessment="ok",
                missing_information="",
            )

    fake_chain = FakeEvalChain()
    monkeypatch.setattr(rag_module, "evaluate_docs", fake_chain)

    result = workflow._evaluate({"question": "Q", "documents": documents})

    assert len(result["documents"]) == 1
    assert result["online_search"] is True
    assert result["search_method"] == "online"
    assert len(result["document_evaluations"]) == 2


def test_search_online_appends_results(monkeypatch):
    workflow, _ = _setup_workflow(monkeypatch)

    class FakeTavily:
        def __init__(self, k):
            self.k = k

        def invoke(self, payload):
            assert payload["query"] == "Q"
            return [{"content": "c1"}, {"content": "c2"}]

    monkeypatch.setattr(rag_module, "TavilySearchResults", FakeTavily)
    documents = [Document(page_content="Existing", metadata={})]

    result = workflow._search_online({"question": "Q", "documents": documents})

    assert len(result["documents"]) == 2
    assert result["search_method"] == "online"
    assert "c1" in result["documents"][-1].page_content


def test_check_hallucinations_records_scores(monkeypatch):
    workflow, _ = _setup_workflow(monkeypatch)

    class FakeDocRel:
        def __init__(self, binary_score):
            self.binary_score = binary_score

    class FakeDocRelChain:
        def __init__(self, binary_score):
            self.binary_score = binary_score

        def invoke(self, payload):
            return FakeDocRel(self.binary_score)

    class FakeQuestionRelChain:
        def __init__(self):
            self.called = False

        def invoke(self, payload):
            self.called = True
            return SimpleNamespace(relevance_score=1.0)

    fake_doc_chain = FakeDocRelChain(True)
    fake_q_chain = FakeQuestionRelChain()
    monkeypatch.setattr(rag_module, "document_relevance", fake_doc_chain)
    monkeypatch.setattr(rag_module, "question_relevance", fake_q_chain)

    state = {"question": "Q", "documents": [], "solution": "Ans"}
    route = workflow._check_hallucinations(state)

    assert route == "Answers Question"
    assert state["document_relevance_score"].binary_score is True
    assert fake_q_chain.called is True
    assert "question_relevance_score" in state


def test_check_hallucinations_skips_question_eval_when_not_grounded(monkeypatch):
    workflow, _ = _setup_workflow(monkeypatch)

    class FakeDocRelChain:
        def invoke(self, payload):
            return SimpleNamespace(binary_score=False)

    class FakeQuestionRelChain:
        def __init__(self):
            self.called = False

        def invoke(self, payload):
            self.called = True
            return None

    monkeypatch.setattr(rag_module, "document_relevance", FakeDocRelChain())
    fake_q = FakeQuestionRelChain()
    monkeypatch.setattr(rag_module, "question_relevance", fake_q)

    state = {"question": "Q", "documents": [], "solution": "Ans"}
    route = workflow._check_hallucinations(state)

    assert route == "Answers Question"
    assert "question_relevance_score" not in state
    assert fake_q.called is False
