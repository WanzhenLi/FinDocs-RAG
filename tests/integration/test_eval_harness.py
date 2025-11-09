import json
import os
from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from evaluation import harness


class FakeLoader:
    def is_supported_format(self, path):
        return True

    def load_multiple_documents(self, file_paths):
        docs = []
        for file_path in file_paths:
            name = os.path.basename(file_path)
            docs.append(Document(page_content=f"content-{name}", metadata={"original_filename": name}))
        return docs


class FakeChroma:
    collections = {}

    def __init__(self, collection_name, persist_directory, embedding_function=None):
        docs = self.collections.get(collection_name, [])
        self._collection = SimpleNamespace(count=lambda: len(docs))
        self._docs = docs

    @classmethod
    def from_documents(cls, documents, collection_name, embedding, persist_directory):
        cls.collections[collection_name] = list(documents)
        return cls(collection_name, persist_directory, embedding_function=None)

    def similarity_search(self, query, k):
        if not self._docs:
            raise ValueError("empty")
        return self._docs[:1]

    def as_retriever(self):
        docs = list(self._docs)

        class _Retriever:
            def __init__(self, docs):
                self.docs = docs

            def invoke(self, question):
                return list(self.docs)

        return _Retriever(docs)


def test_ensure_eval_index_builds_collection(monkeypatch):
    FakeChroma.collections = {}
    monkeypatch.setattr(harness, "MultiFormatDocumentLoader", FakeLoader)
    monkeypatch.setattr(harness, "Chroma", FakeChroma)

    retriever = harness.ensure_eval_index()

    assert FakeChroma.collections[harness.CHROMA_EVAL_COLLECTION]
    assert retriever.invoke("any")


def test_run_eval_computes_expected_metrics(monkeypatch, tmp_path):
    monkeypatch.setattr(harness, "ensure_eval_index", lambda: SimpleNamespace(invoke=lambda q: []))

    tc_path = Path("evaluation/test_cases.json")
    with open(tc_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    documents_per_case = []
    for idx, case in enumerate(test_cases):
        answer = case["expected_answers"][0] if idx % 2 == 0 else "mismatch"
        doc_name = (case.get("expected_docs") or ["doc.pdf"])[0]
        documents_per_case.append(
            {
                "solution": answer,
                "documents": [
                    Document(page_content=f"snippet {idx}", metadata={"original_filename": doc_name}),
                    Document(page_content=f"duplicate {idx}", metadata={"original_filename": doc_name}),
                    Document(page_content=f"nometa {idx}", metadata={}),
                ],
                "document_relevance_score": None,
                "question_relevance_score": None,
            }
        )

    class FakeWorkflow:
        def __init__(self, results):
            self.results = results
            self.calls = 0

        def set_retriever(self, retriever):
            self.retriever = retriever

        def process_question(self, question):
            result = self.results[self.calls]
            self.calls += 1
            return result

    workflow = FakeWorkflow(documents_per_case)

    summary, details = harness.run_eval(workflow)

    soft_ratio = sum(1 for d in details if d["soft_em"]) / len(details)
    assert summary["soft_em_avg"] == soft_ratio
    assert 0 < soft_ratio < 1
    assert summary["retrieval_hit_rate"] == 1.0
    assert len(details) == len(test_cases)
    assert details[0]["soft_em"] is True
