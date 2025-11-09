from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from evaluation import harness


def _patch_embeddings(monkeypatch):
    monkeypatch.setattr(harness, "OpenAIEmbeddings", lambda *args, **kwargs: None)


def test_eval_collection_has_data_via_collection_count(monkeypatch):
    _patch_embeddings(monkeypatch)

    class FakeCollection:
        def count(self):
            return 3

    class FakeChroma:
        def __init__(self, *args, **kwargs):
            self._collection = FakeCollection()

    monkeypatch.setattr(harness, "Chroma", FakeChroma)

    assert harness._eval_collection_has_data() is True


def test_eval_collection_uses_similarity_search_when_count_missing(monkeypatch):
    _patch_embeddings(monkeypatch)

    class FakeChroma:
        def __init__(self, *args, **kwargs):
            self._collection = None

        def similarity_search(self, query, k):
            return ["result"]

    monkeypatch.setattr(harness, "Chroma", FakeChroma)

    assert harness._eval_collection_has_data() is True


def test_eval_collection_returns_false_when_similarity_fails(monkeypatch):
    _patch_embeddings(monkeypatch)

    class FakeChroma:
        def __init__(self, *args, **kwargs):
            self._collection = None

        def similarity_search(self, *args, **kwargs):
            raise ValueError("empty")

    monkeypatch.setattr(harness, "Chroma", FakeChroma)

    assert harness._eval_collection_has_data() is False


def test_eval_collection_handles_faulty_collection_count(monkeypatch):
    _patch_embeddings(monkeypatch)

    class FaultyCollection:
        def count(self):
            raise RuntimeError("fail")

    class FakeChroma:
        def __init__(self, *args, **kwargs):
            self._collection = FaultyCollection()

        def similarity_search(self, query, k):
            return ["fallback"]

    monkeypatch.setattr(harness, "Chroma", FakeChroma)

    assert harness._eval_collection_has_data() is True


def test_eval_collection_returns_false_when_constructor_errors(monkeypatch):
    _patch_embeddings(monkeypatch)

    def _fail(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(harness, "Chroma", _fail)

    assert harness._eval_collection_has_data() is False


def test_ensure_eval_index_short_circuits_when_collection_exists(monkeypatch):
    _patch_embeddings(monkeypatch)
    monkeypatch.setattr(harness, "_eval_collection_has_data", lambda: True)

    class FakeChroma:
        def __init__(self, *args, **kwargs):
            self.called = True

        def as_retriever(self):
            return "retriever"

    monkeypatch.setattr(harness, "Chroma", FakeChroma)

    assert harness.ensure_eval_index() == "retriever"


def test_ensure_eval_index_raises_when_docs_dir_missing(monkeypatch):
    _patch_embeddings(monkeypatch)
    monkeypatch.setattr(harness, "_eval_collection_has_data", lambda: False)
    monkeypatch.setattr(harness.os.path, "isdir", lambda path: False)

    with pytest.raises(FileNotFoundError):
        harness.ensure_eval_index()


def test_ensure_eval_index_raises_when_no_supported_docs(monkeypatch):
    _patch_embeddings(monkeypatch)
    monkeypatch.setattr(harness, "_eval_collection_has_data", lambda: False)
    monkeypatch.setattr(harness.os.path, "isdir", lambda path: True)
    monkeypatch.setattr(harness.os, "listdir", lambda path: ["doc.unsupported"])
    monkeypatch.setattr(harness.os.path, "isfile", lambda path: True)

    class LoaderStub:
        def is_supported_format(self, path):
            return False

    monkeypatch.setattr(harness, "MultiFormatDocumentLoader", lambda: LoaderStub())

    with pytest.raises(FileNotFoundError, match="No supported evaluation documents"):
        harness.ensure_eval_index()


def test_matches_soft_skips_invalid_regex(monkeypatch):
    assert harness._matches_soft("text", [], ["["]) is False


def test_doc_hit_requires_docs_and_expectations():
    assert harness._doc_hit([], ["doc.pdf"]) is False
    assert harness._doc_hit([Document(page_content="", metadata={})], []) is False


def test_compute_token_f1_handles_overlap_and_empty():
    assert harness.compute_token_f1("", "") == 0.0
    score = harness.compute_token_f1("Revenue was 5 billion", "5 billion guidance")
    assert 0 < score <= 1
    assert harness.compute_token_f1("alpha beta", "gamma delta") == 0.0


def test_run_eval_raises_when_test_cases_missing(monkeypatch):
    monkeypatch.setattr(harness, "ensure_eval_index", lambda: SimpleNamespace(invoke=lambda q: []))

    real_exists = harness.os.path.exists

    def fake_exists(path):
        if path == harness.os.path.join("evaluation", "test_cases.json"):
            return False
        return real_exists(path)

    monkeypatch.setattr(harness.os.path, "exists", fake_exists)

    with pytest.raises(FileNotFoundError, match="Test cases not found"):
        harness.run_eval(SimpleNamespace(set_retriever=lambda r: None, process_question=lambda q: {}))
