import pytest
from langchain_core.documents import Document

import document_processor
import utils
from document_processor import DocumentProcessor


class SessionState(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


class FakeProgress:
    def __init__(self):
        self.values = []
        self.cleared = False

    def progress(self, value):
        self.values.append(value)

    def empty(self):
        self.cleared = True


class FakeStatus:
    def __init__(self):
        self.messages = []
        self.cleared = False

    def text(self, message):
        self.messages.append(message)

    def empty(self):
        self.cleared = True


class FakeStreamlit:
    def __init__(self):
        self.session_state = SessionState()
        self.errors = []
        self.infos = []
        self.successes = []
        self.progresses = []
        self.statuses = []

    def progress(self, initial_value):
        progress = FakeProgress()
        progress.progress(initial_value)
        self.progresses.append(progress)
        return progress

    def empty(self):
        status = FakeStatus()
        self.statuses.append(status)
        return status

    def error(self, message):
        self.errors.append(message)

    def info(self, message):
        self.infos.append(message)

    def success(self, message):
        self.successes.append(message)


class FakeUploadedFile:
    def __init__(self, name="doc.pdf", content=b"hello world", file_type="application/pdf"):
        self.name = name
        self._content = content
        self.type = file_type

    def getvalue(self):
        return self._content

    @property
    def size(self):
        return len(self._content)


def _make_processor():
    processor = DocumentProcessor.__new__(DocumentProcessor)
    processor.document_loader = object()
    processor.embedding_function = None
    return processor


@pytest.fixture
def fake_streamlit(monkeypatch):
    fake = FakeStreamlit()
    monkeypatch.setattr(document_processor, "st", fake)
    return fake


def test_create_document_chunks_adds_metadata():
    processor = _make_processor()
    documents = [
        Document(page_content="A" * 80, metadata={"original_filename": "fileA.pdf"}),
        Document(page_content="B" * 80, metadata={"original_filename": "fileB.pdf"}),
    ]

    chunks = processor._create_document_chunks(documents)

    assert chunks, "Chunk list should not be empty"
    total = len(chunks)
    assert all("chunk_id" in chunk.metadata for chunk in chunks)
    assert all(chunk.metadata["total_chunks"] == total for chunk in chunks)
    assert all("chunk_size" in chunk.metadata for chunk in chunks)


def test_create_document_chunks_uses_configured_strategy(monkeypatch):
    processor = _make_processor()
    documents = [Document(page_content="X" * 200, metadata={"original_filename": "fileA.pdf"})]

    captured = {}
    custom_size = 42
    custom_overlap = 17

    class FakeSplitter:
        def split_documents(self, docs):
            captured["docs"] = docs
            return [Document(page_content="chunk", metadata={})]

    class FakeCharacterTextSplitter:
        @staticmethod
        def from_tiktoken_encoder(*, chunk_size, chunk_overlap):
            captured["chunk_size"] = chunk_size
            captured["chunk_overlap"] = chunk_overlap
            return FakeSplitter()

    monkeypatch.setattr(document_processor, "CHUNK_SIZE", custom_size)
    monkeypatch.setattr(document_processor, "CHUNK_OVERLAP", custom_overlap)
    monkeypatch.setattr(document_processor, "CharacterTextSplitter", FakeCharacterTextSplitter)

    chunks = processor._create_document_chunks(documents)

    assert captured["chunk_size"] == custom_size
    assert captured["chunk_overlap"] == custom_overlap
    assert captured["docs"] == documents
    assert chunks[0].metadata["chunk_id"] == 0
    assert chunks[0].metadata["total_chunks"] == 1
    assert chunks[0].metadata["chunk_size"] == len(chunks[0].page_content)


def test_create_document_chunks_respects_actual_size_limits():
    processor = _make_processor()
    paragraph = "This is a financial document containing important quarterly earnings data and market analysis.\n\n"
    long_text = paragraph * 300
    documents = [Document(page_content=long_text, metadata={"original_filename": "long_doc.pdf"})]
    
    chunks = processor._create_document_chunks(documents)

    assert len(chunks) > 1, f"Long text should be split into multiple chunks, but got only {len(chunks)} chunk(s)"
    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_id"] == i, f"Chunk {i}'s chunk_id is incorrect"
        assert chunk.metadata["total_chunks"] == total_chunks, f"Chunk {i}'s total_chunks is incorrect"
        assert chunk.metadata["chunk_size"] == len(chunk.page_content), f"Chunk {i}'s chunk_size does not match actual content length"
        assert "original_filename" in chunk.metadata, f"Chunk {i} is missing original_filename in metadata"
    if len(chunks) >= 2:
        first_chunk_end = chunks[0].page_content[-50:]
        second_chunk_start = chunks[1].page_content[:100]
        has_overlap = any(
            word in second_chunk_start 
            for word in first_chunk_end.split()[-5:] 
            if len(word) > 3
        )
        assert has_overlap, "There should be content overlap between adjacent chunks"
    all_chunk_text = " ".join(chunk.page_content for chunk in chunks)
    assert len(all_chunk_text) >= len(long_text) * 0.9, "Total length of chunked content should not be significantly less than the original"


def test_process_file_returns_none_when_no_file(fake_streamlit):
    processor = _make_processor()

    assert processor.process_file(None) is None


def test_process_file_uses_cached_retriever(fake_streamlit, monkeypatch):
    processor = _make_processor()
    uploaded = FakeUploadedFile("report.pdf", b"report data")
    cached_retriever = object()
    key = utils.get_file_key(uploaded)
    fake_streamlit.session_state.processed_file = key
    fake_streamlit.session_state.retriever = cached_retriever

    def _fail(*args, **kwargs):
        raise AssertionError("Should not process when cache hits")

    monkeypatch.setattr(DocumentProcessor, "_process_new_file", _fail)

    assert processor.process_file(uploaded) is cached_retriever


def test_process_file_logs_error_when_pipeline_fails(fake_streamlit, monkeypatch):
    processor = _make_processor()
    uploaded = FakeUploadedFile("error.pdf")

    def _boom(self, user_file, key):
        raise RuntimeError("explode")

    monkeypatch.setattr(DocumentProcessor, "_process_new_file", _boom)

    result = processor.process_file(uploaded)

    assert result is None
    assert fake_streamlit.errors[-1].startswith("❌ Error processing file")
    assert "supported format" in fake_streamlit.infos[-1]


def test_process_new_file_rejects_unsupported_types(fake_streamlit, monkeypatch):
    file_info = {
        "filename": "notes.xls",
        "extension": "xls",
        "is_supported": False,
        "size": 10,
    }

    class LoaderStub:
        def get_upload_info(self, user_file):
            return file_info

        def get_supported_extensions_display(self):
            return ".pdf, .txt"

    processor = _make_processor()
    processor.document_loader = LoaderStub()

    recorded_info = []
    monkeypatch.setattr(document_processor, "render_file_analysis", lambda info: recorded_info.append(info))

    result = processor._process_new_file(FakeUploadedFile("notes.xls"), "key")

    assert result is None
    assert recorded_info == [file_info]
    assert "Unsupported file type" in fake_streamlit.errors[-1]
    assert ".pdf, .txt" in fake_streamlit.infos[-1]


def test_process_new_file_runs_pipeline_for_supported_files(fake_streamlit, monkeypatch):
    file_info = {
        "filename": "alpha.pdf",
        "extension": "pdf",
        "is_supported": True,
        "size": 20,
    }

    class LoaderStub:
        def get_upload_info(self, user_file):
            return file_info

    processor = _make_processor()
    processor.document_loader = LoaderStub()
    sentinel = object()
    monkeypatch.setattr(DocumentProcessor, "_execute_processing_pipeline", lambda self, user_file, info, key: sentinel)
    monkeypatch.setattr(document_processor, "render_file_analysis", lambda info: None)

    result = processor._process_new_file(FakeUploadedFile("alpha.pdf"), "file-key")

    assert result is sentinel


def test_execute_processing_pipeline_updates_session_state(fake_streamlit, monkeypatch):
    documents = [Document(page_content="chunk", metadata={"original_filename": "alpha.pdf"})]

    class LoaderStub:
        def load_uploaded_file(self, user_file):
            return documents

    class FakeRetriever:
        def __init__(self):
            self.queries = []

        def invoke(self, query):
            self.queries.append(query)
            return []

    class FakeVectorStore:
        def __init__(self):
            self.retriever = FakeRetriever()

        def as_retriever(self):
            return self.retriever

    processor = _make_processor()
    processor.document_loader = LoaderStub()

    monkeypatch.setattr(DocumentProcessor, "_create_document_chunks", lambda self, docs: docs)
    monkeypatch.setattr(DocumentProcessor, "_create_vector_database", lambda self, doc_splits: FakeVectorStore())
    monkeypatch.setattr(document_processor.time, "sleep", lambda *_: None)
    monkeypatch.setattr(document_processor, "render_file_analysis", lambda info: None)

    file_info = {"filename": "alpha.pdf"}
    retriever = processor._execute_processing_pipeline(FakeUploadedFile("alpha.pdf"), file_info, "cache-key")

    assert fake_streamlit.session_state.processed_file == "cache-key"
    assert fake_streamlit.session_state.retriever is retriever
    assert fake_streamlit.progresses[0].values == [0, 25, 50, 75, 90, 100]
    assert fake_streamlit.statuses[0].messages[0] == "🔄 Loading document..."
    assert fake_streamlit.successes  # extracted content message
    assert retriever.queries == ["test"]


def test_process_files_returns_none_for_empty_list(fake_streamlit):
    processor = _make_processor()

    assert processor.process_files([]) is None


def test_process_files_returns_cached_retriever_when_files_key_matches(fake_streamlit, monkeypatch):
    processor = _make_processor()
    retriever = object()
    fake_streamlit.session_state.processed_files_key = "files-key"
    fake_streamlit.session_state.retriever = retriever

    monkeypatch.setattr(document_processor, "get_files_key", lambda files: "files-key")

    result = processor.process_files([FakeUploadedFile("cached.pdf")])

    assert result is retriever


def test_process_files_handles_delete_and_persist_errors(fake_streamlit, monkeypatch):
    class LoaderStub:
        def load_multiple_uploaded_files(self, uploaded_files):
            return [
                Document(page_content="doc", metadata={"original_filename": uploaded_files[0].name})
            ]

    class FaultyChroma:
        def __init__(self):
            self.deleted = []
            self.added_docs = []
            self.persist_calls = 0
            self.retriever = object()

        def delete(self, where):
            self.deleted.append(where)
            raise RuntimeError("delete failed")

        def add_documents(self, docs):
            self.added_docs.extend(docs)

        def persist(self):
            self.persist_calls += 1
            raise RuntimeError("persist failed")

        def as_retriever(self):
            return self.retriever

    processor = _make_processor()
    processor.document_loader = LoaderStub()
    fake_store = FaultyChroma()
    fake_streamlit.session_state.processed_file_ids = {"stale"}

    monkeypatch.setattr(document_processor, "compute_file_hash", lambda uploaded: uploaded.name)
    monkeypatch.setattr(document_processor, "get_files_key", lambda files: "new-key")
    monkeypatch.setattr(DocumentProcessor, "_create_document_chunks", lambda self, docs: docs)
    monkeypatch.setattr(DocumentProcessor, "_get_chroma_store", lambda self: fake_store)
    monkeypatch.setattr(document_processor.time, "sleep", lambda *_: None)

    retriever = processor.process_files([FakeUploadedFile("fresh.pdf", b"ok")])

    assert retriever is fake_store.retriever
    assert fake_store.deleted, "Should attempt to delete stale IDs"
    assert fake_store.added_docs, "New documents should be added"
    assert fake_store.persist_calls == 1
    assert fake_streamlit.session_state.processed_file_ids == {"fresh.pdf"}


def test_process_files_recovers_from_loader_failure(fake_streamlit, monkeypatch):
    class BrokenLoader:
        def load_multiple_uploaded_files(self, uploaded_files):
            raise RuntimeError("boom")

    processor = _make_processor()
    processor.document_loader = BrokenLoader()
    fake_streamlit.session_state.processed_file_ids = set()

    monkeypatch.setattr(document_processor, "get_files_key", lambda files: "key")
    monkeypatch.setattr(document_processor.time, "sleep", lambda *_: None)

    result = processor.process_files([FakeUploadedFile("bad.pdf")])

    assert result is None
    assert fake_streamlit.errors[-1].startswith("❌ Error processing files")
    assert fake_streamlit.progresses[-1].cleared
    assert fake_streamlit.statuses[-1].cleared


def test_execute_processing_pipeline_handles_retriever_probe_failure(fake_streamlit, monkeypatch):
    class LoaderStub:
        def load_uploaded_file(self, user_file):
            return [Document(page_content="chunk", metadata={"original_filename": user_file.name})]

    class BrokenRetriever:
        def invoke(self, query):
            raise RuntimeError("cannot probe")

    class FakeVectorStore:
        def as_retriever(self):
            return BrokenRetriever()

    processor = _make_processor()
    processor.document_loader = LoaderStub()

    monkeypatch.setattr(DocumentProcessor, "_create_document_chunks", lambda self, docs: docs)
    monkeypatch.setattr(DocumentProcessor, "_create_vector_database", lambda self, docs: FakeVectorStore())
    monkeypatch.setattr(document_processor.time, "sleep", lambda *_: None)
    monkeypatch.setattr(document_processor, "render_file_analysis", lambda info: None)

    retriever = processor._execute_processing_pipeline(FakeUploadedFile("alpha.pdf"), {"filename": "alpha.pdf"}, "cache")

    assert isinstance(retriever, BrokenRetriever)


def test_execute_processing_pipeline_cleans_up_on_exception(fake_streamlit, monkeypatch):
    class LoaderStub:
        def load_uploaded_file(self, user_file):
            raise RuntimeError("fail")

    processor = _make_processor()
    processor.document_loader = LoaderStub()

    monkeypatch.setattr(document_processor.time, "sleep", lambda *_: None)
    monkeypatch.setattr(document_processor, "render_file_analysis", lambda info: None)

    with pytest.raises(RuntimeError):
        processor._execute_processing_pipeline(FakeUploadedFile("bad.pdf"), {"filename": "bad.pdf"}, "cache")

    assert fake_streamlit.progresses[-1].cleared
    assert fake_streamlit.statuses[-1].cleared


def test_create_vector_database_invokes_chroma(monkeypatch):
    class FakeChroma:
        from_documents_args = None

        @classmethod
        def from_documents(cls, documents, collection_name, embedding, persist_directory):
            cls.from_documents_args = {
                "documents": documents,
                "collection_name": collection_name,
                "embedding": embedding,
                "persist_directory": persist_directory,
            }
            return "vector-db"

    processor = _make_processor()
    processor.embedding_function = "embeddings"

    monkeypatch.setattr(document_processor, "Chroma", FakeChroma)
    monkeypatch.setattr(document_processor, "get_session_collection_name", lambda base: f"{base}-session")

    docs = [Document(page_content="chunk", metadata={})]
    result = processor._create_vector_database(docs)

    assert result == "vector-db"
    assert FakeChroma.from_documents_args["collection_name"].endswith("-session")


def test_get_chroma_store_uses_session_collection_name(fake_streamlit, monkeypatch):
    init_calls = []

    class FakeChroma:
        def __init__(self, collection_name, persist_directory, embedding_function):
            init_calls.append(
                {
                    "collection_name": collection_name,
                    "persist_directory": persist_directory,
                    "embedding_function": embedding_function,
                }
            )

    processor = _make_processor()
    processor.embedding_function = "emb"

    monkeypatch.setattr(document_processor, "Chroma", FakeChroma)
    monkeypatch.setattr(document_processor, "get_session_collection_name", lambda base: "session-collection")

    store = processor._get_chroma_store()

    assert isinstance(store, FakeChroma)
    assert init_calls[0]["collection_name"] == "session-collection"
