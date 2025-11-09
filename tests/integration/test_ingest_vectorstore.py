from types import SimpleNamespace

from langchain_core.documents import Document

import document_processor
import utils


class FakeUploadedFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content
        self.type = "application/pdf"

    def getvalue(self):
        return self._content

    @property
    def size(self):
        return len(self._content)


class FakeLoader:
    def load_multiple_uploaded_files(self, uploaded_files):
        docs = []
        for f in uploaded_files:
            docs.append(Document(page_content=f"content-{f.name}", metadata={"original_filename": f.name}))
        return docs


class FakeProgress:
    def progress(self, value):
        self.value = value

    def empty(self):
        pass


class FakeStatus:
    def text(self, message):
        self.message = message

    def empty(self):
        pass


class SessionState(dict):
    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value


class FakeStreamlit:
    def __init__(self):
        self.session_state = SessionState()
        self.last_error = None

    def progress(self, value):
        return FakeProgress()

    def empty(self):
        return FakeStatus()

    def error(self, message):
        self.last_error = message


class FakeChroma:
    def __init__(self):
        self.added_docs = []
        self.deleted_ids = []
        self.persist_calls = 0
        self.retriever = SimpleNamespace(name="fake-retriever")

    def delete(self, where):
        self.deleted_ids.append(where.get("file_id"))

    def add_documents(self, docs):
        self.added_docs.extend(docs)

    def persist(self):
        self.persist_calls += 1

    def as_retriever(self):
        return self.retriever


def test_process_files_updates_vector_store_incrementally(monkeypatch):
    fake_st = FakeStreamlit()
    monkeypatch.setattr(document_processor, "st", fake_st)

    processor = document_processor.DocumentProcessor.__new__(document_processor.DocumentProcessor)
    processor.document_loader = FakeLoader()
    processor.embedding_function = None

    monkeypatch.setattr(
        document_processor.DocumentProcessor,
        "_create_document_chunks",
        lambda self, docs: [Document(page_content=d.page_content, metadata=dict(d.metadata)) for d in docs],
    )

    fake_chroma = FakeChroma()
    monkeypatch.setattr(document_processor.DocumentProcessor, "_get_chroma_store", lambda self: fake_chroma)

    file_a = FakeUploadedFile("a.pdf", b"a1")
    file_b = FakeUploadedFile("b.pdf", b"b1")
    retriever = processor.process_files([file_a, file_b])

    ids_round1 = {utils.compute_file_hash(f) for f in (file_a, file_b)}
    added_ids = {doc.metadata["file_id"] for doc in fake_chroma.added_docs}
    assert added_ids == ids_round1
    assert fake_st.last_error is None
    assert retriever is fake_chroma.retriever
    assert fake_st.session_state["processed_file_ids"] == ids_round1

    fake_chroma.added_docs.clear()
    processor.process_files([file_a])

    removed_id = utils.compute_file_hash(file_b)
    assert removed_id in fake_chroma.deleted_ids
    assert not fake_chroma.added_docs  # no re-add for unchanged file
    assert fake_st.session_state["processed_file_ids"] == {utils.compute_file_hash(file_a)}
