import os
from pathlib import Path

import pytest
from langchain_core.documents import Document

import document_loader


class FakeUploadedFile:
    def __init__(self, name, content=b"data", file_type="application/pdf"):
        self.name = name
        self._content = content
        self.type = file_type

    def getvalue(self):
        return self._content

    @property
    def size(self):
        return len(self._content)


class FakeBaseLoader:
    def __init__(self, supported=True, documents=None, error=None):
        self.supported = supported
        self.documents = documents or [Document(page_content="chunk", metadata={})]
        self.error = error
        self.last_path = None
        self.checked = []

    def load_document(self, file_path):
        self.last_path = file_path
        if self.error:
            raise self.error
        return [Document(page_content=doc.page_content, metadata=dict(doc.metadata)) for doc in self.documents]

    def is_supported_format(self, filename):
        self.checked.append(filename)
        return self.supported

    def get_supported_extensions(self):
        return ["pdf", "txt", "md"]


def _patch_base_loader(monkeypatch, fake_loader):
    monkeypatch.setattr(document_loader, "BaseMultiFormatLoader", lambda: fake_loader)


def test_load_document_delegates_to_base_loader(monkeypatch):
    fake_loader = FakeBaseLoader()
    _patch_base_loader(monkeypatch, fake_loader)
    loader = document_loader.StreamlitMultiFormatDocumentLoader()

    result = loader.load_document("foo.pdf")

    assert fake_loader.last_path == "foo.pdf"
    assert result[0].metadata == {}


def test_load_uploaded_file_enriches_metadata_and_cleans_tempfile(monkeypatch):
    docs = [Document(page_content="hello", metadata={})]
    fake_loader = FakeBaseLoader(documents=docs)
    _patch_base_loader(monkeypatch, fake_loader)
    loader = document_loader.StreamlitMultiFormatDocumentLoader()
    uploaded = FakeUploadedFile("sample.pdf", b"alpha", "application/pdf")

    result = loader.load_uploaded_file(uploaded)

    assert result[0].metadata["original_filename"] == "sample.pdf"
    assert result[0].metadata["upload_size"] == uploaded.size
    assert result[0].metadata["upload_type"] == uploaded.type
    assert result[0].metadata["processed_via"] == "streamlit_upload"
    assert fake_loader.last_path and not Path(fake_loader.last_path).exists()


def test_load_uploaded_file_warns_when_tempfile_cleanup_fails(monkeypatch):
    fake_loader = FakeBaseLoader()
    _patch_base_loader(monkeypatch, fake_loader)
    loader = document_loader.StreamlitMultiFormatDocumentLoader()
    uploaded = FakeUploadedFile("sample.pdf")
    deleted_paths = []

    def _raise_os_error(path):
        deleted_paths.append(path)
        raise OSError("cannot delete")

    monkeypatch.setattr(document_loader.os, "unlink", _raise_os_error)

    result = loader.load_uploaded_file(uploaded)

    assert deleted_paths, "Temp path should be attempted for deletion"
    assert result, "Documents should still be returned"


def test_load_uploaded_file_rejects_unsupported_extension(monkeypatch):
    fake_loader = FakeBaseLoader(supported=False)
    _patch_base_loader(monkeypatch, fake_loader)
    loader = document_loader.StreamlitMultiFormatDocumentLoader()

    with pytest.raises(ValueError, match="Unsupported file type"):
        loader.load_uploaded_file(FakeUploadedFile("notes.xls", b"123"))


def test_load_uploaded_file_wraps_loader_errors(monkeypatch):
    fake_loader = FakeBaseLoader(error=RuntimeError("boom"))
    _patch_base_loader(monkeypatch, fake_loader)
    loader = document_loader.StreamlitMultiFormatDocumentLoader()

    with pytest.raises(Exception, match="Failed to process uploaded file sample.pdf: boom"):
        loader.load_uploaded_file(FakeUploadedFile("sample.pdf", b"x"))

    assert fake_loader.last_path and not os.path.exists(fake_loader.last_path)


def test_load_multiple_uploaded_files_collects_docs_and_continues_on_failure(monkeypatch):
    loader = document_loader.StreamlitMultiFormatDocumentLoader()
    success = FakeUploadedFile("a.pdf")
    failure = FakeUploadedFile("b.pdf")

    def _fake_load(uploaded_file):
        if uploaded_file.name == "b.pdf":
            raise RuntimeError("bad file")
        return [Document(page_content=uploaded_file.name, metadata={"original_filename": uploaded_file.name})]

    loader.load_uploaded_file = _fake_load
    docs = loader.load_multiple_uploaded_files([success, failure])

    assert [doc.page_content for doc in docs] == ["a.pdf"]


def test_load_multiple_uploaded_files_without_failures():
    loader = document_loader.StreamlitMultiFormatDocumentLoader()

    def _fake_load(uploaded_file):
        return [Document(page_content=uploaded_file.name, metadata={})]

    loader.load_uploaded_file = _fake_load
    docs = loader.load_multiple_uploaded_files([FakeUploadedFile("ok.pdf")])

    assert len(docs) == 1


def test_get_supported_extensions_and_display(monkeypatch):
    fake_loader = FakeBaseLoader()
    _patch_base_loader(monkeypatch, fake_loader)
    loader = document_loader.StreamlitMultiFormatDocumentLoader()

    extensions = loader.get_supported_extensions()
    display = loader.get_supported_extensions_display()

    assert set(extensions) == {"pdf", "txt", "md"}
    assert display == ".md, .pdf, .txt"


def test_get_upload_info_reports_support_and_type(monkeypatch):
    fake_loader = FakeBaseLoader()
    _patch_base_loader(monkeypatch, fake_loader)
    loader = document_loader.StreamlitMultiFormatDocumentLoader()
    uploaded = FakeUploadedFile("info.pdf", b"xyz", "application/pdf")

    info = loader.get_upload_info(uploaded)

    assert info["filename"] == "info.pdf"
    assert info["size"] == uploaded.size
    assert info["extension"] == "pdf"
    assert info["is_supported"] is True
    assert info["type"] == "application/pdf"
    assert fake_loader.checked[-1].endswith(".pdf")


def test_module_level_helpers_create_loader(monkeypatch):
    created = []

    class LoaderStub:
        def __init__(self):
            created.append(self)

        def load_document(self, file_path):
            return ["from-file", file_path]

        def load_uploaded_file(self, uploaded_file):
            return ["from-upload", uploaded_file.name]

    monkeypatch.setattr(document_loader, "StreamlitMultiFormatDocumentLoader", LoaderStub)

    assert document_loader.load_document("foo.pdf")[0] == "from-file"
    uploaded = FakeUploadedFile("bar.pdf")
    assert document_loader.load_uploaded_file(uploaded)[0] == "from-upload"
    assert len(created) == 2
