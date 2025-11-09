from pathlib import Path

import pytest

from multimodal_loader import MultiFormatDocumentLoader


def test_is_supported_format_recognizes_known_extensions():
    loader = MultiFormatDocumentLoader()
    assert loader.is_supported_format("docs.pdf")
    assert loader.is_supported_format("docs.docx")
    assert not loader.is_supported_format("docs.zip")
    assert not loader.is_supported_format(Path("docs.png"))

def test_load_document_adds_basic_metadata(tmp_path):
    loader = MultiFormatDocumentLoader()
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Revenue guidance improved.\n")

    documents = loader.load_document(file_path)

    assert documents, "Loader should return at least one chunk"
    meta = documents[0].metadata
    assert meta["file_name"] == "sample.txt"
    assert meta["file_type"] == "txt"
    assert meta["file_size"] == file_path.stat().st_size
    assert Path(meta["source"]).name == "sample.txt"


def test_load_document_raises_for_missing_file(tmp_path):
    loader = MultiFormatDocumentLoader()
    missing_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        loader.load_document(missing_path)
