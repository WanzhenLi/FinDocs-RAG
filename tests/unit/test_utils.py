from types import SimpleNamespace

import pytest

import utils


def test_clear_chroma_db_removes_directory(monkeypatch, tmp_path):
    data_dir = tmp_path / "chroma"
    data_dir.mkdir()
    (data_dir / "data.bin").write_text("vector-bytes")
    monkeypatch.setattr(utils, "CHROMA_PERSIST_DIR", str(data_dir))

    utils.clear_chroma_db()
    assert not data_dir.exists()

    # Second call should be a no-op and not raise
    utils.clear_chroma_db()


class SessionState(dict):
    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value


def test_initialize_session_state_sets_missing_defaults(monkeypatch):
    # Pre-populate a couple of keys to ensure they are preserved
    session_state = SessionState(
        processed_file="kept",
        retriever="existing_retriever",
    )
    fake_streamlit = SimpleNamespace(session_state=session_state)
    monkeypatch.setattr(utils, "st", fake_streamlit)

    utils.initialize_session_state()

    expected_keys = {
        "processed_file",
        "retriever",
        "graph_instance",
        "db_cleared",
        "session_id",
        "processed_files_key",
        "processed_file_ids",
        "processing",
        "pending_question",
        "eval_requested",
        "eval_status",
        "current_view",
        "eval_results",
        "eval_completed_at",
    }
    assert expected_keys.issubset(fake_streamlit.session_state.keys())
    assert fake_streamlit.session_state["processed_file"] == "kept"


def test_initialize_session_state_initializes_blank_state(monkeypatch):
    session_state = SessionState()
    fake_streamlit = SimpleNamespace(session_state=session_state)
    monkeypatch.setattr(utils, "st", fake_streamlit)

    utils.initialize_session_state()
    session_id = session_state.session_id
    # Run again to exercise the "keys already exist" branch path
    utils.initialize_session_state()

    assert session_state.processed_file is None
    assert session_state.retriever is None
    assert isinstance(session_state.processed_file_ids, set)
    assert session_state.session_id == session_id


def test_get_file_key_and_format_file_size():
    class DummyUpload:
        def __init__(self, name: str, size: int):
            self.name = name
            self.size = size

    upload = DummyUpload("report.pdf", 2048)
    assert utils.get_file_key(upload) == "report.pdf_2048"

    assert utils.format_file_size(512) == "512 bytes"
    assert utils.format_file_size(2048) == "2.0 KB"
    assert utils.format_file_size(2 * 1024 * 1024) == "2.00 MB"
    assert utils.get_file_key(None) is None


def test_compute_file_hash_reads_when_getvalue_missing():
    class DummyFile:
        def __init__(self, data):
            self._data = data

        def read(self):
            return self._data

    dummy = DummyFile(b"abc123")
    assert utils.compute_file_hash(dummy) == utils.hashlib.sha256(b"abc123").hexdigest()


def test_get_files_key_handles_empty_and_sorted(monkeypatch):
    class DummyFile:
        def __init__(self, data: bytes):
            self._data = data

        def getvalue(self):
            return self._data

    first = DummyFile(b"b")
    second = DummyFile(b"a")
    key = utils.get_files_key([first, second])
    expected_hashes = sorted(
        [utils.compute_file_hash(first), utils.compute_file_hash(second)]
    )
    assert key == "|".join(expected_hashes)
    assert utils.get_files_key([]) == ""


def test_get_session_collection_name_defaults_without_session_id(monkeypatch):
    session_state = SessionState()
    fake_streamlit = SimpleNamespace(session_state=session_state)
    monkeypatch.setattr(utils, "st", fake_streamlit)

    assert utils.get_session_collection_name("collection") == "collection-default"
    session_state.session_id = "custom"
    assert utils.get_session_collection_name("collection") == "collection-custom"
