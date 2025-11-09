from contextlib import nullcontext
from types import SimpleNamespace

import importlib
from langchain_core.documents import Document

import os

os.environ.setdefault("OPENAI_API_KEY", "test")
app = importlib.import_module("app")


class SessionState(dict):
    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value


class FakeExpander:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def markdown(self, *args, **kwargs):
        pass

    def dataframe(self, *args, **kwargs):
        pass


class FakeStreamlit:
    def __init__(self, initial_state=None):
        self.session_state = SessionState()
        if initial_state:
            self.session_state.update(initial_state)
        self.success_messages = []
        self.warning_messages = []
        self.tables = []
        self.rerun_called = False

    def markdown(self, *args, **kwargs):
        pass

    def success(self, message):
        self.success_messages.append(message)

    def caption(self, *args, **kwargs):
        pass

    def warning(self, message):
        self.warning_messages.append(message)

    def table(self, data):
        self.tables.append(data)

    def write(self, *args, **kwargs):
        pass

    def expander(self, *args, **kwargs):
        return FakeExpander()

    def dataframe(self, *args, **kwargs):
        pass

    def spinner(self, *args, **kwargs):
        return nullcontext()

    def rerun(self):
        self.rerun_called = True


class FakePlaceholder:
    def container(self):
        return nullcontext()


class FakeWorkflow:
    def __init__(self):
        self.questions = []

    def process_question(self, question):
        self.questions.append(question)
        return {
            "solution": f"Result:{question}[1]",
            "documents": [
                Document(page_content="Doc content", metadata={"original_filename": "sample.pdf"}),
            ],
            "document_evaluations": [
                SimpleNamespace(score="yes", relevance_score=1.0, coverage_assessment="full")
            ],
            "document_relevance_score": SimpleNamespace(binary_score=True, reasoning="ok"),
            "question_relevance_score": SimpleNamespace(reasoning="ok"),
        }


def test_handle_question_processing_renders_answer(monkeypatch):
    fake_st = FakeStreamlit()
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "rag_workflow", FakeWorkflow())

    app.handle_question_processing(
        "Explain revenue?",
        FakePlaceholder(),
        FakePlaceholder(),
        FakePlaceholder(),
    )

    assert fake_st.success_messages[-1].startswith("Result:Explain revenue?")
    assert fake_st.session_state["last_result"]["solution"].startswith("Result:")


def test_handle_user_interaction_with_pending_question(monkeypatch):
    initial_state = {"processing": True, "pending_question": "Explain guidance", "question_input": "temp"}
    fake_st = FakeStreamlit(initial_state=initial_state)
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "rag_workflow", FakeWorkflow())

    app.handle_user_interaction(
        uploaded_files=[object()],
        answer_placeholder=FakePlaceholder(),
        sources_placeholder=FakePlaceholder(),
        metrics_placeholder=FakePlaceholder(),
    )

    assert fake_st.session_state["processing"] is False
    assert fake_st.session_state["pending_question"] is None
    assert fake_st.session_state["question_input"] == ""
    assert fake_st.rerun_called is True
    assert fake_st.session_state["last_result"]["solution"].startswith("Result:Explain guidance")
