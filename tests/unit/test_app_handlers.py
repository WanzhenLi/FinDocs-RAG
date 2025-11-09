from contextlib import nullcontext
import importlib
import os

os.environ.setdefault("OPENAI_API_KEY", "test")
app = importlib.import_module("app")


class FakePlaceholder:
    def container(self):
        return nullcontext()


class SessionState(dict):
    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value


class FakeStreamlit:
    def __init__(self):
        self.session_state = SessionState()
        self.rerun_called = False

    def markdown(self, *args, **kwargs):
        pass

    def caption(self, *args, **kwargs):
        pass

    def success(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def spinner(self, *args, **kwargs):
        return nullcontext()

    def table(self, *args, **kwargs):
        pass

    def rerun(self):
        self.rerun_called = True


class FakeWorkflow:
    def __init__(self):
        self.questions = []

    def process_question(self, question):
        self.questions.append(question)
        return {"solution": f"Ans:{question}", "documents": []}


def test_handle_question_processing_invokes_workflow(monkeypatch):
    fake_st = FakeStreamlit()
    monkeypatch.setattr(app, "st", fake_st)
    fake_workflow = FakeWorkflow()
    monkeypatch.setattr(app, "rag_workflow", fake_workflow)
    recorded = {}
    monkeypatch.setattr(app, "_render_answer_sources_metrics", lambda result, *_: recorded.setdefault("result", result))

    app.handle_question_processing(
        "What happened?",
        FakePlaceholder(),
        FakePlaceholder(),
        FakePlaceholder(),
    )

    assert fake_workflow.questions == ["What happened?"]
    assert fake_st.session_state["last_result"]["solution"].startswith("Ans")
    assert recorded["result"]["solution"].startswith("Ans")


def test_handle_user_interaction_processes_pending_question(monkeypatch):
    fake_st = FakeStreamlit()
    fake_st.session_state.update({"processing": True, "pending_question": "Next question", "question_input": "old"})
    monkeypatch.setattr(app, "st", fake_st)

    captured = {}

    def fake_handle_question_processing(question, *_):
        captured["question"] = question

    monkeypatch.setattr(app, "handle_question_processing", fake_handle_question_processing)

    app.handle_user_interaction(
        uploaded_files=[object()],
        answer_placeholder=FakePlaceholder(),
        sources_placeholder=FakePlaceholder(),
        metrics_placeholder=FakePlaceholder(),
    )

    assert captured["question"] == "Next question"
    assert fake_st.session_state["processing"] is False
    assert fake_st.session_state["pending_question"] is None
    assert fake_st.session_state["question_input"] == ""
    assert fake_st.rerun_called is True
