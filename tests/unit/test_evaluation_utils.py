from langchain_core.documents import Document

from evaluation import harness


def test_normalize_text_strips_punctuation_and_units():
    normalized = harness.normalize_text("  $39.331 Billion  ")
    assert normalized == "39331 b"


def test_matches_soft_detects_substring_or_regex():
    pred_norm = harness.normalize_text("$39.331 billion")
    expected_norm_list = [harness.normalize_text("39,331 million")]
    regex = [r"39331\s*(m|b)"]

    assert harness._matches_soft(pred_norm, expected_norm_list, regex) is True


def test_doc_hit_matches_expected_documents():
    docs = [
        Document(
            page_content="NVDA guidance details",
            metadata={"original_filename": "NVDA-Q4FY25-CFO-Commentary.pdf"},
        )
    ]
    assert harness._doc_hit(docs, ["nvda-q4fy25"]) is True
    assert harness._doc_hit(docs, ["chase-report"]) is False
