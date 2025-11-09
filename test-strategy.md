# Test Strategy

## Overview

98.03% line coverage, 96.43% branch coverage. Two-tier testing strategy: unit tests for component isolation, integration tests for end-to-end workflows.

## Test Structure

```
tests/
├── conftest.py          # Shared fixtures (Streamlit mock)
├── unit/                # Unit tests (9 files)
│   ├── test_config.py
│   ├── test_utils.py
│   ├── test_multimodal_loader.py
│   ├── test_document_loader.py
│   ├── test_document_processor.py
│   ├── test_rag_workflow.py
│   ├── test_evaluation_utils.py
│   ├── test_evaluation_harness.py
│   └── test_app_handlers.py
└── integration/         # Integration (workflow-level) tests (4 files)
    ├── test_ingest_vectorstore.py
    ├── test_rag_workflow_integ.py
    ├── test_eval_harness.py
    └── test_streamlit_flow.py
```

## Unit Tests

**Goal**: Validate individual component logic in isolation

**Approach**:
- Mock all external dependencies (OpenAI, Chroma, Streamlit, filesystem)
- Test business logic, error handling, edge cases
- Fast execution (<1s per test)

**Coverage**:

| Module | Tests | Focus |
|--------|-------|-------|
| `test_config.py` | Environment variable handling | LangChain tracing settings |
| `test_utils.py` | Session state, DB cleanup | Streamlit session initialization |
| `test_multimodal_loader.py` | File format support | PDF/DOCX/CSV/XLSX loading |
| `test_document_loader.py` | Upload handling | Metadata enrichment, temp file cleanup |
| `test_document_processor.py` | Chunking & vectorization | Incremental updates, error recovery |
| `test_rag_workflow.py` | Workflow nodes | Retrieve, grade, generate, check logic |
| `test_evaluation_utils.py` | Metric helpers | Text normalization, fuzzy matching |
| `test_evaluation_harness.py` | Eval infrastructure | Index building, score computation |
| `test_app_handlers.py` | UI event handlers | Question processing, state transitions |

**Key Patterns**:
- **Fake objects**: Lightweight mocks (`FakeStreamlit`, `FakeChroma`, `FakeRetriever`)
- **Monkeypatch isolation**: Replace module-level dependencies
- **State verification**: Assert inputs, outputs, side effects

## Integration Tests

**Goal**: Validate component interactions and end-to-end workflows

**Approach**:
- Mock external APIs (OpenAI, Chroma) but test real integration code
- Verify data flows between components
- Test stateful operations (session management, incremental updates)

**Coverage**:

| Test | Workflow | Validation |
|------|----------|------------|
| `test_ingest_vectorstore.py` | Upload → Chunk → Index | Incremental file updates, deduplication |
| `test_rag_workflow_integ.py` | Retrieve → Grade → Generate → Check | Full pipeline orchestration, metrics collection |
| `test_eval_harness.py` | Load eval docs → Build index → Run test cases | EM/Hit@1 calculation with real test_cases.json |
| `test_streamlit_flow.py` | User input → Process → Render | Async question processing, state resets |

**Key Patterns**:
- **Pipeline testing**: Chain multiple components
- **State machine validation**: Verify state transitions
- **Resource management**: Test cleanup, incremental updates

## Test Infrastructure

**Fixtures** (`conftest.py`):
- `patch_streamlit`: Injectable Streamlit mock

**Mocking Strategy**:
- **External APIs**: Always mocked (OpenAI, Chroma)
- **Filesystem**: Real for integration, mocked for unit
- **Streamlit**: Always mocked (no UI rendering in tests)

**Run Tests**:
```bash
# Fast: Unit + integration
pytest

# With coverage report
./scripts/run_tests.sh
```

## What We Don't Test

- OpenAI API responses (covered by LangChain)
- Chroma internals (covered by ChromaDB)
- Streamlit rendering (visual testing out of scope)
- Real document parsing accuracy (relies on LangChain loaders)

## Test Design Principles

1. **Fast feedback**: Unit tests run in <10s total
2. **Deterministic**: No randomness, fixed inputs/outputs
3. **Isolated**: Tests don't share state or depend on execution order
4. **Readable**: Descriptive test names, clear assertions
5. **Maintainable**: Fake objects over complex mocking frameworks

