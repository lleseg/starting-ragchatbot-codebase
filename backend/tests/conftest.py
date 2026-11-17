"""Shared fixtures for backend tests.

The production app in ``app.py`` builds a real ``RAGSystem`` (ChromaDB +
sentence-transformers) at import time and mounts ``../frontend`` as static
files, neither of which is available in the test environment. Instead,
``create_test_app`` mirrors the API endpoints from ``app.py`` against an
injected (mocked) RAG system and a temporary frontend directory.
"""

from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient
from pydantic import BaseModel

from models import Source


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    session_id: str


class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


def create_test_app(rag_system, frontend_dir: Optional[str] = None) -> FastAPI:
    """Build a FastAPI app with the same endpoints as app.py."""
    app = FastAPI(title="Course Materials RAG System (test)")

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()

            answer, sources = rag_system.query(request.query, session_id)

            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}", status_code=204)
    async def delete_session(session_id: str):
        rag_system.session_manager.delete_session(session_id)
        return Response(status_code=204)

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    if frontend_dir is not None:
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")

    return app


# Test data

@pytest.fixture
def sample_course_titles():
    return [
        "Building Towards Computer Use with Anthropic",
        "MCP: Build Rich-Context AI Apps with Anthropic",
    ]


@pytest.fixture
def sample_sources():
    return [
        {
            "text": "Building Towards Computer Use with Anthropic - Lesson 1",
            "link": "https://example.com/computer-use/lesson-1",
        },
        {
            "text": "Building Towards Computer Use with Anthropic - Lesson 2",
            "link": None,
        },
    ]


# Mocks

@pytest.fixture
def mock_rag_system(sample_course_titles, sample_sources):
    """A RAGSystem stand-in with canned responses."""
    rag = MagicMock()
    rag.session_manager.create_session.return_value = "session_1"
    rag.query.return_value = ("Computer use lets Claude operate a desktop.", sample_sources)
    rag.get_course_analytics.return_value = {
        "total_courses": len(sample_course_titles),
        "course_titles": sample_course_titles,
    }
    return rag


@pytest.fixture
def frontend_dir(tmp_path):
    """A minimal frontend directory to stand in for ../frontend."""
    (tmp_path / "index.html").write_text(
        "<!DOCTYPE html><html><head><title>Course Materials Assistant</title></head>"
        "<body><div id=\"chatMessages\"></div></body></html>"
    )
    (tmp_path / "script.js").write_text("console.log('test');")
    return tmp_path


@pytest.fixture
def test_app(mock_rag_system, frontend_dir):
    return create_test_app(mock_rag_system, str(frontend_dir))


@pytest.fixture
def client(test_app):
    return TestClient(test_app)


# Real components for search / generator / RAG tests

SAMPLE_COURSE_DOC = """Course Title: Intro to Retrieval Testing
Course Link: https://example.com/retrieval
Course Instructor: Ada Tester

Lesson 1: Embeddings
Lesson Link: https://example.com/retrieval/lesson-1
Embeddings map text to dense vectors. Similar sentences end up close together in vector space.

Lesson 2: Vector Databases
Lesson Link: https://example.com/retrieval/lesson-2
ChromaDB stores embeddings and supports metadata filters. Queries return the nearest chunks.
"""


@pytest.fixture(scope="session")
def sample_course_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("docs") / "retrieval_course.txt"
    path.write_text(SAMPLE_COURSE_DOC)
    return path


@pytest.fixture(scope="session")
def test_config(tmp_path_factory):
    """Production config pointed at a throwaway ChromaDB directory."""
    from config import Config

    return Config(
        ANTHROPIC_API_KEY="test-key",
        CHROMA_PATH=str(tmp_path_factory.mktemp("chroma")),
    )


@pytest.fixture(scope="session")
def populated_vector_store(test_config, sample_course_file):
    """A real VectorStore (ChromaDB + embeddings) holding the sample course."""
    from document_processor import DocumentProcessor
    from vector_store import VectorStore

    store = VectorStore(
        test_config.CHROMA_PATH, test_config.EMBEDDING_MODEL, test_config.MAX_RESULTS
    )
    processor = DocumentProcessor(test_config.CHUNK_SIZE, test_config.CHUNK_OVERLAP)
    course, chunks = processor.process_course_document(str(sample_course_file))
    store.add_course_metadata(course)
    store.add_course_content(chunks)
    return store


# Fake Anthropic responses


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_use_block(name, tool_input, block_id="toolu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


def fake_response(*blocks, stop_reason="end_turn"):
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason)
