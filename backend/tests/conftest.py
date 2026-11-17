"""Shared fixtures for backend tests.

The production app in ``app.py`` builds a real ``RAGSystem`` (ChromaDB +
sentence-transformers) at import time and mounts ``../frontend`` as static
files, neither of which is available in the test environment. Instead,
``create_test_app`` mirrors the API endpoints from ``app.py`` against an
injected (mocked) RAG system and a temporary frontend directory.
"""

from typing import List, Optional
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException
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
