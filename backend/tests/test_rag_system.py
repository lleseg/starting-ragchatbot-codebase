"""Tests for RAGSystem's handling of content questions.

The vector store, document processing and tools are real; only the Anthropic
client is mocked (except in the live tests).
"""

import os
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from config import config
from rag_system import RAGSystem
from tests.conftest import fake_response, text_block, tool_use_block

live = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1" or not config.ANTHROPIC_API_KEY,
    reason="set RUN_LIVE_TESTS=1 and ANTHROPIC_API_KEY to call the real API",
)


@pytest.fixture(scope="module")
def loaded_rag(tmp_path_factory, test_config, sample_course_file):
    """A RAGSystem with the sample course ingested from a docs folder."""
    rag_config = replace(test_config, CHROMA_PATH=str(tmp_path_factory.mktemp("rag")))
    rag = RAGSystem(rag_config)
    courses, chunks = rag.add_course_folder(str(sample_course_file.parent))
    assert courses == 1 and chunks > 0
    return rag


@pytest.fixture
def rag(loaded_rag):
    loaded_rag.ai_generator.client = MagicMock()
    loaded_rag.tool_manager.reset_sources()
    return loaded_rag


def search_then_answer(rag, tool_input, answer="Embeddings are dense vectors."):
    rag.ai_generator.client.messages.create.side_effect = [
        fake_response(
            tool_use_block("search_course_content", tool_input),
            stop_reason="tool_use",
        ),
        fake_response(text_block(answer)),
    ]


class TestContentQuery:
    def test_returns_answer_and_sources(self, rag):
        search_then_answer(rag, {"query": "embeddings", "lesson_number": 1})

        answer, sources = rag.query("What are embeddings?")

        assert answer == "Embeddings are dense vectors."
        assert sources
        assert all(
            s
            == {
                "text": "Intro to Retrieval Testing - Lesson 1",
                "link": "https://example.com/retrieval/lesson-1",
            }
            for s in sources
        )

    def test_search_results_reach_second_claude_call(self, rag):
        search_then_answer(rag, {"query": "embeddings", "course_name": "Retrieval"})

        rag.query("What are embeddings?")

        final_messages = rag.ai_generator.client.messages.create.call_args.kwargs[
            "messages"
        ]
        tool_result = final_messages[-1]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert "Embeddings map text to dense vectors" in tool_result["content"]

    def test_query_is_wrapped_and_tools_are_offered(self, rag):
        search_then_answer(rag, {"query": "embeddings"})

        rag.query("What are embeddings?")

        first = rag.ai_generator.client.messages.create.call_args_list[0].kwargs
        assert first["messages"][0]["content"] == (
            "Answer this question about course materials: What are embeddings?"
        )
        assert {t["name"] for t in first["tools"]} == {
            "search_course_content",
            "get_course_outline",
        }

    def test_sources_are_reset_after_query(self, rag):
        search_then_answer(rag, {"query": "embeddings"})
        rag.query("What are embeddings?")

        rag.ai_generator.client.messages.create.side_effect = None
        rag.ai_generator.client.messages.create.return_value = fake_response(
            text_block("4")
        )
        answer, sources = rag.query("What is 2+2?")

        assert answer == "4"
        assert sources == []

    def test_unknown_course_reports_no_match_to_claude(self, rag):
        # With a single course in the catalog, any name resolves to it, so use
        # an out-of-range lesson to exercise the empty-results path.
        search_then_answer(rag, {"query": "x", "lesson_number": 42}, answer="None.")

        answer, sources = rag.query("What is in lesson 42?")

        tool_result = rag.ai_generator.client.messages.create.call_args.kwargs[
            "messages"
        ][-1]["content"][0]
        assert tool_result["content"] == "No relevant content found in lesson 42."
        assert sources == []

    def test_session_history_is_used_on_follow_up(self, rag):
        session_id = rag.session_manager.create_session()
        search_then_answer(rag, {"query": "embeddings"})
        rag.query("What are embeddings?", session_id)

        search_then_answer(rag, {"query": "vector databases"}, answer="ChromaDB.")
        rag.query("And vector databases?", session_id)

        # Calls 0-1 answer the first question; call 2 starts the follow-up.
        system = rag.ai_generator.client.messages.create.call_args_list[2].kwargs[
            "system"
        ]
        assert "What are embeddings?" in system
        assert "Embeddings are dense vectors." in system

    def test_api_error_propagates(self, rag):
        # app.py turns this into a 500, which the frontend shows as "Query failed".
        rag.ai_generator.client.messages.create.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            rag.query("What are embeddings?")


@live
class TestLiveContentQuery:
    """End to end with the real API, configured model and the sample course."""

    def test_content_question_returns_answer_with_sources(
        self, tmp_path, sample_course_file
    ):
        rag = RAGSystem(replace(config, CHROMA_PATH=str(tmp_path)))
        rag.add_course_folder(str(sample_course_file.parent))

        answer, sources = rag.query(
            "In the Retrieval Testing course, what does lesson 1 say about embeddings?"
        )

        assert answer.strip()
        assert any("Intro to Retrieval Testing" in s["text"] for s in sources)
