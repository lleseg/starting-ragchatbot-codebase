"""Tests for CourseSearchTool.execute, against mocked and real vector stores."""

from unittest.mock import MagicMock

import pytest

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


def make_results(docs, metas):
    return SearchResults(documents=docs, metadata=metas, distances=[0.1] * len(docs))


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.get_lesson_link.return_value = "https://example.com/lesson"
    return store


class TestExecuteWithMockStore:
    def test_passes_arguments_to_store(self, mock_store):
        mock_store.search.return_value = make_results([], [])
        tool = CourseSearchTool(mock_store)

        tool.execute("embeddings", course_name="Retrieval", lesson_number=2)

        mock_store.search.assert_called_once_with(
            query="embeddings", course_name="Retrieval", lesson_number=2
        )

    def test_formats_results_with_course_and_lesson_header(self, mock_store):
        mock_store.search.return_value = make_results(
            ["Chunk A", "Chunk B"],
            [
                {"course_title": "Course X", "lesson_number": 1},
                {"course_title": "Course X", "lesson_number": None},
            ],
        )
        tool = CourseSearchTool(mock_store)

        result = tool.execute("anything")

        assert result == "[Course X - Lesson 1]\nChunk A\n\n[Course X]\nChunk B"

    def test_records_sources_with_lesson_links(self, mock_store):
        mock_store.search.return_value = make_results(
            ["Chunk A", "Chunk B"],
            [
                {"course_title": "Course X", "lesson_number": 1},
                {"course_title": "Course X"},
            ],
        )
        tool = CourseSearchTool(mock_store)

        tool.execute("anything")

        assert tool.last_sources == [
            {"text": "Course X - Lesson 1", "link": "https://example.com/lesson"},
            {"text": "Course X", "link": None},
        ]
        mock_store.get_lesson_link.assert_called_once_with("Course X", 1)

    def test_sources_cite_each_lesson_once(self, mock_store):
        mock_store.search.return_value = make_results(
            ["Chunk A", "Chunk B", "Chunk C"],
            [
                {"course_title": "Course X", "lesson_number": 1},
                {"course_title": "Course X", "lesson_number": 2},
                {"course_title": "Course X", "lesson_number": 1},
            ],
        )
        tool = CourseSearchTool(mock_store)

        result = tool.execute("anything")

        assert [s["text"] for s in tool.last_sources] == [
            "Course X - Lesson 1",
            "Course X - Lesson 2",
        ]
        assert result.count("[Course X - Lesson 1]") == 2

    def test_returns_store_error_message(self, mock_store):
        mock_store.search.return_value = SearchResults.empty(
            "No course found matching 'Nope'"
        )
        tool = CourseSearchTool(mock_store)

        assert (
            tool.execute("x", course_name="Nope") == "No course found matching 'Nope'"
        )
        assert tool.last_sources == []

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({}, "No relevant content found."),
            ({"course_name": "C"}, "No relevant content found in course 'C'."),
            ({"lesson_number": 3}, "No relevant content found in lesson 3."),
            (
                {"course_name": "C", "lesson_number": 3},
                "No relevant content found in course 'C' in lesson 3.",
            ),
        ],
    )
    def test_empty_results_message(self, mock_store, kwargs, expected):
        mock_store.search.return_value = make_results([], [])
        tool = CourseSearchTool(mock_store)

        assert tool.execute("x", **kwargs) == expected

    def test_tool_manager_dispatches_by_name(self, mock_store):
        mock_store.search.return_value = make_results(
            ["Chunk"], [{"course_title": "C", "lesson_number": 0}]
        )
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_store))

        result = manager.execute_tool("search_course_content", query="q")

        assert result == "[C - Lesson 0]\nChunk"
        assert manager.get_last_sources()[0]["text"] == "C - Lesson 0"
        manager.reset_sources()
        assert manager.get_last_sources() == []

    def test_tool_manager_raises_for_unknown_tool(self):
        manager = ToolManager()

        with pytest.raises(KeyError, match="no_such_tool"):
            manager.execute_tool("no_such_tool", query="q")


class TestExecuteWithRealStore:
    """Runs the tool against real ChromaDB + embeddings with a sample course."""

    def test_finds_relevant_content(self, populated_vector_store):
        tool = CourseSearchTool(populated_vector_store)

        result = tool.execute("What are embeddings?")

        assert "[Intro to Retrieval Testing - Lesson" in result
        assert "Embeddings map text to dense vectors" in result

    def test_partial_course_name_and_lesson_filter(self, populated_vector_store):
        tool = CourseSearchTool(populated_vector_store)

        result = tool.execute(
            "metadata filters", course_name="Retrieval", lesson_number=2
        )

        assert result.startswith("[Intro to Retrieval Testing - Lesson 2]")
        assert "Lesson 1]" not in result
        assert tool.last_sources[0] == {
            "text": "Intro to Retrieval Testing - Lesson 2",
            "link": "https://example.com/retrieval/lesson-2",
        }

    def test_missing_lesson_returns_no_content(self, populated_vector_store):
        tool = CourseSearchTool(populated_vector_store)

        result = tool.execute("anything", lesson_number=99)

        assert result == "No relevant content found in lesson 99."
