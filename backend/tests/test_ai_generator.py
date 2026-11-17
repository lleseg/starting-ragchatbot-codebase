"""Tests for AIGenerator's tool calling, with a mocked and a live Anthropic client."""

import os
from unittest.mock import MagicMock, call

import pytest

from ai_generator import AIGenerator
from config import config
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from tests.conftest import fake_response, text_block, tool_use_block

SEARCH_TOOL_DEF = CourseSearchTool(MagicMock()).get_tool_definition()
OUTLINE_TOOL_DEF = CourseOutlineTool(MagicMock()).get_tool_definition()
TOOLS = [SEARCH_TOOL_DEF, OUTLINE_TOOL_DEF]

live = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1" or not config.ANTHROPIC_API_KEY,
    reason="set RUN_LIVE_TESTS=1 and ANTHROPIC_API_KEY to call the real API",
)


@pytest.fixture
def generator():
    gen = AIGenerator(api_key="test-key", model="test-model")
    gen.client = MagicMock()
    return gen


@pytest.fixture
def tool_manager():
    manager = MagicMock()
    manager.execute_tool.return_value = "[Course X - Lesson 1]\nEmbeddings are vectors."
    return manager


class TestToolCalling:
    def test_direct_answer_without_tool_use(self, generator, tool_manager):
        generator.client.messages.create.return_value = fake_response(text_block("4"))

        answer = generator.generate_response(
            "What is 2+2?", tools=[SEARCH_TOOL_DEF], tool_manager=tool_manager
        )

        assert answer == "4"
        tool_manager.execute_tool.assert_not_called()
        assert generator.client.messages.create.call_count == 1

    def test_first_call_offers_tools_with_auto_choice(self, generator, tool_manager):
        generator.client.messages.create.return_value = fake_response(text_block("ok"))

        generator.generate_response(
            "q", tools=[SEARCH_TOOL_DEF], tool_manager=tool_manager
        )

        params = generator.client.messages.create.call_args.kwargs
        assert params["tools"] == [SEARCH_TOOL_DEF]
        assert params["tool_choice"] == {"type": "auto"}
        assert params["messages"] == [{"role": "user", "content": "q"}]
        assert params["model"] == "test-model"

    def test_tool_use_executes_search_tool_with_claude_input(
        self, generator, tool_manager
    ):
        tool_input = {"query": "embeddings", "course_name": "Retrieval"}
        generator.client.messages.create.side_effect = [
            fake_response(
                tool_use_block("search_course_content", tool_input),
                stop_reason="tool_use",
            ),
            fake_response(text_block("Embeddings are vectors.")),
        ]

        answer = generator.generate_response(
            "What are embeddings?", tools=[SEARCH_TOOL_DEF], tool_manager=tool_manager
        )

        assert answer == "Embeddings are vectors."
        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="embeddings", course_name="Retrieval"
        )

    def test_second_call_carries_tool_result(self, generator, tool_manager):
        tool_block = tool_use_block("search_course_content", {"query": "q"}, "toolu_9")
        generator.client.messages.create.side_effect = [
            fake_response(tool_block, stop_reason="tool_use"),
            fake_response(text_block("final")),
        ]

        generator.generate_response(
            "q", tools=[SEARCH_TOOL_DEF], tool_manager=tool_manager
        )

        final_params = generator.client.messages.create.call_args_list[1].kwargs
        messages = final_params["messages"]
        assert messages[0] == {"role": "user", "content": "q"}
        assert messages[1] == {"role": "assistant", "content": [tool_block]}
        assert messages[2] == {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_9",
                    "content": tool_manager.execute_tool.return_value,
                }
            ],
        }
        assert final_params["tools"] == [SEARCH_TOOL_DEF]
        assert final_params["tool_choice"] == {"type": "auto"}

    def test_history_is_added_to_system_prompt(self, generator):
        generator.client.messages.create.return_value = fake_response(text_block("ok"))

        generator.generate_response(
            "q", conversation_history="User: hi\nAssistant: hey"
        )

        system = generator.client.messages.create.call_args.kwargs["system"]
        assert system.startswith(AIGenerator.SYSTEM_PROMPT)
        assert system.endswith("Previous conversation:\nUser: hi\nAssistant: hey")


def outline_call(block_id="toolu_1"):
    return tool_use_block("get_course_outline", {"course_title": "X"}, block_id)


def search_call(block_id="toolu_2"):
    return tool_use_block("search_course_content", {"query": "RAG"}, block_id)


def tool_response(*blocks):
    return fake_response(*blocks, stop_reason="tool_use")


def calls(generator):
    return [c.kwargs for c in generator.client.messages.create.call_args_list]


class TestSequentialToolCalling:
    def test_two_sequential_rounds_outline_then_search(self, generator, tool_manager):
        tool_manager.execute_tool.side_effect = ["Lesson 4: RAG", "[Course Y] RAG"]
        generator.client.messages.create.side_effect = [
            tool_response(outline_call()),
            tool_response(search_call()),
            fake_response(text_block("Course Y covers RAG.")),
        ]

        answer = generator.generate_response(
            "q", tools=TOOLS, tool_manager=tool_manager
        )

        assert answer == "Course Y covers RAG."
        assert tool_manager.execute_tool.call_args_list == [
            call("get_course_outline", course_title="X"),
            call("search_course_content", query="RAG"),
        ]
        final_messages = calls(generator)[2]["messages"]
        assert [m["role"] for m in final_messages] == [
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
        ]
        assert final_messages[2]["content"][0]["content"] == "Lesson 4: RAG"
        assert final_messages[4]["content"][0]["content"] == "[Course Y] RAG"

    def test_second_round_sees_first_tool_result_and_can_use_tools(
        self, generator, tool_manager
    ):
        tool_manager.execute_tool.side_effect = ["Lesson 4: RAG", "[Course Y] RAG"]
        generator.client.messages.create.side_effect = [
            tool_response(outline_call("toolu_a")),
            tool_response(search_call()),
            fake_response(text_block("done")),
        ]

        generator.generate_response("q", tools=TOOLS, tool_manager=tool_manager)

        second = calls(generator)[1]
        assert second["messages"][-1]["content"] == [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_a",
                "content": "Lesson 4: RAG",
            }
        ]
        assert second["tools"] == TOOLS
        assert second["tool_choice"] == {"type": "auto"}

    def test_final_call_after_max_rounds_forces_no_tools(self, generator, tool_manager):
        generator.client.messages.create.side_effect = [
            tool_response(outline_call()),
            tool_response(search_call()),
            fake_response(text_block("done")),
        ]

        generator.generate_response("q", tools=TOOLS, tool_manager=tool_manager)

        final = calls(generator)[2]
        assert final["tools"] == TOOLS
        assert final["tool_choice"] == {"type": "none"}

    def test_stops_after_max_rounds_even_if_claude_wants_more(
        self, generator, tool_manager
    ):
        generator.client.messages.create.side_effect = [
            tool_response(outline_call()),
            tool_response(search_call()),
            tool_response(text_block("Partial answer."), search_call("toolu_3")),
        ]

        answer = generator.generate_response(
            "q", tools=TOOLS, tool_manager=tool_manager
        )

        assert answer == "Partial answer."
        assert generator.client.messages.create.call_count == 3
        assert tool_manager.execute_tool.call_count == 2

    def test_stops_after_one_round_when_no_more_tool_use(self, generator, tool_manager):
        generator.client.messages.create.side_effect = [
            tool_response(search_call()),
            fake_response(text_block("answer")),
        ]

        answer = generator.generate_response(
            "q", tools=TOOLS, tool_manager=tool_manager
        )

        assert answer == "answer"
        assert generator.client.messages.create.call_count == 2
        tool_manager.execute_tool.assert_called_once()

    def test_tool_exception_returns_error_result_and_stops(
        self, generator, tool_manager
    ):
        tool_manager.execute_tool.side_effect = RuntimeError("store down")
        generator.client.messages.create.side_effect = [
            tool_response(search_call("toolu_e")),
            fake_response(text_block("Search is unavailable.")),
        ]

        answer = generator.generate_response(
            "q", tools=TOOLS, tool_manager=tool_manager
        )

        assert answer == "Search is unavailable."
        assert generator.client.messages.create.call_count == 2
        final = calls(generator)[1]
        assert final["tool_choice"] == {"type": "none"}
        assert final["messages"][-1]["content"] == [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_e",
                "content": "Tool 'search_course_content' failed: store down",
                "is_error": True,
            }
        ]

    def test_unknown_tool_counts_as_failure(self, generator):
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(MagicMock()))
        generator.client.messages.create.side_effect = [
            tool_response(tool_use_block("no_such_tool", {}, "toolu_u")),
            fake_response(text_block("Sorry.")),
        ]

        answer = generator.generate_response("q", tools=TOOLS, tool_manager=manager)

        assert answer == "Sorry."
        final = calls(generator)[1]
        assert final["tool_choice"] == {"type": "none"}
        result = final["messages"][-1]["content"][0]
        assert result["is_error"] is True
        assert "no_such_tool" in result["content"]

    def test_error_string_from_tool_is_not_failure(self, generator, tool_manager):
        tool_manager.execute_tool.side_effect = [
            "No course found matching 'X'",
            "[Course Y] RAG",
        ]
        generator.client.messages.create.side_effect = [
            tool_response(outline_call()),
            tool_response(search_call()),
            fake_response(text_block("done")),
        ]

        generator.generate_response("q", tools=TOOLS, tool_manager=tool_manager)

        assert tool_manager.execute_tool.call_count == 2
        assert calls(generator)[1]["tool_choice"] == {"type": "auto"}

    def test_multiple_tool_blocks_in_one_round(self, generator, tool_manager):
        tool_manager.execute_tool.side_effect = ["outline", "content"]
        generator.client.messages.create.side_effect = [
            tool_response(outline_call("toolu_a"), search_call("toolu_b")),
            fake_response(text_block("done")),
        ]

        generator.generate_response("q", tools=TOOLS, tool_manager=tool_manager)

        assert tool_manager.execute_tool.call_count == 2
        results = calls(generator)[1]["messages"][-1]["content"]
        assert [(r["tool_use_id"], r["content"]) for r in results] == [
            ("toolu_a", "outline"),
            ("toolu_b", "content"),
        ]

    def test_partial_failure_still_returns_results_for_all_blocks(
        self, generator, tool_manager
    ):
        tool_manager.execute_tool.side_effect = [RuntimeError("boom"), "content"]
        generator.client.messages.create.side_effect = [
            tool_response(outline_call("toolu_a"), search_call("toolu_b")),
            fake_response(text_block("done")),
        ]

        generator.generate_response("q", tools=TOOLS, tool_manager=tool_manager)

        final = calls(generator)[1]
        results = final["messages"][-1]["content"]
        assert [r["tool_use_id"] for r in results] == ["toolu_a", "toolu_b"]
        assert results[0]["is_error"] is True
        assert results[1] == {
            "type": "tool_result",
            "tool_use_id": "toolu_b",
            "content": "content",
        }
        assert final["tool_choice"] == {"type": "none"}

    def test_text_alongside_tool_use_is_kept_in_history(self, generator, tool_manager):
        thinking = text_block("Let me look that up.")
        tool_block = search_call()
        generator.client.messages.create.side_effect = [
            tool_response(thinking, tool_block),
            fake_response(text_block("answer")),
        ]

        answer = generator.generate_response(
            "q", tools=TOOLS, tool_manager=tool_manager
        )

        assert answer == "answer"
        assert calls(generator)[1]["messages"][1] == {
            "role": "assistant",
            "content": [thinking, tool_block],
        }

    def test_respects_custom_max_rounds(self, tool_manager):
        gen = AIGenerator(api_key="test-key", model="test-model", max_tool_rounds=1)
        gen.client = MagicMock()
        gen.client.messages.create.side_effect = [
            tool_response(search_call()),
            fake_response(text_block("done")),
        ]

        gen.generate_response("q", tools=TOOLS, tool_manager=tool_manager)

        assert gen.client.messages.create.call_count == 2
        assert calls(gen)[1]["tool_choice"] == {"type": "none"}


class TestResponseText:
    def test_returns_all_text_blocks_joined(self, generator):
        generator.client.messages.create.return_value = fake_response(
            text_block("Part one. "), text_block("Part two.")
        )

        assert generator.generate_response("q") == "Part one. Part two."

    def test_returns_fallback_when_no_text_block(self, generator, tool_manager):
        generator.client.messages.create.side_effect = [
            tool_response(search_call()),
            fake_response(),
        ]

        answer = generator.generate_response(
            "q", tools=TOOLS, tool_manager=tool_manager
        )

        assert answer == AIGenerator.FALLBACK_RESPONSE

    def test_without_tool_manager_does_not_send_tools(self, generator):
        generator.client.messages.create.return_value = fake_response(text_block("ok"))

        answer = generator.generate_response("q", tools=TOOLS)

        assert answer == "ok"
        params = generator.client.messages.create.call_args.kwargs
        assert "tools" not in params
        assert "tool_choice" not in params


@live
class TestLiveAnthropic:
    """Calls the real API with the configured model and request parameters."""

    def test_configured_model_answers(self):
        gen = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)

        answer = gen.generate_response("Reply with the single word: pong")

        assert answer.strip()

    def test_content_question_calls_search_tool(self, tool_manager):
        gen = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)

        answer = gen.generate_response(
            "Answer this question about course materials: "
            "What does lesson 1 of the Retrieval course say about embeddings?",
            tools=[SEARCH_TOOL_DEF],
            tool_manager=tool_manager,
        )

        tool_manager.execute_tool.assert_called_once()
        (name,) = tool_manager.execute_tool.call_args.args
        assert name == "search_course_content"
        assert "query" in tool_manager.execute_tool.call_args.kwargs
        assert answer.strip()

    def test_outline_then_search_multi_step(self):
        gen = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
        manager = MagicMock()

        def execute(name, **kwargs):
            if name == "get_course_outline":
                return (
                    "Course Title: Intro to Retrieval Testing\n"
                    "Course Link: https://example.com/retrieval\n"
                    "Lessons (4):\n"
                    "- Lesson 1: Embeddings\n"
                    "- Lesson 4: Reranking Search Results"
                )
            return "[Advanced Search - Lesson 2]\nRerankers reorder retrieved chunks."

        manager.execute_tool.side_effect = execute

        answer = gen.generate_response(
            "Answer this question about course materials: Find a course that "
            "discusses the same topic as lesson 4 of the Intro to Retrieval "
            "Testing course",
            tools=TOOLS,
            tool_manager=manager,
        )

        names = [c.args[0] for c in manager.execute_tool.call_args_list]
        assert names[0] == "get_course_outline"
        assert "search_course_content" in names
        assert answer.strip()
