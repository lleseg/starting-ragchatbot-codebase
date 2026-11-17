"""Tests for AIGenerator's tool calling, with a mocked and a live Anthropic client."""

import os
from unittest.mock import MagicMock

import pytest

from ai_generator import AIGenerator
from config import config
from search_tools import CourseSearchTool
from tests.conftest import fake_response, text_block, tool_use_block

SEARCH_TOOL_DEF = CourseSearchTool(MagicMock()).get_tool_definition()

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
        assert "tool_choice" not in final_params

    def test_history_is_added_to_system_prompt(self, generator):
        generator.client.messages.create.return_value = fake_response(text_block("ok"))

        generator.generate_response(
            "q", conversation_history="User: hi\nAssistant: hey"
        )

        system = generator.client.messages.create.call_args.kwargs["system"]
        assert system.startswith(AIGenerator.SYSTEM_PROMPT)
        assert system.endswith("Previous conversation:\nUser: hi\nAssistant: hey")


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
