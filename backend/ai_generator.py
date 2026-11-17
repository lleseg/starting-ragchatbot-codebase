from typing import Any, Dict, List, Optional, Tuple

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Tool Usage:
- **search_course_content**: use **only** for questions about specific course content or detailed educational materials
- **get_course_outline**: use for questions about a course's outline, structure, syllabus or list of lessons
- **Up to 2 sequential tool calls per query**: make a second call only when the first result is needed to form it (e.g. get a course outline, then search for the topic of one of its lessons)
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives
- If a tool reports an error, say briefly that the information is unavailable; do not make up content

Outline Responses:
- When answering an outline question, always include:
  1. The course title
  2. The course link
  3. Every lesson, each with its lesson number and lesson title
- List all lessons in order; do not skip, merge or summarize them

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline questions**: Get the course outline first, then answer
- **Course content questions**: Search first, then answer
- **Multi-step questions**: Get the course outline first, then search using what it returned (e.g. a lesson title)
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    # Returned when Claude's final response contains no text block
    FALLBACK_RESPONSE = "I couldn't generate a response. Please try rephrasing."

    def __init__(self, api_key: str, model: str, max_tool_rounds: int = 2):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tool_rounds = max_tool_rounds

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Claude may use tools in up to ``max_tool_rounds`` sequential rounds, each
        a separate API call that sees the results of the previous rounds. After
        the last round, or when a tool fails, one more call is made with
        ``tool_choice`` set to none so that Claude answers with what it has.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        params = {**self.base_params, "system": system_content}

        # Tools are only offered when they can be executed
        if tools and tool_manager:
            params["tools"] = tools
            params["tool_choice"] = {"type": "auto"}

        messages = [{"role": "user", "content": query}]
        response = self.client.messages.create(**params, messages=messages)

        if "tools" not in params:
            return self._extract_text(response)

        for round_number in range(1, self.max_tool_rounds + 1):
            if not any(block.type == "tool_use" for block in response.content):
                break

            messages, failed = self._run_tool_round(response, messages, tool_manager)

            # Stop offering tools after the last round or a failed tool call
            last_round = failed or round_number == self.max_tool_rounds
            tool_choice = {"type": "none"} if last_round else {"type": "auto"}
            response = self.client.messages.create(
                **{**params, "tool_choice": tool_choice}, messages=messages
            )

            if last_round:
                break

        return self._extract_text(response)

    def _run_tool_round(
        self, response, messages: List[Dict[str, Any]], tool_manager
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Execute every tool call in a response and append the exchange to the history.

        Args:
            response: The response containing tool use requests
            messages: Conversation messages so far
            tool_manager: Manager to execute tools

        Returns:
            The new message list, and whether any tool call failed
        """
        tool_results = []
        failed = False

        # Every tool_use block needs a result, so keep going after a failure
        for block in response.content:
            if block.type != "tool_use":
                continue

            result = {"type": "tool_result", "tool_use_id": block.id}
            try:
                result["content"] = tool_manager.execute_tool(block.name, **block.input)
            except Exception as e:
                result["content"] = f"Tool '{block.name}' failed: {e}"
                result["is_error"] = True
                failed = True
            tool_results.append(result)

        new_messages = [
            *messages,
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": tool_results},
        ]
        return new_messages, failed

    def _extract_text(self, response) -> str:
        """Join the text blocks of a response, or return a fallback if it has none."""
        text = "".join(block.text for block in response.content if block.type == "text")
        return text or self.FALLBACK_RESPONSE
