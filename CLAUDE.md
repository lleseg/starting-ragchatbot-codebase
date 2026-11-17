# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Dependencies are managed with `uv` (Python 3.13). There is no test suite or build step configured. Code quality uses `black` (formatting) and `ruff` (linting and import sorting), configured in `pyproject.toml` as dev dependencies.

Always use `uv` to run the server and to manage all dependencies. Add or remove packages with `uv add` / `uv remove` so `pyproject.toml` and `uv.lock` stay in sync, install with `uv sync`, and run anything Python with `uv run`. Never use `pip` directly.

```bash
uv sync                                                # install dependencies
./run.sh                                               # start the app on :8000
cd backend && uv run uvicorn app:app --reload --port 8000   # manual start
rm -rf backend/chroma_db                               # force re-ingestion of docs/ on next start
./scripts/format.sh                                    # auto-fix lint issues and format with black
./scripts/quality.sh                                   # check formatting and lint (no changes, non-zero exit on failure)
```

- Requires `ANTHROPIC_API_KEY` in `.env` at the repo root (see `.env.example`).
- The server must be started from inside `backend/`: `app.py` uses relative paths `../docs`, `../frontend`, and `./chroma_db`.
- Web UI at `http://localhost:8000`, OpenAPI docs at `/docs`.
- First startup downloads the `all-MiniLM-L6-v2` sentence-transformers model.

## Architecture

A RAG chatbot over course transcripts. FastAPI backend (`backend/`) serves a static vanilla-JS frontend (`frontend/`) and two endpoints: `POST /api/query` and `GET /api/courses`. All settings live in `backend/config.py`.

### Ingestion (on server startup)

`app.py` startup hook calls `RAGSystem.add_course_folder("../docs")`. Courses whose title already exists in ChromaDB are skipped, so edits to an existing doc are not picked up until `chroma_db` is cleared.

`DocumentProcessor` expects this plain-text format (`.pdf`/`.docx` are accepted by extension but read as text, with no real parsing):

```text
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <lesson title>
Lesson Link: <url>        (optional, must immediately follow the lesson line)
<lesson body...>
```

Lesson bodies are split into sentence-based chunks (`CHUNK_SIZE`/`CHUNK_OVERLAP`, in characters). The course title is the unique ID for a course everywhere.

`VectorStore` keeps two ChromaDB collections:

- `course_catalog`: one document per course (the title), with lessons serialized into a `lessons_json` metadata field. Used to resolve fuzzy course names to exact titles via semantic search.
- `course_content`: the chunks, with `course_title`, `lesson_number`, `chunk_index` metadata used as `where` filters.

### Query flow

1. `RAGSystem.query` loads session history from `SessionManager` (in-memory, lost on restart) and calls `AIGenerator.generate_response` with the tool definitions from `ToolManager`.
2. `AIGenerator` makes a first Claude call with `tool_choice: auto`. Conversation history is injected as text into the system prompt, not as message turns.
3. If Claude requests `search_course_content`, `CourseSearchTool` runs `VectorStore.search`: resolve `course_name` against the catalog (top-1 match), build a filter, query `course_content`.
4. Claude can use tools in up to `MAX_TOOL_ROUNDS` (2) sequential rounds, each a separate API call that sees the previous results. After the last round, or when a tool raises (an unknown tool name raises `KeyError`), the next call keeps `tools` but sets `tool_choice: none` to force a text answer. The failure is sent back as a `tool_result` with `is_error: true`. Error strings returned by tools (no results, search errors) are normal results.
5. Sources for the UI are passed out-of-band: `CourseSearchTool` stores them on `last_sources`, and `RAGSystem` reads and resets them via `ToolManager` after the response is generated. This is shared mutable state on a single tool instance.

New tools are added by subclassing `Tool` in `search_tools.py` (implement `get_tool_definition` and `execute`) and registering the tool with `ToolManager` in `RAGSystem.__init__`.

### Frontend

`frontend/script.js` posts `{query, session_id}` and stores the `session_id` returned by the first response. Assistant answers are rendered as markdown with `marked`.
