"""API endpoint tests for /api/query, /api/courses and the static root."""


class TestQueryEndpoint:
    def test_query_without_session_creates_one(self, client, mock_rag_system, sample_sources):
        response = client.post("/api/query", json={"query": "What is computer use?"})

        assert response.status_code == 200
        data = response.json()
        assert data == {
            "answer": "Computer use lets Claude operate a desktop.",
            "sources": sample_sources,
            "session_id": "session_1",
        }
        mock_rag_system.session_manager.create_session.assert_called_once()
        mock_rag_system.query.assert_called_once_with("What is computer use?", "session_1")

    def test_query_with_existing_session_reuses_it(self, client, mock_rag_system):
        response = client.post(
            "/api/query", json={"query": "Tell me more", "session_id": "session_42"}
        )

        assert response.status_code == 200
        assert response.json()["session_id"] == "session_42"
        mock_rag_system.session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with("Tell me more", "session_42")

    def test_query_with_no_sources(self, client, mock_rag_system):
        mock_rag_system.query.return_value = ("Hello!", [])

        response = client.post("/api/query", json={"query": "Hi"})

        assert response.status_code == 200
        assert response.json()["sources"] == []

    def test_query_missing_query_field_returns_422(self, client, mock_rag_system):
        response = client.post("/api/query", json={"session_id": "session_1"})

        assert response.status_code == 422
        mock_rag_system.query.assert_not_called()

    def test_query_wrong_type_returns_422(self, client):
        response = client.post("/api/query", json={"query": 123})

        assert response.status_code == 422

    def test_query_invalid_json_returns_422(self, client):
        response = client.post(
            "/api/query", content="not json", headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_query_rag_error_returns_500(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("Anthropic API unavailable")

        response = client.post("/api/query", json={"query": "What is MCP?"})

        assert response.status_code == 500
        assert response.json()["detail"] == "Anthropic API unavailable"

    def test_query_get_is_not_handled(self, client, mock_rag_system):
        # Without a GET route, the request falls through to the "/" static
        # mount, which returns 404 rather than 405.
        response = client.get("/api/query")

        assert response.status_code == 404
        mock_rag_system.query.assert_not_called()


class TestCoursesEndpoint:
    def test_get_courses_returns_stats(self, client, sample_course_titles):
        response = client.get("/api/courses")

        assert response.status_code == 200
        assert response.json() == {
            "total_courses": 2,
            "course_titles": sample_course_titles,
        }

    def test_get_courses_empty_catalog(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }

        response = client.get("/api/courses")

        assert response.status_code == 200
        assert response.json() == {"total_courses": 0, "course_titles": []}

    def test_get_courses_error_returns_500(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = Exception("ChromaDB unavailable")

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert response.json()["detail"] == "ChromaDB unavailable"

    def test_get_courses_malformed_analytics_returns_500(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {"total_courses": 1}

        response = client.get("/api/courses")

        assert response.status_code == 500


class TestRootEndpoint:
    def test_root_serves_index_html(self, client):
        response = client.get("/")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "Course Materials Assistant" in response.text

    def test_static_asset_is_served(self, client):
        response = client.get("/script.js")

        assert response.status_code == 200
        assert "console.log" in response.text

    def test_unknown_static_path_returns_404(self, client):
        response = client.get("/does-not-exist.css")

        assert response.status_code == 404

    def test_api_routes_take_precedence_over_static_mount(self, client):
        response = client.get("/api/courses")

        assert response.headers["content-type"].startswith("application/json")
