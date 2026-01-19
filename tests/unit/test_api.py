"""Unit tests for API module."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError


@pytest.mark.unit
class TestQueryRequest:
    """Test QueryRequest validation."""

    def test_query_request_defaults(self):
        """Test QueryRequest with defaults."""
        from konte.api.schemas import QueryRequest

        req = QueryRequest(query="test query")
        assert req.query == "test query"
        assert req.mode == "hybrid"
        assert req.top_k is None

    def test_query_request_all_fields(self):
        """Test QueryRequest with all fields."""
        from konte.api.schemas import QueryRequest

        req = QueryRequest(query="test", mode="semantic", top_k=20)
        assert req.query == "test"
        assert req.mode == "semantic"
        assert req.top_k == 20

    def test_query_request_empty_query_fails(self):
        """Test QueryRequest validation rejects empty query."""
        from konte.api.schemas import QueryRequest

        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_query_request_invalid_mode_fails(self):
        """Test QueryRequest validation rejects invalid mode."""
        from konte.api.schemas import QueryRequest

        with pytest.raises(ValidationError):
            QueryRequest(query="test", mode="invalid")

    def test_query_request_top_k_bounds(self):
        """Test QueryRequest top_k bounds."""
        from konte.api.schemas import QueryRequest

        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k=0)

        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k=101)


@pytest.mark.unit
class TestAskRequest:
    """Test AskRequest validation."""

    def test_ask_request_defaults(self):
        """Test AskRequest with defaults."""
        from konte.api.schemas import AskRequest

        req = AskRequest(query="test question")
        assert req.query == "test question"
        assert req.mode == "hybrid"
        assert req.top_k is None
        assert req.max_chunks == 10

    def test_ask_request_max_chunks_bounds(self):
        """Test AskRequest max_chunks bounds."""
        from konte.api.schemas import AskRequest

        with pytest.raises(ValidationError):
            AskRequest(query="test", max_chunks=0)

        with pytest.raises(ValidationError):
            AskRequest(query="test", max_chunks=51)


@pytest.mark.unit
class TestAPIEndpoints:
    """Test API endpoint behavior with mocks."""

    def test_list_projects_empty(self):
        """Test listing projects when none exist."""
        with patch("konte.api.app.list_projects") as mock_list:
            mock_list.return_value = []

            from fastapi.testclient import TestClient

            from konte.api.app import app

            client = TestClient(app)
            response = client.get("/projects")

            assert response.status_code == 200
            data = response.json()
            assert data["projects"] == []
            assert data["total"] == 0

    def test_list_projects_with_projects(self):
        """Test listing projects."""
        with patch("konte.api.app.list_projects") as mock_list:
            mock_list.return_value = ["project1", "project2"]

            from fastapi.testclient import TestClient

            from konte.api.app import app

            client = TestClient(app)
            response = client.get("/projects")

            assert response.status_code == 200
            data = response.json()
            assert data["projects"] == ["project1", "project2"]
            assert data["total"] == 2

    def test_project_not_found(self):
        """Test 404 for non-existent project."""
        with patch("konte.api.app.project_exists") as mock_exists:
            mock_exists.return_value = False

            from fastapi.testclient import TestClient

            from konte.api.app import app

            client = TestClient(app)
            response = client.get("/projects/nonexistent")

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    def test_project_exists_true(self):
        """Test project exists endpoint returns true."""
        with patch("konte.api.app.project_exists") as mock_exists:
            mock_exists.return_value = True

            from fastapi.testclient import TestClient

            from konte.api.app import app

            client = TestClient(app)
            response = client.get("/projects/myproject/exists")

            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "myproject"
            assert data["exists"] is True

    def test_project_exists_false(self):
        """Test project exists endpoint returns false."""
        with patch("konte.api.app.project_exists") as mock_exists:
            mock_exists.return_value = False

            from fastapi.testclient import TestClient

            from konte.api.app import app

            client = TestClient(app)
            response = client.get("/projects/nonexistent/exists")

            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "nonexistent"
            assert data["exists"] is False

    def test_health_check(self):
        """Test health check endpoint."""
        from fastapi.testclient import TestClient

        from konte.api.app import app

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_query_project_not_found(self):
        """Test query returns 404 for non-existent project."""
        with patch("konte.api.app.project_exists") as mock_exists:
            mock_exists.return_value = False

            from fastapi.testclient import TestClient

            from konte.api.app import app

            client = TestClient(app)
            response = client.post(
                "/projects/nonexistent/query",
                json={"query": "test query"},
            )

            assert response.status_code == 404

    def test_query_project_success(self):
        """Test query endpoint success."""
        mock_response = MagicMock()
        mock_response.results = []
        mock_response.query = "test query"
        mock_response.total_found = 0
        mock_response.top_score = 0.0
        mock_response.score_spread = 0.0
        mock_response.has_high_confidence = False
        mock_response.suggested_action = "refine_query"
        mock_response.model_dump.return_value = {
            "results": [],
            "query": "test query",
            "total_found": 0,
            "top_score": 0.0,
            "score_spread": 0.0,
            "has_high_confidence": False,
            "suggested_action": "refine_query",
        }

        mock_project = MagicMock()
        mock_project.query.return_value = mock_response

        with patch("konte.api.app.project_exists") as mock_exists:
            with patch("konte.api.app.get_project") as mock_get:
                mock_exists.return_value = True
                mock_get.return_value = mock_project

                from fastapi.testclient import TestClient

                from konte.api.app import app

                client = TestClient(app)
                response = client.post(
                    "/projects/myproject/query",
                    json={"query": "test query", "mode": "hybrid", "top_k": 10},
                )

                assert response.status_code == 200
                mock_project.query.assert_called_once_with(
                    query="test query",
                    mode="hybrid",
                    top_k=10,
                )
