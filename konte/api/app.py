"""FastAPI application for Konte contextual RAG."""

import structlog
from fastapi import FastAPI, HTTPException

from konte import (
    ProjectConfig,
    RetrievalResponse,
    __version__,
    get_project,
    list_projects,
    project_exists,
)
from konte.api.schemas import (
    AskRequest,
    AskResponse,
    ProjectExistsResponse,
    ProjectListResponse,
    QueryRequest,
)
from konte.project import Project

logger = structlog.get_logger()

app = FastAPI(
    title="Konte API",
    description="Contextual RAG API with hybrid retrieval",
    version=__version__,
)


def _load_project(name: str) -> Project:
    """Load a project by name.

    Args:
        name: Project name.

    Returns:
        The loaded project.

    Raises:
        HTTPException: 404 when no project with that name exists.
    """
    if not project_exists(name):
        raise HTTPException(status_code=404, detail=f"Project not found: {name}")
    return get_project(name)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/projects", response_model=ProjectListResponse)
def list_all_projects() -> ProjectListResponse:
    """List all available projects."""
    projects = list_projects()
    return ProjectListResponse(projects=projects, total=len(projects))


@app.get("/projects/{name}", response_model=ProjectConfig)
def get_project_info(name: str) -> ProjectConfig:
    """Get project configuration and info."""
    return _load_project(name).config


@app.get("/projects/{name}/exists", response_model=ProjectExistsResponse)
def check_project_exists(name: str) -> ProjectExistsResponse:
    """Check if a project exists."""
    return ProjectExistsResponse(name=name, exists=project_exists(name))


@app.post("/projects/{name}/query", response_model=RetrievalResponse)
def query_project(name: str, request: QueryRequest) -> RetrievalResponse:
    """Query a project for relevant chunks."""
    return _load_project(name).query(
        query=request.query,
        mode=request.mode,
        top_k=request.top_k,
        use_keyword_extraction=request.use_keyword_extraction,
    )


@app.post("/projects/{name}/ask", response_model=AskResponse)
async def ask_project(name: str, request: AskRequest) -> AskResponse:
    """Query a project and generate an LLM answer."""
    project = _load_project(name)
    response, answer = await project.query_with_answer(
        query=request.query,
        mode=request.mode,
        top_k=request.top_k,
        max_chunks=request.max_chunks,
        use_keyword_extraction=request.use_keyword_extraction,
    )
    return AskResponse(retrieval=response, answer=answer)


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)
