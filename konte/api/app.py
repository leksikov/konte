"""FastAPI application for Konte contextual RAG."""

import structlog
from fastapi import FastAPI, HTTPException

from konte import (
    get_project,
    list_projects,
    project_exists,
    RetrievalResponse,
    ProjectConfig,
)
from konte.api.schemas import (
    QueryRequest,
    AskRequest,
    AskResponse,
    ProjectListResponse,
    ProjectExistsResponse,
)

logger = structlog.get_logger()

app = FastAPI(
    title="Konte API",
    description="Contextual RAG API with hybrid retrieval",
    version="0.1.0",
)


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
    if not project_exists(name):
        raise HTTPException(status_code=404, detail=f"Project not found: {name}")
    project = get_project(name)
    return project.config


@app.get("/projects/{name}/exists", response_model=ProjectExistsResponse)
def check_project_exists(name: str) -> ProjectExistsResponse:
    """Check if a project exists."""
    exists = project_exists(name)
    return ProjectExistsResponse(name=name, exists=exists)


@app.post("/projects/{name}/query", response_model=RetrievalResponse)
def query_project(name: str, request: QueryRequest) -> RetrievalResponse:
    """Query a project for relevant chunks."""
    if not project_exists(name):
        raise HTTPException(status_code=404, detail=f"Project not found: {name}")

    project = get_project(name)
    return project.query(
        query=request.query,
        mode=request.mode,
        top_k=request.top_k,
    )


@app.post("/projects/{name}/ask", response_model=AskResponse)
async def ask_project(name: str, request: AskRequest) -> AskResponse:
    """Query a project and generate an LLM answer."""
    if not project_exists(name):
        raise HTTPException(status_code=404, detail=f"Project not found: {name}")

    project = get_project(name)
    response, answer = await project.query_with_answer(
        query=request.query,
        mode=request.mode,
        top_k=request.top_k,
        max_chunks=request.max_chunks,
    )
    return AskResponse(retrieval=response, answer=answer)


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)
