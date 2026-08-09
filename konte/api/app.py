"""FastAPI application for Konte contextual RAG."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, HTTPException

from konte import (
    ProjectConfig,
    RetrievalResponse,
    __version__,
    get_shared_project,
    list_projects,
    preload_projects,
    project_exists,
    settings,
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


def _preload_targets() -> list[str]:
    """Resolve settings.PRELOAD_PROJECTS into the project names to open."""
    requested = [part.strip() for part in settings.PRELOAD_PROJECTS.split(",") if part.strip()]
    return list_projects() if "*" in requested else requested


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Open the configured projects before the first request arrives."""
    targets = _preload_targets()
    if targets:
        loaded = preload_projects(targets)
        logger.info("projects_preloaded", requested=len(targets), loaded=len(loaded))

    yield


app = FastAPI(
    title="Konte API",
    description="Contextual RAG API with hybrid retrieval",
    version=__version__,
    lifespan=lifespan,
)


def resolve_project(name: str) -> Project:
    """Hand an endpoint the shared instance of the project it addresses.

    Sync on purpose: FastAPI runs a sync dependency in a worker thread, so a
    cold open does not stall the event loop.

    Args:
        name: Project name, taken from the path.

    Returns:
        The shared project, to query but not to modify.

    Raises:
        HTTPException: 404 when no project with that name exists.
    """
    if not project_exists(name):
        raise HTTPException(status_code=404, detail=f"Project not found: {name}")

    try:
        return get_shared_project(name)
    except FileNotFoundError as error:  # deleted between the check and the load
        raise HTTPException(status_code=404, detail=f"Project not found: {name}") from error


LoadedProject = Annotated[Project, Depends(resolve_project)]


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
def get_project_info(project: LoadedProject) -> ProjectConfig:
    """Get project configuration and info."""
    return project.config


@app.get("/projects/{name}/exists", response_model=ProjectExistsResponse)
def check_project_exists(name: str) -> ProjectExistsResponse:
    """Check if a project exists."""
    return ProjectExistsResponse(name=name, exists=project_exists(name))


@app.post("/projects/{name}/query", response_model=RetrievalResponse)
def query_project(project: LoadedProject, request: QueryRequest) -> RetrievalResponse:
    """Query a project for relevant chunks."""
    return project.query(
        query=request.query,
        mode=request.mode,
        top_k=request.top_k,
        use_keyword_extraction=request.use_keyword_extraction,
    )


@app.post("/projects/{name}/ask", response_model=AskResponse)
async def ask_project(project: LoadedProject, request: AskRequest) -> AskResponse:
    """Query a project and generate an LLM answer."""
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
