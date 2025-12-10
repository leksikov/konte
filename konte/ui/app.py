"""Gradio UI for Konte contextual RAG."""

import gradio as gr

from konte import (
    get_project,
    list_projects,
    project_exists,
    RetrievalResponse,
    ProjectConfig,
)


def get_project_choices() -> list[str]:
    """Get list of available projects for dropdown."""
    projects = list_projects()
    return projects if projects else []


def format_results(response: RetrievalResponse) -> str:
    """Format retrieval results for display."""
    if not response.results:
        return "No results found."

    lines = [
        f"**Found {response.total_found} results** | "
        f"Top Score: {response.top_score:.3f} | "
        f"Suggested: {response.suggested_action}\n",
    ]

    for i, result in enumerate(response.results, 1):
        lines.append(f"---\n**[{i}] Score: {result.score:.3f}** | Source: {result.source}")

        if result.context:
            lines.append(f"\n*Context:* {result.context}")

        lines.append(f"\n{result.content}\n")

    return "\n".join(lines)


def format_config(config: ProjectConfig) -> str:
    """Format project config for display."""
    return f"""
**Name:** {config.name}

**Segmentation:**
- Segment size: {config.segment_size} tokens
- Segment overlap: {config.segment_overlap} tokens

**Chunking:**
- Chunk size: {config.chunk_size} tokens
- Chunk overlap: {config.chunk_overlap} tokens

**Models:**
- Embedding: {config.embedding_model}
- Context: {config.context_model}

**Indexes:**
- FAISS: {'Enabled' if config.enable_faiss else 'Disabled'}
- BM25: {'Enabled' if config.enable_bm25 else 'Disabled'}

**Fusion Weights:**
- Semantic: {config.fusion_weight_semantic}
- Lexical: {config.fusion_weight_lexical}
"""


def query_handler(
    project_name: str,
    query: str,
    mode: str,
    top_k: int,
) -> str:
    """Handle query request."""
    if not project_name:
        return "Please select a project."

    if not query.strip():
        return "Please enter a query."

    if not project_exists(project_name):
        return f"Project not found: {project_name}"

    project = get_project(project_name)
    response = project.query(query, mode=mode, top_k=top_k)
    return format_results(response)


async def ask_handler(
    project_name: str,
    query: str,
    mode: str,
    top_k: int,
    max_chunks: int,
) -> tuple[str, str]:
    """Handle ask request (returns answer and sources)."""
    if not project_name:
        return "Please select a project.", ""

    if not query.strip():
        return "Please enter a question.", ""

    if not project_exists(project_name):
        return f"Project not found: {project_name}", ""

    project = get_project(project_name)
    response, answer = await project.query_with_answer(
        query=query,
        mode=mode,
        top_k=top_k,
        max_chunks=max_chunks,
    )

    answer_text = (
        f"**Answer:**\n\n{answer.answer}\n\n---\n"
        f"*Model: {answer.model} | Sources used: {answer.sources_used}*"
    )
    sources_text = format_results(response)

    return answer_text, sources_text


def config_handler(project_name: str) -> str:
    """Get project configuration."""
    if not project_name:
        return "Please select a project."

    if not project_exists(project_name):
        return f"Project not found: {project_name}"

    project = get_project(project_name)
    return format_config(project.config)


def refresh_projects() -> gr.Dropdown:
    """Refresh the project list."""
    choices = get_project_choices()
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def create_interface() -> gr.Blocks:
    """Create the Gradio interface."""
    with gr.Blocks(title="Konte - Contextual RAG", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# Konte - Contextual RAG")
        gr.Markdown("Query your document projects with hybrid semantic + lexical search.")

        with gr.Row():
            project_dropdown = gr.Dropdown(
                choices=get_project_choices(),
                label="Select Project",
                scale=3,
            )
            refresh_btn = gr.Button("Refresh", scale=1)

        with gr.Tabs():
            # Query Tab
            with gr.TabItem("Query"):
                with gr.Row():
                    with gr.Column(scale=2):
                        query_input = gr.Textbox(
                            label="Query",
                            placeholder="Enter your search query...",
                            lines=2,
                        )
                        with gr.Row():
                            mode_select = gr.Radio(
                                choices=["hybrid", "semantic", "lexical"],
                                value="hybrid",
                                label="Retrieval Mode",
                            )
                            top_k_slider = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=10,
                                step=1,
                                label="Top K Results",
                            )
                        query_btn = gr.Button("Search", variant="primary")

                    with gr.Column(scale=3):
                        query_output = gr.Markdown(label="Results")

            # Ask Tab (RAG with LLM answer)
            with gr.TabItem("Ask (RAG)"):
                with gr.Row():
                    with gr.Column(scale=2):
                        ask_input = gr.Textbox(
                            label="Question",
                            placeholder="Ask a question about your documents...",
                            lines=2,
                        )
                        with gr.Row():
                            ask_mode_select = gr.Radio(
                                choices=["hybrid", "semantic", "lexical"],
                                value="hybrid",
                                label="Retrieval Mode",
                            )
                        with gr.Row():
                            ask_top_k = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=10,
                                step=1,
                                label="Top K Chunks",
                            )
                            max_chunks_slider = gr.Slider(
                                minimum=1,
                                maximum=20,
                                value=5,
                                step=1,
                                label="Max Chunks for Answer",
                            )
                        ask_btn = gr.Button("Ask", variant="primary")

                    with gr.Column(scale=3):
                        answer_output = gr.Markdown(label="Answer")
                        with gr.Accordion("Sources", open=False):
                            sources_output = gr.Markdown()

            # Config Tab
            with gr.TabItem("Project Config"):
                config_btn = gr.Button("Load Config")
                config_output = gr.Markdown()

        # Event handlers
        refresh_btn.click(
            fn=refresh_projects,
            outputs=[project_dropdown],
        )

        query_btn.click(
            fn=query_handler,
            inputs=[project_dropdown, query_input, mode_select, top_k_slider],
            outputs=[query_output],
        )

        ask_btn.click(
            fn=ask_handler,
            inputs=[project_dropdown, ask_input, ask_mode_select, ask_top_k, max_chunks_slider],
            outputs=[answer_output, sources_output],
        )

        config_btn.click(
            fn=config_handler,
            inputs=[project_dropdown],
            outputs=[config_output],
        )

    return interface


def launch(
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False,
) -> None:
    """Launch the Gradio interface."""
    interface = create_interface()
    interface.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
    )
