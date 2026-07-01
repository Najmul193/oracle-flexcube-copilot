"""Command-line interface for Oracle FLEXCUBE Copilot."""

import time
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from oracle_flexcube_copilot.chunking.strategy import SemanticSectionChunker
from oracle_flexcube_copilot.config import settings
from oracle_flexcube_copilot.embedding.cache import EmbeddingCache
from oracle_flexcube_copilot.embedding.engine import EmbeddingEngine
from oracle_flexcube_copilot.enrichment.service import DocumentEnrichmentService
from oracle_flexcube_copilot.evaluation.benchmark import RetrievalEvaluator
from oracle_flexcube_copilot.evaluation.dataset import load_dataset
from oracle_flexcube_copilot.indexing.entity_index import OracleEntityIndex
from oracle_flexcube_copilot.retrieval.entity import EntityRetriever
from oracle_flexcube_copilot.evaluation.report import generate_markdown_report
from oracle_flexcube_copilot.indexing.entity_index import OracleEntityIndex
from oracle_flexcube_copilot.indexing.indexer import ChromaIndexer
from oracle_flexcube_copilot.ingestion.service import DocumentIngestionService
from oracle_flexcube_copilot.llm import ConsoleAnswerFormatter, LLMConfig, RAGAnswerGenerator, StreamHandler
from oracle_flexcube_copilot.prompting.builder import RAGPromptBuilder
from oracle_flexcube_copilot.prompting.models import ContextConfig
from oracle_flexcube_copilot.retrieval.bm25 import BM25Indexer, BM25Retriever
from oracle_flexcube_copilot.retrieval.engine import VectorRetriever
from oracle_flexcube_copilot.retrieval.fusion import RRFFuser

console = Console()


@click.group()
def main() -> None:
    """Oracle FLEXCUBE Copilot CLI."""
    pass


@main.command()
@click.argument("query")
@click.option("--top-k", default=5, help="Number of results to return.")
def search(query: str, top_k: int) -> None:
    """Perform a pure vector search."""
    console.print(f"[bold blue]Searching for:[/bold blue] {query}")

    # Initialize components
    indexer = ChromaIndexer()
    cache = EmbeddingCache(cache_dir=settings.resolved_cache_dir)
    embedder = EmbeddingEngine(cache=cache)
    vector_retriever = VectorRetriever(embedder=embedder, indexer=indexer)
    bm25_retriever = BM25Retriever()
    fuser = RRFFuser()

    vector_results = vector_retriever.retrieve(query, top_k=top_k)
    bm25_results = bm25_retriever.retrieve(query, top_k=top_k)

    results = fuser.fuse([vector_results, bm25_results], top_k=top_k)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    for i, res in enumerate(results, 1):
        console.print(f"\n[bold green]{i}. {res.source_document}[/bold green] (Page {res.page})")
        console.print(f"Score: [bold cyan]{res.score:.4f}[/bold cyan] | Section: {res.heading}")
        console.print(
            f"Entities: {', '.join(res.oracle_entities) if res.oracle_entities else 'None'}"
        )
        console.print("-" * 40)


@main.command()
@click.argument("query")
@click.option("--top-k", default=5, help="Number of context chunks to retrieve.")
@click.option(
    "--mode",
    default="concise",
    type=click.Choice(["concise", "detailed", "expert"], case_sensitive=False),
    help="Answer mode: concise (default), detailed, or expert.",
)
@click.option("--stream/--no-stream", default=True, help="Stream tokens as they are generated.")
@click.option("--min-score", default=None, type=float, help="Minimum relevance score threshold.")
def ask(query: str, top_k: int, mode: str, stream: bool, min_score: float | None) -> None:
    """Ask a question — retrieves context from PDFs and answers using Qwen3.

    Three answer modes:
    \b
      concise  2-5 sentence answer with citations (default)
      detailed  Full explanation with steps and Oracle terminology
      expert   Technical details, screen names, field descriptions, cross-references
    """
    # Initialize retrieval pipeline
    indexer = ChromaIndexer()
    cache = EmbeddingCache(cache_dir=settings.resolved_cache_dir)
    embedder = EmbeddingEngine(cache=cache)
    vector_retriever = VectorRetriever(embedder=embedder, indexer=indexer)
    bm25_retriever = BM25Retriever()
    fuser = RRFFuser()

    # Initialize LLM pipeline
    llm_config = LLMConfig(
        base_url=settings.ollama_base_url,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        top_p=settings.llm_top_p,
        repeat_penalty=settings.llm_repeat_penalty,
        num_ctx=settings.llm_num_ctx,
        num_predict=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
    )
    generator = RAGAnswerGenerator()
    builder = RAGPromptBuilder()
    formatter = ConsoleAnswerFormatter()

    # 1. Retrieve context
    t_retrieval_start = time.perf_counter()
    with console.status("[bold yellow]Searching documentation...[/bold yellow]"):
        vector_results = vector_retriever.retrieve(query, top_k=top_k)
        bm25_results = bm25_retriever.retrieve(query, top_k=top_k)
        results = fuser.fuse([vector_results, bm25_results], top_k=top_k)
    t_retrieval_end = time.perf_counter()
    retrieval_time = t_retrieval_end - t_retrieval_start

    if not results:
        console.print("[yellow]No relevant documentation found. Cannot answer.[/yellow]")
        return

    # 2. Build structured prompt
    config = ContextConfig(
        max_tokens=settings.prompt_max_tokens,
        min_score=min_score if min_score is not None else settings.prompt_min_score,
    )
    prompt_request = builder.build(query, results, config=config)

    # 3. Generate or stream answer
    if stream:
        console.print(formatter.format_stream_start(query, mode))
        stream_handler = StreamHandler()
        token_stream = generator.stream(prompt_request, mode=mode)
        for token in stream_handler.handle(token_stream):
            console.print(token, end="", markup=False)
        console.print()

        # Build minimal response for footer
        answer_response = generator.generate(prompt_request, mode=mode)
        answer_response.answer = stream_handler.text
        answer_response.metadata.generation_time = 0.0  # streaming time not tracked
        answer_response.metadata.completion_tokens = stream_handler.token_count
        answer_response.metadata.total_tokens = (
            prompt_request.estimated_tokens + stream_handler.token_count
        )
        answer_response.metadata.retrieval_time = retrieval_time

        footer = formatter.format_stream_end(answer_response)
        console.print(footer)
    else:
        answer_response = generator.generate(prompt_request, mode=mode)
        answer_response.metadata.retrieval_time = retrieval_time
        output = formatter.format(answer_response, query)
        console.print(output)


@main.command()
@click.argument("query")
@click.option("--top-k", default=5, help="Number of context chunks to retrieve.")
@click.option("--max-tokens", default=None, type=int, help="Override the prompt token budget.")
@click.option("--min-score", default=None, type=float, help="Minimum relevance score threshold.")
@click.option("--show-citations/--no-citations", default=True, help="Show citations table.")
@click.option("--show-context/--no-context", default=True, help="Show context blocks.")
@click.option("--show-system/--no-system", default=True, help="Show system prompt.")
def prompt(
    query: str,
    top_k: int,
    max_tokens: int | None,
    min_score: float | None,
    show_citations: bool,
    show_context: bool,
    show_system: bool,
) -> None:
    """Build and display the prompt that would be sent to the LLM — no LLM call."""
    console.print(f"[bold blue]Building prompt for:[/bold blue] {query}\n")

    indexer = ChromaIndexer()
    cache = EmbeddingCache(cache_dir=settings.resolved_cache_dir)
    embedder = EmbeddingEngine(cache=cache)
    vector_retriever = VectorRetriever(embedder=embedder, indexer=indexer)
    bm25_retriever = BM25Retriever()
    entity_index = OracleEntityIndex()
    entity_retriever = EntityRetriever(entity_index=entity_index, indexer=indexer)
    fuser = RRFFuser()

    with console.status("[bold yellow]Searching documentation...[/bold yellow]"):
        vector_results = vector_retriever.retrieve(query, top_k=top_k)
        bm25_results = bm25_retriever.retrieve(query, top_k=top_k)
        entity_results = entity_retriever.retrieve(query, top_k=top_k)
        results = fuser.fuse([vector_results, bm25_results, entity_results], top_k=top_k)

    config = ContextConfig(
        max_tokens=max_tokens or settings.prompt_max_tokens,
        min_score=min_score if min_score is not None else settings.prompt_min_score,
    )
    builder = RAGPromptBuilder()
    prompt_request = builder.build(query, results, config=config)

    # Summary header
    budget_str = f"{prompt_request.estimated_tokens} / {config.max_tokens}"
    console.print(f"Estimated tokens: [bold cyan]{budget_str}[/bold cyan]")
    console.print(f"Context blocks: [bold green]{len(prompt_request.context_blocks)}[/bold green]")
    console.print()

    # Citations table
    if show_citations and prompt_request.citations:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#")
        table.add_column("Document")
        table.add_column("Section")
        table.add_column("Page")
        table.add_column("Score")
        for i, c in enumerate(prompt_request.citations, 1):
            table.add_row(
                str(i),
                c.document,
                c.section or "-",
                str(c.page),
                f"{c.score:.4f}",
            )
        console.print(table)
        console.print()

    if not results:
        console.print("[yellow]No relevant documentation found.[/yellow]")
        return

    # Assemble and display prompt sections
    prompt_parts: list[str] = []

    if show_system and prompt_request.system_prompt:
        prompt_parts.append(
            f"[bold underline]System Prompt[/bold underline]\n{prompt_request.system_prompt}"
        )

    if show_context and prompt_request.formatted_context:
        prompt_parts.append(
            f"[bold underline]Retrieved Context[/bold underline]\n{prompt_request.formatted_context}"
        )

    prompt_parts.append(
        f"[bold underline]User Question[/bold underline]\n{prompt_request.user_prompt}"
    )

    for part in prompt_parts:
        console.print(part)
        console.print()


@main.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--top-k", default=10, type=int, help="Evaluation depth (default 10).")
def benchmark(dataset_path: str, top_k: int) -> None:
    """Run the evaluation framework against a dataset.

    Measures the full hybrid pipeline: Vector + BM25 + Entity Index + RRF.
    """
    console.print(f"[bold blue]Running benchmark using dataset:[/bold blue] {dataset_path}")

    try:
        queries = load_dataset(dataset_path)
    except Exception as e:
        console.print(f"[bold red]Failed to load dataset:[/bold red] {e}")
        return

    indexer = ChromaIndexer()
    cache = EmbeddingCache(cache_dir=settings.resolved_cache_dir)
    embedder = EmbeddingEngine(cache=cache)
    evaluator = RetrievalEvaluator(embedder=embedder, indexer=indexer)

    with console.status("[bold green]Evaluating queries...[/bold green]"):
        metrics = evaluator.evaluate(queries, top_k=top_k)

    report = generate_markdown_report(metrics)
    console.print(report)


@main.command()
@click.argument("directory", type=click.Path(exists=True))
def index(directory: str) -> None:
    """Index a directory of PDF documents."""
    console.print(f"[bold blue]Starting indexation for directory:[/bold blue] {directory}")

    ingestor = DocumentIngestionService(manifest_dir=settings.resolved_cache_dir)
    enricher = DocumentEnrichmentService()
    chunker = SemanticSectionChunker(
        target_tokens=settings.chunk_size, overlap_tokens=settings.chunk_overlap
    )
    cache = EmbeddingCache(cache_dir=settings.resolved_cache_dir)
    embedder = EmbeddingEngine(cache=cache)
    indexer = ChromaIndexer()

    # 1. Ingest
    with console.status("[bold yellow]Ingesting PDFs...[/bold yellow]"):
        documents = ingestor.ingest_directory(data_dir=Path(directory))
    if not documents:
        console.print("[red]No documents found to index.[/red]")
        return
    console.print(f"[green]Ingested {len(documents)} documents.[/green]")

    # 2. Enrich & Chunk & Embed & Index (Pipeline)
    total_chunks_indexed = 0
    all_chunks = []

    for doc in documents:
        console.print(f"Processing [bold cyan]{doc.filename}[/bold cyan]...")

        # Enrich
        doc = enricher.enrich(doc)

        # Chunk
        chunks = chunker.chunk(doc)
        if not chunks:
            continue

        # Embed
        embedded_chunks, _ = embedder.embed_chunks(chunks)

        # Index
        metrics = indexer.add_chunks(embedded_chunks)
        total_chunks_indexed += metrics.chunks_added

        all_chunks.extend(chunks)

        console.print(f"  Indexed [bold green]{metrics.chunks_added}[/bold green] chunks.")

    console.print("\n[bold green]Building BM25 Sparse Index...[/bold green]")
    bm25_indexer = BM25Indexer()
    bm25_indexer.build(all_chunks)

    console.print(
        f"\n[bold green]Indexation complete![/bold green] Total chunks added: {total_chunks_indexed}"
    )


@main.command()
def stats() -> None:
    """Show database and index health statistics."""
    console.print("[bold blue]System Health Statistics[/bold blue]")

    indexer = ChromaIndexer()
    health = indexer.health_check()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Component")
    table.add_column("Status")
    table.add_column("Details")

    chroma_status = "[green]OK[/green]" if health.is_accessible else "[red]FAIL[/red]"
    chroma_details = f"Chunks: {health.total_chunks}" if health.is_accessible else str(health.error)

    table.add_row("ChromaDB Vector Index", chroma_status, chroma_details)

    # Check SQLite Entity Index
    try:
        entity_index = OracleEntityIndex(db_dir=indexer.db_dir)
        # We don't have a count method yet, but instantiation verifies DB accessibility
        table.add_row("SQLite Entity Index", "[green]OK[/green]", "Accessible")
    except Exception as e:
        table.add_row("SQLite Entity Index", "[red]FAIL[/red]", str(e))

    console.print(table)


if __name__ == "__main__":
    main()
