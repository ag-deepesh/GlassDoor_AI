"""
GlassBox CLI -- every pipeline stage is runnable standalone from the terminal,
not just through the UI. Run `python -m cli.main --help` for all commands.
"""
from __future__ import annotations
from pathlib import Path
import json
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from core.parsing.assorted import parse_corpus, EXT_TO_METHOD
from core.registry import options as registry_options
from core.chunking.base import ChunkConfig
from core.pipeline import Pipeline
from core.pipeline_config import PipelineConfig
from core.schemas import StageError

load_dotenv()  # local, gitignored .env -- so --*-key envvar options pick it up

app = typer.Typer(help="GlassBox: a glass-box RAG/agent training lab.")
console = Console()


def _print_report(item) -> None:
    if isinstance(item, StageError):
        console.print(f"\n[bold red]✗ {item.stage} · {item.method} failed[/]")
        console.print(f"[red]{item.what_failed}[/]")
        console.print(f"[dim]→ {item.hint}[/]")
        return
    t = item.trace
    console.print(f"\n[bold cyan]── {item.stage} · {item.method} ──[/]")
    console.print(f"[dim]{t.input_summary} -> {t.output_summary}  ({t.latency_ms} ms"
                  + (f", {t.tokens} tok, ${t.cost_usd:.4f})" if t.tokens else ")") + "[/]")
    if item.eval_reference_free:
        console.print(f"[yellow]eval:[/] {item.eval_reference_free['scores']}")
        console.print(f"[yellow]→[/] {item.eval_reference_free['recommendation']}")
    if item.eval_with_reference:
        console.print(f"[green]eval (with reference):[/] {item.eval_with_reference['scores']}")
        console.print(f"[green]→[/] {item.eval_with_reference['recommendation']}")


@app.command()
def parse(
    input_dir: Path = typer.Argument(..., help="Folder of source documents (mixed formats OK)."),
    out_dir: Path = typer.Option(Path("out/parsed"), help="Where to write parsed JSON + extracted assets."),
    extract_images: bool = typer.Option(True, help="Extract embedded images for retrieval."),
    ocr: bool = typer.Option(False, help="Run Tesseract OCR on pages with little/no native text."),
):
    """Parse every supported file in INPUT_DIR (the 'Assorted' path)."""
    paths = [p for p in input_dir.iterdir() if p.suffix.lower() in EXT_TO_METHOD]
    if not paths:
        console.print(f"[yellow]No supported files found in {input_dir}. Supported: {sorted(EXT_TO_METHOD)}[/]")
        raise typer.Exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = out_dir / "assets"
    docs = parse_corpus(paths, extract_images=extract_images, ocr=ocr, assets_dir=assets_dir)

    table = Table(title=f"Parsed {len(docs)} document(s)")
    for col in ["doc_id", "format", "n_pages", "n_words", "n_images", "n_tables", "n_ocr_blocks"]:
        table.add_column(col)
    for doc in docs:
        s = doc.stats()
        table.add_row(*(str(s[c]) for c in ["doc_id", "format", "n_pages", "n_words", "n_images", "n_tables", "n_ocr_blocks"]))
        (out_dir / f"{doc.doc_id}.json").write_text(json.dumps({
            "doc_id": doc.doc_id, "format": doc.format, "n_pages": doc.n_pages,
            "text_blocks": [b.__dict__ for b in doc.text_blocks],
            "images": [{**i.__dict__, "path": str(i.path)} for i in doc.images],
            "tables": [t.__dict__ for t in doc.tables],
        }, indent=2))

    console.print(table)
    console.print(f"[green]Wrote parsed JSON + assets to {out_dir}[/]")


@app.command()
def chunk(
    input_dir: Path = typer.Argument(..., help="Folder of source documents."),
    method: str = typer.Option("recursive", help=f"One of: {registry_options('chunking') or 'see show-registry'}"),
    chunk_size: int = typer.Option(512, help="Target chunk size in approx. tokens."),
    overlap: int = typer.Option(64, help="Overlap between consecutive chunks, in approx. tokens."),
    ocr: bool = typer.Option(False),
):
    """Parse + chunk in one standalone step -- no embedding/API key needed."""
    from core.registry import get
    paths = [p for p in input_dir.iterdir() if p.suffix.lower() in EXT_TO_METHOD]
    docs = parse_corpus(paths, ocr=ocr, assets_dir=Path("out/assets"))
    chunker = get("chunking", method)(ChunkConfig(chunk_size=chunk_size, overlap=overlap))
    chunks = [c for doc in docs for c in chunker.chunk(doc)]

    table = Table(title=f"{len(chunks)} chunks via '{method}'")
    for col in ["chunk_id", "n_tokens", "preview"]:
        table.add_column(col)
    for c in chunks[:15]:
        table.add_row(c.chunk_id, str(c.n_tokens), c.text[:60].replace("\n", " ") + "...")
    console.print(table)
    if len(chunks) > 15:
        console.print(f"[dim]... and {len(chunks) - 15} more[/]")


@app.command()
def run(
    input_dir: Path = typer.Argument(..., help="Folder of source documents."),
    query: str = typer.Option(..., "--query", "-q", help="Question to answer."),
    reference: str = typer.Option(None, help="Gold answer, if you have one -- unlocks reference-based eval."),
    step: bool = typer.Option(False, help="Step-by-step: require Enter to acknowledge each stage before continuing."),
    claude_key: str = typer.Option(None, envvar="ANTHROPIC_API_KEY"),
    gemini_key: str = typer.Option(None, envvar="GOOGLE_API_KEY"),
    openai_key: str = typer.Option(None, envvar="OPENAI_API_KEY"),
    tavily_key: str = typer.Option(None, envvar="TAVILY_API_KEY"),
    groq_key: str = typer.Option(None, envvar="GROQ_API_KEY"),
    generation_method: str = typer.Option("claude-sonnet"),
    embedding_method: str = typer.Option("minilm-l6"),
    retrieval_method: str = typer.Option("hybrid-rrf"),
    top_k: int = typer.Option(10),
    web: bool = typer.Option(False, help="Blend live web search (Tavily) into retrieval, and re-query it every ReAct iteration."),
    react: bool = typer.Option(False, help="Enable the ReAct refinement loop between reranking and generation."),
    react_max_iterations: int = typer.Option(3, min=1, max=5),
    react_judge_method: str = typer.Option("gemini-2.5-flash", help=f"One of: {registry_options('judge') or 'see show-registry'}"),
):
    """Run the full pipeline: parsing -> chunking -> embedding -> retrieval
    -> [react] -> reranking -> generation, each stage reporting output +
    eval + recommendation. Add --step to require acknowledgment before
    advancing. This is a one-shot convenience wrapping Pipeline.build_kb()
    + Pipeline.answer_query() into a single call with an ad-hoc workdir --
    for a KB you'll query more than once, build it via the API instead."""
    cfg = PipelineConfig(
        embedding_method=embedding_method, retrieval_method=retrieval_method, top_k=top_k,
        generation_method=generation_method, web_enabled=web, react_enabled=react,
        react_max_iterations=react_max_iterations, react_judge_method=react_judge_method,
        api_keys={"claude": claude_key, "gemini": gemini_key, "openai": openai_key, "tavily": tavily_key, "groq": groq_key},
    )
    pipeline = Pipeline(cfg, workdir=Path("out/run"))

    def on_stage(item) -> bool:
        _print_report(item)
        if step and not isinstance(item, StageError):
            typer.confirm("Acknowledge and continue?", default=True, abort=True)
        return True

    reports = pipeline.run_all(input_dir, query, reference=reference, interactive=True, on_stage_reviewed=on_stage)
    if reports and isinstance(reports[-1], StageError):
        console.print("[dim]Earlier stage reports above are still valid -- this is where execution stopped.[/]")
        raise typer.Exit(1)

    console.print(f"\n[bold green]Answer:[/] {pipeline.answer}")


@app.command(name="show-registry")
def show_registry():
    """List every registered method per stage -- what each dropdown will show."""
    for stage in ["parsing", "chunking", "embedding", "retrieval", "reranking", "judge", "generation"]:
        opts = registry_options(stage)
        console.print(f"[bold]{stage}[/]: {opts if opts else '(none yet)'}")


if __name__ == "__main__":
    app()
