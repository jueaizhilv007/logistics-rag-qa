"""rag-qa 命令行入口。

所有子命令先 load_builtin_plugins() 触发内置实现注册，
否则配置驱动路径会在 from_config 阶段报 UnknownKindError。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    help="rag-qa · 物流规则问答内核命令行工具",
    no_args_is_help=True,
    invoke_without_command=True,
)
console = Console()


@app.callback()
def _main(ctx: typer.Context) -> None:
    """顶层入口。无子命令时显示帮助。"""
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


def _ensure_plugins() -> None:
    from rag_core.plugins import load_builtin_plugins

    load_builtin_plugins()


def _resolve_config(config: Optional[Path]) -> str | None:
    return str(config) if config is not None else None


@app.command()
def ingest(
    docs: Path = typer.Option(
        Path("data/docs"), "--docs", "-d", help="语料目录或单个文件"
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="config.yaml 路径，默认向上查找"
    ),
    drop: bool = typer.Option(False, "--drop", help="清空已有索引后重建"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只切分不写入"),
    sample: int = typer.Option(5, "--sample", help="dry-run 模式下打印的 chunk 样本数"),
) -> None:
    """把语料目录下的文档写入 LanceDB 索引。"""
    _ensure_plugins()
    from rag_core.container import RagCore

    core, _cfg, _secrets = RagCore.from_config(_resolve_config(config))
    assert core.ingest is not None  # from_config 路径一定装配

    if drop and not dry_run:
        core.store.clear()
        console.print("[yellow]已清空 collection，准备重建索引[/yellow]")

    # 收集要处理的文件
    if docs.is_file():
        candidates = [docs]
    elif docs.is_dir():
        candidates = sorted(p for p in docs.rglob("*") if p.is_file())
    else:
        console.print(f"[red]路径不存在：{docs}[/red]")
        raise typer.Exit(1)

    supported = core.ingest.supported_extensions()
    candidates = [p for p in candidates if p.suffix.lower() in supported]
    if not candidates:
        console.print(f"[yellow]未在 {docs} 找到支持的文件（支持：{sorted(supported)}）[/yellow]")
        raise typer.Exit(0)

    counts = {"indexed": 0, "skipped": 0, "error": 0}
    for path in candidates:
        if dry_run:
            result, chunks, _doc = core.ingest.prepare(path)
            label = {
                "indexed": "[cyan]ready[/cyan]",
                "skipped": "[yellow]skipped[/yellow]",
                "error": "[red]error[/red]",
            }[result.status]
            console.print(f"  {label} {path.name}: {result.reason or f'{result.chunks} chunks'}")
            if chunks:
                shown = chunks[:sample]
                for c in shown:
                    console.print(
                        f"    [dim]#{c.chunk_index:03d}[/dim] "
                        f"[bold]{c.heading_path or '(无标题)'}[/bold] "
                        f"embed_text[0:60]={c.embed_text[:60]!r}"
                    )
            counts[result.status] += 1
            continue

        result = core.ingest.run(path)
        label = {
            "indexed": "[green]indexed[/green]",
            "skipped": "[yellow]skipped[/yellow]",
            "error": "[red]error[/red]",
        }[result.status]
        console.print(f"  {label} {path.name}: {result.reason or f'{result.chunks} chunks'}")
        counts[result.status] += 1

    if not dry_run:
        core.store.optimize()

    summary = Table(title="入库汇总", show_header=False)
    summary.add_column("状态")
    summary.add_column("数量", justify="right")
    for status, n in counts.items():
        summary.add_row(status, str(n))
    console.print(summary)


@app.command()
def search(
    query: str = typer.Argument(..., help="检索语句"),
    top_k: Optional[int] = typer.Option(
        None, "--top-k", "-k", help="返回条数，默认取配置 retrieval.top_k"
    ),
    mode: Optional[str] = typer.Option(
        None, "--mode", "-m", help="临时覆盖 retrieval.mode：vector / hybrid"
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="config.yaml 路径，默认向上查找"
    ),
) -> None:
    """在索引中检索，打印三路排名、来源与正文片段。

    三路排名并排显示是为了让「这条结果为什么排第一」可以直接看出来：
    只在向量路出现、只在 BM25 路出现、两路都出现，三种情况的处置完全不同。
    """
    _ensure_plugins()
    from rag_core.container import RagCore

    core, cfg, _secrets = RagCore.from_config(
        _resolve_config(config), retrieval_mode=mode
    )
    assert core.retriever is not None  # from_config 路径一定装配

    console.print(
        f"[dim]策略标签 {core.describe()} · 模式 {cfg.retrieval.mode} · "
        f"索引 {core.store.count()} 条[/dim]"
    )
    results = core.retriever.retrieve(query, top_k)
    if not results:
        console.print("[yellow]没有命中。若索引为空，先执行 rag-qa ingest[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"检索结果 · {query}")
    table.add_column("#", justify="right")
    table.add_column("vec", justify="right")
    table.add_column("bm25", justify="right")
    table.add_column("fused", justify="right")
    table.add_column("来源")
    table.add_column("标题链")
    table.add_column("正文", overflow="fold")

    for hit in results:
        table.add_row(
            str(hit.final_rank),
            "-" if hit.vector_rank is None else str(hit.vector_rank),
            "-" if hit.bm25_rank is None else str(hit.bm25_rank),
            f"{hit.fused_score:.4f}",
            Path(hit.chunk.source).name,
            hit.chunk.heading_path or "-",
            hit.chunk.text[:100].replace("\n", " "),
        )
    console.print(table)


if __name__ == "__main__":
    app()
