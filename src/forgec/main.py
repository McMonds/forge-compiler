import typer
from rich.console import Console
from forgec.pipeline import CompilerPipeline

app = typer.Typer(
    name="forgec",
    help="The Aether Programming Language Compiler",
    add_completion=False,
)
console = Console()

@app.command()
def compile(
    source_file: str = typer.Argument(..., help="Path to the source file"),
    visualize: bool = typer.Option(False, "--visualize", "-v", help="Generate visualization data"),
):
    """
    Compile an Aether source file.
    """
    pipeline = CompilerPipeline(source_file, visualize=visualize)
    try:
        pipeline.run()
        console.print(f"[green]Successfully compiled {source_file}[/green]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
