from dataclasses import dataclass
from typing import List, Optional
from rich.console import Console
from rich.panel import Panel

@dataclass
class Span:
    start: int
    end: int
    line: int
    column: int

    def __repr__(self):
        return f"{self.line}:{self.column}"

@dataclass
class Diagnostic:
    message: str
    span: Span
    level: str = "error"  # error, warning, info
    hint: Optional[str] = None

class DiagnosticEngine:
    def __init__(self):
        self.diagnostics: List[Diagnostic] = []
        self.has_errors = False
        try:
            from rich.console import Console
            self.console = Console()
            self.use_rich = True
        except ImportError:
            self.console = None
            self.use_rich = False

    def report(self, level: str, message: str, span: Span, hint: Optional[str] = None):
        diag = Diagnostic(message, span, level, hint)
        self.diagnostics.append(diag)
        if level == "error":
            self.has_errors = True
        
        if self.use_rich:
            color = "red" if level == "error" else "yellow"
            self.console.print(f"[{color} bold]{level.upper()}:[/] {message} at {span}")
            if hint:
                self.console.print(f"  [blue]Hint:[/blue] {hint}")
        else:
            print(f"{level.upper()}: {message} at {span}")
            if hint:
                print(f"  Hint: {hint}")

    def error(self, message: str, span: Span, hint: Optional[str] = None):
        self.report("error", message, span, hint)
