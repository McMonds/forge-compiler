from pathlib import Path
from typing import Optional, Dict, Any
import json

from forgec.lexer import Lexer
from forgec.parser import Parser
from forgec.semantic import TypeChecker
from forgec.ir_gen import IRGenerator
from forgec.diagnostics import DiagnosticEngine

class CompilerPipeline:
    def __init__(self, source_path: str, visualize: bool = False):
        self.source_path = Path(source_path)
        self.visualize = visualize
        self.source_code = ""
        self.artifacts: Dict[str, Any] = {}
        self.diagnostics = DiagnosticEngine()

    def run(self):
        """
        Execute the compilation pipeline.
        """
        self._read_source()
        
        # 1. Lexer
        lexer = Lexer(self.source_code, self.diagnostics)
        tokens = lexer.tokenize()
        self.artifacts["tokens"] = [
            {"type": t.type.name, "lexeme": t.lexeme, "span": str(t.span)} 
            for t in tokens
        ]
        if self.diagnostics.has_errors:
            raise Exception("Lexing failed")

        # 2. Parser
        parser = Parser(tokens, self.diagnostics)
        program = parser.parse()
        # TODO: Serialize AST for artifacts
        if self.diagnostics.has_errors:
            raise Exception("Parsing failed")

        # 3. Semantic Analysis
        checker = TypeChecker(self.diagnostics)
        checker.check(program)
        if self.diagnostics.has_errors:
            raise Exception("Semantic analysis failed")

        # 4. IR Generation
        ir_gen = IRGenerator(program, self.diagnostics)
        ir_code = ir_gen.generate(program)
        self.artifacts["ir"] = ir_code

        # Write IR to file
        ir_path = self.source_path.with_suffix(".ll")
        ir_path.write_text(ir_code)
        
        if self.visualize:
            self._export_visualization_data()

    def _read_source(self):
        if not self.source_path.exists():
            raise FileNotFoundError(f"Source file not found: {self.source_path}")
        self.source_code = self.source_path.read_text()
        self.artifacts["source"] = self.source_code

    def _export_visualization_data(self):
        """
        Export collected artifacts to a JSON file for the dashboard.
        """
        output_path = self.source_path.with_suffix(".json")
        # Structure for the dashboard timeline
        dashboard_data = {
            "filename": self.source_path.name,
            "source": self.source_code,
            "timeline": [
                {"stage": "Lexer", "data": self.artifacts.get("tokens")},
                {"stage": "IR Generation", "data": self.artifacts.get("ir")}
            ]
        }
        
        with open(output_path, "w") as f:
            json.dump(dashboard_data, f, indent=2)
        print(f"Visualization data written to {output_path}")
