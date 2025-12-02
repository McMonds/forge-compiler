from forgec.lexer import Lexer
from forgec.parser import Parser
from forgec.diagnostics import DiagnosticEngine

def test_parser_enum_definition():
    source = "enum Option { Some(int), None }"
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    parser = Parser(lexer.tokenize(), diag)
    program = parser.parse()
    
    assert len(program.enums) == 1
    assert program.enums[0].name == "Option"
    assert len(program.enums[0].variants) == 2
    assert program.enums[0].variants[0].name == "Some"
    assert program.enums[0].variants[0].payload_type == "int"
    assert program.enums[0].variants[1].name == "None"
    assert program.enums[0].variants[1].payload_type is None

def test_parser_enum_instantiation():
    source = "fn main() { let x = Option::Some(42); let y = Option::None; }"
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    parser = Parser(lexer.tokenize(), diag)
    program = parser.parse()
    
    # Should parse without errors
    assert len(program.functions) == 1
