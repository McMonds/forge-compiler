from forgec.lexer import Lexer
from forgec.parser import Parser
from forgec.diagnostics import DiagnosticEngine

def test_parser_struct_definition():
    source = "struct Point { x: int, y: int }"
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    parser = Parser(lexer.tokenize(), diag)
    program = parser.parse()
    
    assert len(program.structs) == 1
    assert program.structs[0].name == "Point"
    assert len(program.structs[0].fields) == 2
    assert program.structs[0].fields[0] == ("x", "int")
    assert program.structs[0].fields[1] == ("y", "int")

def test_parser_struct_instantiation():
    source = "fn main() { let p = Point { x: 10, y: 20 }; }"
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    parser = Parser(lexer.tokenize(), diag)
    program = parser.parse()
    
    # Should parse without errors
    assert len(program.functions) == 1

def test_parser_field_access():
    source = "fn main() { let x = p.x; }"
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    parser = Parser(lexer.tokenize(), diag)
    program = parser.parse()
    
    # Should parse without errors
    assert len(program.functions) == 1
