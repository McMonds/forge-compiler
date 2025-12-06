from forgec.lexer import Lexer
from forgec.parser import Parser
from forgec.semantic import TypeChecker
from forgec.diagnostics import DiagnosticEngine

def test_semantic_basic_types():
    source = "fn main() { let x: int = 10; let y: bool = true; }"
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    parser = Parser(lexer.tokenize(), diag)
    program = parser.parse()
    
    checker = TypeChecker(diag)
    checker.check(program)
    
    assert not diag.has_errors

def test_semantic_type_mismatch():
    source = "fn main() { let x: int = true; }"
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    parser = Parser(lexer.tokenize(), diag)
    program = parser.parse()
    
    checker = TypeChecker(diag)
    checker.check(program)
    
    assert diag.has_errors
    assert "Type mismatch" in diag.diagnostics[0].message

def test_semantic_undefined_variable():
    source = "fn main() { let x = y; }"
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    parser = Parser(lexer.tokenize(), diag)
    program = parser.parse()
    
    checker = TypeChecker(diag)
    checker.check(program)
    
    assert diag.has_errors
    assert "Undefined variable" in diag.diagnostics[0].message

def test_semantic_scope_shadowing():
    # Forge allows shadowing in separate scopes
    source = "fn main() { let x = 10; { let x = 20; } }"
    # Note: My parser doesn't support blocks as statements yet, only function bodies.
    # Let's test variable redefinition in same scope
    source = "fn main() { let x = 10; let x = 20; }"
    
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    parser = Parser(lexer.tokenize(), diag)
    program = parser.parse()
    
    checker = TypeChecker(diag)
    checker.check(program)
    
    assert diag.has_errors
    assert "already defined" in diag.diagnostics[0].message
