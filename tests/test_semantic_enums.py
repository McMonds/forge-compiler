from forgec.lexer import Lexer
from forgec.parser import Parser
from forgec.semantic import TypeChecker
from forgec.diagnostics import DiagnosticEngine

def test_semantic_enum_definition():
    source = """
    enum Option { Some(int), None }
    fn main() { let x = Option::Some(42); }
    """
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    parser = Parser(lexer.tokenize(), diag)
    program = parser.parse()
    
    checker = TypeChecker(diag)
    checker.check(program)
    
    # Should have no errors
    assert not diag.has_errors

def test_semantic_enum_undefined():
    source = """
    fn main() { let x = Foo::Bar(10); }
    """
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    parser = Parser(lexer.tokenize(), diag)
    program = parser.parse()
    
    checker = TypeChecker(diag)
    checker.check(program)
    
    # Should have error about undefined enum
    assert diag.has_errors

def test_semantic_enum_invalid_variant():
    source = """
    enum Option { Some(int), None }
    fn main() { let x = Option::Invalid(10); }
    """
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    parser = Parser(lexer.tokenize(), diag)
    program = parser.parse()
    
    checker = TypeChecker(diag)
    checker.check(program)
    
    # Should have error about invalid variant
    assert diag.has_errors

def test_semantic_enum_payload_mismatch():
    source = """
    enum Option { Some(int), None }
    fn main() { let x = Option::Some(true); }
    """
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    parser = Parser(lexer.tokenize(), diag)
    program = parser.parse()
    
    checker = TypeChecker(diag)
    checker.check(program)
    
    # Should have error about type mismatch
    assert diag.has_errors

def test_semantic_enum_missing_payload():
    source = """
    enum Option { Some(int), None }
    fn main() { let x = Option::Some; }
    """
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    parser = Parser(lexer.tokenize(), diag)
    program = parser.parse()
    
    checker = TypeChecker(diag)
    checker.check(program)
    
    # Should have error about missing payload
    assert diag.has_errors
