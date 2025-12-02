from forgec.lexer import Lexer
from forgec.parser import Parser
from forgec.semantic import TypeChecker
from forgec.diagnostics import DiagnosticEngine

def test_semantic_struct_definition():
    source = """
    struct Point { x: int, y: int }
    fn main() { let p = Point { x: 10, y: 20 }; }
    """
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    parser = Parser(lexer.tokenize(), diag)
    program = parser.parse()
    
    checker = TypeChecker(diag)
    checker.check(program)
    
    # Should have no errors
    assert not diag.has_errors

def test_semantic_struct_missing_field():
    source = """
    struct Point { x: int, y: int }
    fn main() { let p = Point { x: 10 }; }
    """
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    parser = Parser(lexer.tokenize(), diag)
    program = parser.parse()
    
    checker = TypeChecker(diag)
    checker.check(program)
    
    # Should have error about missing field
    assert diag.has_errors

def test_semantic_struct_field_type_mismatch():
    source = """
    struct Point { x: int, y: int }
    fn main() { let p = Point { x: true, y: 20 }; }
    """
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    parser = Parser(lexer.tokenize(), diag)
    program = parser.parse()
    
    checker = TypeChecker(diag)
    checker.check(program)
    
    # Should have error about type mismatch
    assert diag.has_errors

def test_semantic_field_access():
    source = """
    struct Point { x: int, y: int }
    fn main() { 
        let p = Point { x: 10, y: 20 };
        let x_val = p.x;
    }
    """
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    parser = Parser(lexer.tokenize(), diag)
    program = parser.parse()
    
    checker = TypeChecker(diag)
    checker.check(program)
    
    # Should have no errors
    assert not diag.has_errors

def test_semantic_undefined_field():
    source = """
    struct Point { x: int, y: int }
    fn main() { 
        let p = Point { x: 10, y: 20 };
        let z_val = p.z;
    }
    """
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    parser = Parser(lexer.tokenize(), diag)
    program = parser.parse()
    
    checker = TypeChecker(diag)
    checker.check(program)
    
    # Should have error about undefined field
    assert diag.has_errors
