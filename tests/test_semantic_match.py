from forgec.lexer import Lexer
from forgec.parser import Parser
from forgec.semantic import TypeChecker
from forgec.diagnostics import DiagnosticEngine

def test_semantic_match_basic():
    source = """
    enum Option { Some(int), None }
    fn main() {
        let x = Option::Some(42);
        let result = match x {
            Option::Some(val) => { val },
            Option::None => { 0 }
        };
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

def test_semantic_match_non_exhaustive():
    source = """
    enum Option { Some(int), None }
    fn main() {
        let x = Option::Some(42);
        let result = match x {
            Option::Some(val) => { val }
        };
    }
    """
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    parser = Parser(lexer.tokenize(), diag)
    program = parser.parse()
    
    checker = TypeChecker(diag)
    checker.check(program)
    
    # Should have error about non-exhaustive match
    assert diag.has_errors
