from forgec.lexer import Lexer
from forgec.parser import Parser
from forgec.diagnostics import DiagnosticEngine

def test_parser_match_expression():
    source = """
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
    
    # Should parse without errors
    assert len(program.functions) == 1
    assert not diag.has_errors
