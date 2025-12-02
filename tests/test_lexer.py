
from forgec.lexer import Lexer, TokenType
from forgec.diagnostics import DiagnosticEngine

def test_lexer_basic():
    source = "let x = 10;"
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    tokens = lexer.tokenize()

    assert len(tokens) == 6
    assert tokens[0].type == TokenType.LET
    assert tokens[1].type == TokenType.IDENTIFIER
    assert tokens[1].lexeme == "x"
    assert tokens[2].type == TokenType.EQ
    assert tokens[3].type == TokenType.INTEGER
    assert tokens[3].value == 10
    assert tokens[4].type == TokenType.SEMICOLON
    assert tokens[5].type == TokenType.EOF

def test_lexer_keywords():
    source = "if else fn return true false"
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    tokens = lexer.tokenize()
    
    expected = [
        TokenType.IF, TokenType.ELSE, TokenType.FN, 
        TokenType.RETURN, TokenType.TRUE, TokenType.FALSE, 
        TokenType.EOF
    ]
    
    assert len(tokens) == len(expected)
    for i, token in enumerate(tokens):
        assert token.type == expected[i]

def test_lexer_unknown_char():
    source = "let x $ 10;"
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    tokens = lexer.tokenize()

    assert diag.has_errors
    assert any(t.type == TokenType.ERROR for t in tokens)
