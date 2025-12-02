from forgec.lexer import Lexer, TokenType
from forgec.parser import Parser
from forgec.diagnostics import DiagnosticEngine
from forgec.ast_nodes import FunctionDef, LetStmt, BinaryExpr, LiteralExpr, VariableExpr

def test_parser_function():
    source = "fn main() { let x = 10; }"
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    tokens = lexer.tokenize()
    parser = Parser(tokens, diag)
    program = parser.parse()

    assert len(program.functions) == 1
    func = program.functions[0]
    assert isinstance(func, FunctionDef)
    assert func.name == "main"
    assert len(func.body) == 1
    assert isinstance(func.body[0], LetStmt)

def test_parser_expression_precedence():
    source = "fn main() { let x = 1 + 2 * 3; }"
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    tokens = lexer.tokenize()
    parser = Parser(tokens, diag)
    program = parser.parse()

    let_stmt = program.functions[0].body[0]
    expr = let_stmt.initializer
    
    # 1 + (2 * 3)
    assert isinstance(expr, BinaryExpr)
    assert expr.operator == "+"
    assert isinstance(expr.left, LiteralExpr)
    assert expr.left.value == 1
    
    assert isinstance(expr.right, BinaryExpr)
    assert expr.right.operator == "*"
    assert expr.right.left.value == 2
    assert expr.right.right.value == 3

def test_parser_error_recovery():
    # Missing semicolon
    source = "fn main() { let x = 10 let y = 20; }"
    diag = DiagnosticEngine()
    lexer = Lexer(source, diag)
    tokens = lexer.tokenize()
    parser = Parser(tokens, diag)
    program = parser.parse()

    assert diag.has_errors
    # Should still have parsed the function and maybe the first statement
    assert len(program.functions) == 1
