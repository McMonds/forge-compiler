from typing import List, Optional
from forgec.lexer import Token, TokenType
from forgec.diagnostics import DiagnosticEngine, Span
from forgec.ast_nodes import (
    Program, FunctionDef, Stmt, LetStmt, ExprStmt, 
    Expr, BinaryExpr, LiteralExpr, VariableExpr, IfExpr, CallExpr,
    StructDef, StructInstantiationExpr, FieldAccessExpr
)

class Parser:
    def __init__(self, tokens: List[Token], diagnostics: DiagnosticEngine):
        self.tokens = tokens
        self.diagnostics = diagnostics
        self.current = 0

    def parse(self) -> Program:
        functions = []
        structs = []
        while not self._is_at_end():
            try:
                if self._check(TokenType.FN):
                    functions.append(self._function_def())
                elif self._check(TokenType.STRUCT):
                    structs.append(self._struct_def())
                else:
                    # For now, we only allow top-level functions and structs
                    self._error("Expected function or struct definition")
                    self._synchronize()
            except ParseError:
                self._synchronize()
        
        # Create a span for the whole program (simplified)
        span = Span(0, 0, 0, 0) 
        if self.tokens:
            span = Span(self.tokens[0].span.start, self.tokens[-1].span.end, 0, 0)
            
        return Program(span, functions, structs)

    # --- Declarations ---

    def _function_def(self) -> FunctionDef:
        start_token = self._consume(TokenType.FN, "Expected 'fn'")
        name = self._consume(TokenType.IDENTIFIER, "Expected function name").lexeme
        
        self._consume(TokenType.LPAREN, "Expected '(' after function name")
        params = []
        if not self._check(TokenType.RPAREN):
            while True:
                param_name = self._consume(TokenType.IDENTIFIER, "Expected parameter name").lexeme
                self._consume(TokenType.COLON, "Expected ':' after parameter name")
                param_type = self._consume(TokenType.IDENTIFIER, "Expected parameter type").lexeme
                params.append((param_name, param_type))
                if not self._match(TokenType.COMMA):
                    break
        self._consume(TokenType.RPAREN, "Expected ')' after parameters")

        return_type = "void"
        if self._match(TokenType.ARROW):
            return_type = self._consume(TokenType.IDENTIFIER, "Expected return type").lexeme

        self._consume(TokenType.LBRACE, "Expected '{' before function body")
        body = self._block()
        
        span = Span(start_token.span.start, self.tokens[self.current-1].span.end, start_token.span.line, 0)
        return FunctionDef(span, name, params, return_type, body)

    def _struct_def(self) -> StructDef:
        start_token = self._consume(TokenType.STRUCT, "Expected 'struct'")
        name = self._consume(TokenType.IDENTIFIER, "Expected struct name").lexeme
        
        self._consume(TokenType.LBRACE, "Expected '{' after struct name")
        fields = []
        
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            field_name = self._consume(TokenType.IDENTIFIER, "Expected field name").lexeme
            self._consume(TokenType.COLON, "Expected ':' after field name")
            field_type = self._consume(TokenType.IDENTIFIER, "Expected field type").lexeme
            fields.append((field_name, field_type))
            
            if not self._check(TokenType.RBRACE):
                self._consume(TokenType.COMMA, "Expected ',' between fields")
        
        self._consume(TokenType.RBRACE, "Expected '}' after struct fields")
        
        span = Span(start_token.span.start, self.tokens[self.current-1].span.end, start_token.span.line, 0)
        return StructDef(span, name, fields)


    def _block(self) -> List[Stmt]:
        statements = []
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            statements.append(self._declaration())
        self._consume(TokenType.RBRACE, "Expected '}' after block")
        return statements

    def _declaration(self) -> Stmt:
        try:
            if self._match(TokenType.LET):
                return self._let_declaration()
            return self._statement()
        except ParseError:
            self._synchronize()
            return None # Should not happen if synchronize works well

    def _let_declaration(self) -> Stmt:
        start_token = self.tokens[self.current - 1]
        name = self._consume(TokenType.IDENTIFIER, "Expected variable name").lexeme
        
        type_annotation = None
        if self._match(TokenType.COLON):
            type_annotation = self._consume(TokenType.IDENTIFIER, "Expected type").lexeme
        
        self._consume(TokenType.EQ, "Expected '=' after variable name")
        initializer = self._expression()
        self._consume(TokenType.SEMICOLON, "Expected ';' after variable declaration")
        
        span = Span(start_token.span.start, self.tokens[self.current-1].span.end, start_token.span.line, 0)
        return LetStmt(span, name, initializer, type_annotation)

    def _statement(self) -> Stmt:
        expr = self._expression()
        self._consume(TokenType.SEMICOLON, "Expected ';' after expression")
        return ExprStmt(expr.span, expr)

    # --- Expressions ---

    def _expression(self) -> Expr:
        return self._equality()

    def _equality(self) -> Expr:
        expr = self._comparison()
        while self._match(TokenType.EQEQ, TokenType.NEQ):
            operator = self.tokens[self.current - 1].lexeme
            right = self._comparison()
            expr = BinaryExpr(Span(expr.span.start, right.span.end, expr.span.line, 0), expr, operator, right)
        return expr

    def _comparison(self) -> Expr:
        expr = self._term()
        while self._match(TokenType.GT, TokenType.LT): # Add >= <=
            operator = self.tokens[self.current - 1].lexeme
            right = self._term()
            expr = BinaryExpr(Span(expr.span.start, right.span.end, expr.span.line, 0), expr, operator, right)
        return expr

    def _term(self) -> Expr:
        expr = self._factor()
        while self._match(TokenType.MINUS, TokenType.PLUS):
            operator = self.tokens[self.current - 1].lexeme
            right = self._factor()
            expr = BinaryExpr(Span(expr.span.start, right.span.end, expr.span.line, 0), expr, operator, right)
        return expr

    def _factor(self) -> Expr:
        expr = self._postfix()
        while self._match(TokenType.SLASH, TokenType.STAR):
            operator = self.tokens[self.current - 1].lexeme
            right = self._postfix()
            expr = BinaryExpr(Span(expr.span.start, right.span.end, expr.span.line, 0), expr, operator, right)
        return expr

    def _postfix(self) -> Expr:
        expr = self._primary()
        
        # Handle field access (e.g., p.x)
        while self._match(TokenType.DOT):
            field_name = self._consume(TokenType.IDENTIFIER, "Expected field name after '.'").lexeme
            span = Span(expr.span.start, self.tokens[self.current-1].span.end, expr.span.line, 0)
            expr = FieldAccessExpr(span, expr, field_name)
        
        return expr


    def _primary(self) -> Expr:
        if self._match(TokenType.FALSE):
            return LiteralExpr(self.tokens[self.current-1].span, False, "bool")
        if self._match(TokenType.TRUE):
            return LiteralExpr(self.tokens[self.current-1].span, True, "bool")
        if self._match(TokenType.INTEGER):
            return LiteralExpr(self.tokens[self.current-1].span, self.tokens[self.current-1].value, "int")
        
        if self._match(TokenType.IDENTIFIER):
            identifier_token = self.tokens[self.current-1]
            # Check if this is a struct instantiation (e.g., Point { x: 10, y: 20 })
            if self._check(TokenType.LBRACE):
                return self._struct_instantiation(identifier_token)
            return VariableExpr(identifier_token.span, identifier_token.lexeme)
        
        if self._match(TokenType.IF):
            return self._if_expression()
        
        if self._match(TokenType.LPAREN):
            expr = self._expression()
            self._consume(TokenType.RPAREN, "Expected ')' after expression")
            return expr

        raise self._error("Expect expression")

    def _if_expression(self) -> Expr:
        start_token = self.tokens[self.current - 1]
        condition = self._expression()
        
        self._consume(TokenType.LBRACE, "Expected '{' after if condition")
        then_branch = self._block()
        
        else_branch = None
        if self._match(TokenType.ELSE):
            self._consume(TokenType.LBRACE, "Expected '{' after else")
            else_branch = self._block()
            
        span = Span(start_token.span.start, self.tokens[self.current-1].span.end, start_token.span.line, 0)
        return IfExpr(span, condition, then_branch, else_branch)

    def _struct_instantiation(self, struct_name_token: Token) -> StructInstantiationExpr:
        # Point { x: 10, y: 20 }
        start_span = struct_name_token.span
        struct_name = struct_name_token.lexeme
        
        self._consume(TokenType.LBRACE, "Expected '{' for struct instantiation")
        field_values = []
        
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            field_name = self._consume(TokenType.IDENTIFIER, "Expected field name").lexeme
            self._consume(TokenType.COLON, "Expected ':' after field name")
            value_expr = self._expression()
            field_values.append((field_name, value_expr))
            
            if not self._check(TokenType.RBRACE):
                self._consume(TokenType.COMMA, "Expected ',' between fields")
        
        self._consume(TokenType.RBRACE, "Expected '}' after struct fields")
        
        span = Span(start_span.start, self.tokens[self.current-1].span.end, start_span.line, 0)
        return StructInstantiationExpr(span, struct_name, field_values)


    # --- Helpers ---

    def _match(self, *types: TokenType) -> bool:
        for type in types:
            if self._check(type):
                self._advance()
                return True
        return False

    def _check(self, type: TokenType) -> bool:
        if self._is_at_end():
            return False
        return self.tokens[self.current].type == type

    def _advance(self) -> Token:
        if not self._is_at_end():
            self.current += 1
        return self.tokens[self.current - 1]

    def _is_at_end(self) -> bool:
        return self.tokens[self.current].type == TokenType.EOF

    def _consume(self, type: TokenType, message: str) -> Token:
        if self._check(type):
            return self._advance()
        raise self._error(message)

    def _error(self, message: str) -> Exception:
        token = self.tokens[self.current]
        self.diagnostics.error(message, token.span)
        return ParseError()

    def _synchronize(self):
        self._advance()
        while not self._is_at_end():
            if self.tokens[self.current - 1].type == TokenType.SEMICOLON:
                return
            
            if self.tokens[self.current].type in [
                TokenType.FN, TokenType.LET, TokenType.IF, 
                TokenType.RETURN, TokenType.STRUCT, TokenType.ENUM
            ]:
                return
            
            self._advance()

class ParseError(Exception):
    pass
