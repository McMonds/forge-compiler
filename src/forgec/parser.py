from typing import List, Optional
from forgec.lexer import Lexer, Token, TokenType
from forgec.diagnostics import DiagnosticEngine, Span
from forgec.ast_nodes import (
    Program, FunctionDef, Stmt, LetStmt, ExprStmt, 
    Expr, BinaryExpr, LiteralExpr, VariableExpr, IfExpr, CallExpr,
    StructDef, StructInstantiationExpr, FieldAccessExpr,
    EnumDef, EnumVariant, EnumInstantiationExpr,
    Pattern, EnumPattern, MatchArm, MatchExpr,
    TypeParameter, TypeRef
)

class Parser:
    def __init__(self, tokens: List[Token], diagnostics: DiagnosticEngine):
        self.tokens = tokens
        self.diagnostics = diagnostics
        self.current = 0

    def parse(self) -> Program:
        functions = []
        structs = []
        enums = []
        while not self._is_at_end():
            try:
                if self._check(TokenType.FN):
                    functions.append(self._function_def())
                elif self._check(TokenType.STRUCT):
                    structs.append(self._struct_def())
                elif self._check(TokenType.ENUM):
                    enums.append(self._enum_def())
                else:
                    # For now, we only allow top-level functions, structs, and enums
                    self._error("Expected function, struct, or enum definition")
                    self._synchronize()
            except ParseError:
                self._synchronize()
        
        # Create a span for the whole program (simplified)
        span = Span(0, 0, 0, 0) 
        if self.tokens:
            span = Span(self.tokens[0].span.start, self.tokens[-1].span.end, 0, 0)
            
        return Program(span, functions, structs, enums)

    # --- Declarations ---

    def _function_def(self) -> FunctionDef:
        start_token = self._consume(TokenType.FN, "Expected 'fn'")
        name = self._consume(TokenType.IDENTIFIER, "Expected function name").lexeme
        
        # Parse type parameters: fn name<T, U>(...)
        type_params = self._parse_type_params()
        
        self._consume(TokenType.LPAREN, "Expected '(' after function name")
        params = []
        if not self._check(TokenType.RPAREN):
            while True:
                param_name = self._consume(TokenType.IDENTIFIER, "Expected parameter name").lexeme
                self._consume(TokenType.COLON, "Expected ':' after parameter name")
                param_type = self._parse_type_ref()  # Use TypeRef instead of string
                params.append((param_name, param_type))
                if not self._match(TokenType.COMMA):
                    break
        self._consume(TokenType.RPAREN, "Expected ')' after parameters")

        # Parse return type
        return_type_ref = TypeRef(start_token.span, "void", [])  # Default void
        if self._match(TokenType.ARROW):
            return_type_ref = self._parse_type_ref()

        self._consume(TokenType.LBRACE, "Expected '{' before function body")
        body = self._block()
        
        span = Span(start_token.span.start, self.tokens[self.current-1].span.end, start_token.span.line, 0)
        return FunctionDef(span, name, type_params, params, return_type_ref, body)

    def _struct_def(self) -> StructDef:
        start_token = self._consume(TokenType.STRUCT, "Expected 'struct'")
        name = self._consume(TokenType.IDENTIFIER, "Expected struct name").lexeme
        
        # Parse type parameters: struct Name<T, U> { ... }
        type_params = self._parse_type_params()
        
        self._consume(TokenType.LBRACE, "Expected '{' after struct name")
        fields = []
        
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            field_name = self._consume(TokenType.IDENTIFIER, "Expected field name").lexeme
            self._consume(TokenType.COLON, "Expected ':' after field name")
            field_type = self._parse_type_ref()  # Use TypeRef instead of string
            fields.append((field_name, field_type))
            
            if not self._check(TokenType.RBRACE):
                self._consume(TokenType.COMMA, "Expected ',' between fields")
        
        self._consume(TokenType.RBRACE, "Expected '}' after struct fields")
        
        span = Span(start_token.span.start, self.tokens[self.current-1].span.end, start_token.span.line, 0)
        return StructDef(span, name, type_params, fields)

    def _enum_def(self) -> EnumDef:
        start_token = self._consume(TokenType.ENUM, "Expected 'enum'")
        name = self._consume(TokenType.IDENTIFIER, "Expected enum name").lexeme
        
        # Parse type parameters: enum Name<T> { ... }
        type_params = self._parse_type_params()
        
        self._consume(TokenType.LBRACE, "Expected '{' after enum name")
        variants = []
        
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            variant_start = self.tokens[self.current].span
            variant_name = self._consume(TokenType.IDENTIFIER, "Expected variant name").lexeme
            payload_type_ref = None
            
            # Check for payload
            if self._match(TokenType.LPAREN):
                payload_type_ref = self._parse_type_ref()  # Use TypeRef instead of string
                self._consume(TokenType.RPAREN, "Expected ')' after payload type")
            
            variant_span = Span(variant_start.start, self.tokens[self.current-1].span.end, variant_start.line, 0)
            variants.append(EnumVariant(variant_span, variant_name, payload_type_ref))
            
            if not self._check(TokenType.RBRACE):
                self._consume(TokenType.COMMA, "Expected ',' between variants")
        
        self._consume(TokenType.RBRACE, "Expected '}' after enum variants")
        
        span = Span(start_token.span.start, self.tokens[self.current-1].span.end, start_token.span.line, 0)
        return EnumDef(span, name, type_params, variants)


    def _block(self) -> List[Stmt]:
        statements = []
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            try:
                if self._match(TokenType.LET):
                    statements.append(self._let_declaration())
                else:
                    expr = self._expression()
                    
                    # Check for implicit return (last expression in block)
                    if self._check(TokenType.RBRACE):
                        statements.append(ExprStmt(expr.span, expr))
                        break # Done
                    
                    # Check for optional semicolon for block expressions (if, match)
                    if isinstance(expr, (IfExpr, MatchExpr)):
                        # Optional semicolon
                        self._match(TokenType.SEMICOLON)
                        statements.append(ExprStmt(expr.span, expr))
                    else:
                        # Mandatory semicolon for other expressions
                        self._consume(TokenType.SEMICOLON, "Expected ';' after expression")
                        statements.append(ExprStmt(expr.span, expr))
            except ParseError:
                self._synchronize()
                    
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
            
            # Check for type arguments first: Box<int>
            type_args = self._parse_type_args()
            
           # Check for enum instantiation (e.g., Option<int>::Some(42))
            if self._check(TokenType.COLONCOLON):
                # Create a temporary token with type args for _enum_instantiation
                return self._enum_instantiation(identifier_token, type_args)
            
            # Check if this is a struct instantiation (e.g., Box<int> { value: 10 })
            # We need to be careful not to confuse this with a match expression scrutinee followed by {
            # e.g. match x { ... } -> x { ... } looks like struct instantiation
            # We look ahead to see if it looks like { field: ... } or {}
            if self._check(TokenType.LBRACE):
                is_struct = False
                # Look ahead
                if self.tokens[self.current + 1].type == TokenType.RBRACE:
                    is_struct = True # Empty struct Box<int> {}
                elif (self.tokens[self.current + 1].type == TokenType.IDENTIFIER and 
                      self.tokens[self.current + 2].type == TokenType.COLON):
                    is_struct = True # Box<int> { x: ... }
                
                if is_struct:
                    return self._struct_instantiation(identifier_token, type_args)
            
            # If type_args were parsed but not used for instantiation, it's an error or a variable with type args (which is not allowed here)
            if type_args:
                raise self._error("Type arguments are only allowed for struct or enum instantiations here.")

            return VariableExpr(identifier_token.span, identifier_token.lexeme)
        
        if self._match(TokenType.IF):
            return self._if_expression()
        
        if self._match(TokenType.MATCH):
            return self._match_expression()
        
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

    def _struct_instantiation(self, struct_name_token: Token, type_args: List[TypeRef] = None) -> StructInstantiationExpr:
        # Box<int> { value: 42 } or Point { x: 10, y: 20 }
        start_span = struct_name_token.span
        struct_name = struct_name_token.lexeme
        
        # Parse type arguments if not already provided
        if type_args is None:
            type_args = self._parse_type_args()
        
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
        return StructInstantiationExpr(span, struct_name, type_args, field_values)

    def _enum_instantiation(self, enum_name_token: Token, type_args: List[TypeRef] = None) -> EnumInstantiationExpr:
        # Option<int>::Some(42) or Option::None
        start_span = enum_name_token.span
        enum_name = enum_name_token.lexeme
        
        # Parse type arguments if not already provided
        if type_args is None:
            type_args = self._parse_type_args()
        
        self._consume(TokenType.COLONCOLON, "Expected '::' for enum variant")
        variant_name = self._consume(TokenType.IDENTIFIER, "Expected variant name").lexeme
        
        payload = None
        # Check for payload
        if self._match(TokenType.LPAREN):
            payload = self._expression()
            self._consume(TokenType.RPAREN, "Expected ')' after payload")
        
        span = Span(start_span.start, self.tokens[self.current-1].span.end, start_span.line, 0)
        return EnumInstantiationExpr(span, enum_name, type_args, variant_name, payload)

    def _match_expression(self) -> MatchExpr:
        # match value { Option::Some(x) => { ... }, Option::None => { ... } }
        start_token = self.tokens[self.current - 1]
        
        scrutinee = self._expression()
        
        self._consume(TokenType.LBRACE, "Expected '{' after match scrutinee")
        arms = []
        
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            # Parse pattern
            pattern = self._parse_pattern()
            
            # Expect =>
            self._consume(TokenType.FATARROW, "Expected '=>' after pattern")
            
            # Parse arm body (block)
            self._consume(TokenType.LBRACE, "Expected '{' for match arm body")
            body = self._block()
            
            arms.append(MatchArm(pattern.span, pattern, body))
            
            # Optional comma between arms
            self._match(TokenType.COMMA)
        
        self._consume(TokenType.RBRACE, "Expected '}' after match arms")
        
        span = Span(start_token.span.start, self.tokens[self.current-1].span.end, start_token.span.line, 0)
        return MatchExpr(span, scrutinee, arms)

    def _parse_pattern(self) -> Pattern:
        # For now, only enum patterns: Option::Some(x) or Option::None
        start_token = self._consume(TokenType.IDENTIFIER, "Expected pattern")
        enum_name = start_token.lexeme
        
        self._consume(TokenType.COLONCOLON, "Expected '::' in enum pattern")
        variant_name = self._consume(TokenType.IDENTIFIER, "Expected variant name").lexeme
        
        binding = None
        # Check for payload binding
        if self._match(TokenType.LPAREN):
            binding = self._consume(TokenType.IDENTIFIER, "Expected binding variable").lexeme
            self._consume(TokenType.RPAREN, "Expected ')' after binding")
        
        span = Span(start_token.span.start, self.tokens[self.current-1].span.end, start_token.span.line, 0)
        return EnumPattern(span, enum_name, variant_name, binding)


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
    
    # --- Generic Type System Parsing ---
    
    def _parse_type_params(self) -> List[TypeParameter]:
        """Parse type parameters: <T, U, V>. Returns empty list if no type parameters."""
        if not self._match(TokenType.LT):
            return []
        
        params = []
        while True:
            param_token = self._consume(TokenType.IDENTIFIER, "Expected type parameter name")
            params.append(TypeParameter(param_token.span, param_token.lexeme))
            
            if not self._match(TokenType.COMMA):
                break
        
        self._consume(TokenType.GT, "Expected '>' after type parameters")
        return params
    
    def _parse_type_ref(self) -> TypeRef:
        """Parse type reference: int, Box<int>, Option<Box<bool>>. Handles nested generics."""
        start_token = self._consume(TokenType.IDENTIFIER, "Expected type name")
        name = start_token.lexeme
        
        type_args = []
        # Check if generic type (use lookahead to avoid confusion with < operator)
        if self._check(TokenType.LT) and self._is_type_argument_context():
            self._advance()  # Consume <
            
            while True:
                type_args.append(self._parse_type_ref())  # Recursive for nested generics
                
                if not self._match(TokenType.COMMA):
                    break
            
            self._consume(TokenType.GT, "Expected '>' after type arguments")
        
        span = Span(start_token.span.start, self.tokens[self.current-1].span.end, start_token.span.line, 0)
        return TypeRef(span, name, type_args)
    
    def _is_type_argument_context(self) -> bool:
        """Disambiguate < as generic vs comparison.  Returns True for type argument context."""
        if not self._check(TokenType.LT):
            return False
        
        # Lookahead: IDENTIFIER < IDENTIFIER -> likely generic
        next_idx = self.current + 1
        if next_idx < len(self.tokens):
            next_token = self.tokens[next_idx]
            if next_token.type == TokenType.IDENTIFIER:
                return True
        
        return False
    
    def _parse_type_args(self) -> List[TypeRef]:
        """Parse type arguments for instantiation: <int, bool>"""
        if not self._match(TokenType.LT):
            return []
        
        args = []
        while True:
            args.append(self._parse_type_ref())
            if not self._match(TokenType.COMMA):
                break
        
        self._consume(TokenType.GT, "Expected '>' after type arguments")
        return args

class ParseError(Exception):
    pass
