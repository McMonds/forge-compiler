from typing import List, Optional
from forgec.lexer import Lexer, Token, TokenType
from forgec.diagnostics import DiagnosticEngine, Span
from forgec.ast_nodes import (
    Program, FunctionDef, Stmt, LetStmt, ExprStmt, ReturnStmt, 
    Expr, BinaryExpr, LiteralExpr, VariableExpr, IfExpr, CallExpr, MethodCallExpr,
    StructDef, StructInstantiationExpr, FieldAccessExpr,
    EnumDef, EnumVariant, EnumInstantiationExpr,
    Pattern, EnumPattern, MatchArm, MatchExpr,
    TypeParameter, TypeRef,
    TypeParameter, TypeRef,
    TraitDef, ImplBlock, TraitMethod,
    ModDecl, UseDecl,  # NEW
    ExternBlock, ExternFunc # NEW
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
        traits = []
        impls = []
        modules = []
        imports = []
        extern_blocks = []
        while not self._is_at_end():
            try:
                is_public = False
                if self._match(TokenType.PUB):
                    is_public = True

                if self._check(TokenType.FN):
                    functions.append(self._function_def(is_public))
                elif self._check(TokenType.STRUCT):
                    structs.append(self._struct_def(is_public))
                elif self._check(TokenType.ENUM):
                    enums.append(self._enum_def(is_public))
                elif self._check(TokenType.TRAIT):
                    traits.append(self._trait_def(is_public))
                elif self._check(TokenType.IMPL):
                    if is_public:
                        self._error("'pub' is not allowed on impl blocks")
                    impls.append(self._impl_block())
                elif self._check(TokenType.MOD):
                    modules.append(self._mod_decl(is_public))
                elif self._check(TokenType.USE):
                    imports.append(self._use_decl(is_public))
                elif self._check(TokenType.EXTERN):
                    if is_public:
                        self._error("'pub' is not allowed on extern blocks")
                    extern_blocks.append(self._extern_block())
                else:
                    # For now, we only allow top-level functions, structs, and enums
                    self._error("Expected function, struct, enum, trait, or impl definition")
                    self._synchronize()
            except ParseError:
                self._synchronize()
        
        # Create a span for the whole program (simplified)
        span = Span(0, 0, 0, 0) 
        if self.tokens:
            span = Span(self.tokens[0].span.start, self.tokens[-1].span.end, 0, 0)
            
        return Program(span, structs, enums, traits, impls, functions, modules, imports, extern_blocks)

    # --- Declarations ---

    def _extern_block(self) -> ExternBlock:
        start_token = self._consume(TokenType.EXTERN, "Expected 'extern'")
        abi = self._consume(TokenType.STRING, "Expected ABI string (e.g., \"C\")").value
        
        self._consume(TokenType.LBRACE, "Expected '{' after extern ABI")
        functions = []
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            functions.append(self._extern_func())
        self._consume(TokenType.RBRACE, "Expected '}' after extern block")
        
        span = Span(start_token.span.start, self.tokens[self.current-1].span.end, start_token.span.line, 0)
        return ExternBlock(span, abi, functions)

    def _extern_func(self) -> ExternFunc:
        start_token = self._consume(TokenType.FN, "Expected 'fn'")
        name = self._consume(TokenType.IDENTIFIER, "Expected function name").lexeme
        
        self._consume(TokenType.LPAREN, "Expected '('")
        params = []
        if not self._check(TokenType.RPAREN):
            while True:
                p_name = self._consume(TokenType.IDENTIFIER, "Expected parameter name").lexeme
                self._consume(TokenType.COLON, "Expected ':'")
                p_type = self._parse_type_ref()
                params.append((p_name, p_type))
                if not self._match(TokenType.COMMA):
                    break
        self._consume(TokenType.RPAREN, "Expected ')'")
        
        self._consume(TokenType.ARROW, "Expected '->'")
        return_type = self._parse_type_ref()
        self._consume(TokenType.SEMICOLON, "Expected ';' after extern function declaration")
        
        span = Span(start_token.span.start, self.tokens[self.current-1].span.end, start_token.span.line, 0)
        return ExternFunc(span, name, params, return_type)

    def _mod_decl(self, is_public: bool = False) -> ModDecl:
        start_token = self._consume(TokenType.MOD, "Expected 'mod'")
        name = self._consume(TokenType.IDENTIFIER, "Expected module name").lexeme
        self._consume(TokenType.SEMICOLON, "Expected ';' after module declaration")
        
        span = Span(start_token.span.start, self.tokens[self.current-1].span.end, start_token.span.line, 0)
        return ModDecl(span, name, is_public)

    def _use_decl(self, is_public: bool = False) -> UseDecl:
        start_token = self._consume(TokenType.USE, "Expected 'use'")
        
        # Parse path: std::io::print
        path_parts = []
        path_parts.append(self._consume(TokenType.IDENTIFIER, "Expected import path").lexeme)
        
        while self._match(TokenType.COLONCOLON):
            path_parts.append(self._consume(TokenType.IDENTIFIER, "Expected identifier after '::'").lexeme)
            
        path = "::".join(path_parts)
        self._consume(TokenType.SEMICOLON, "Expected ';' after use declaration")
        
        span = Span(start_token.span.start, self.tokens[self.current-1].span.end, start_token.span.line, 0)
        return UseDecl(span, path, is_public)

    def _function_def(self, is_public: bool = False) -> FunctionDef:
        start_token = self._consume(TokenType.FN, "Expected 'fn'")
        name = self._consume(TokenType.IDENTIFIER, "Expected function name").lexeme
        
        # Parse type parameters: fn foo<T>(...)
        type_params = self._parse_type_params()
        
        self._consume(TokenType.LPAREN, "Expected '(' after function name")
        params = []
        
        # Check for 'self' as first parameter
        if self._match(TokenType.SELF):
            # Treat 'self' as a parameter named "self" with type "Self"
            # In a real compiler, we'd handle this more specifically
            params.append(("self", TypeRef(start_token.span, "Self", [])))
            if self._check(TokenType.COMMA):
                self._advance()
        
        if not self._check(TokenType.RPAREN):
            while True:
                param_name = self._consume(TokenType.IDENTIFIER, "Expected parameter name").lexeme
                self._consume(TokenType.COLON, "Expected ':' after parameter name")
                param_type = self._parse_type_ref()
                params.append((param_name, param_type))
                
                if not self._match(TokenType.COMMA):
                    break
        
        self._consume(TokenType.RPAREN, "Expected ')' after parameters")
        
        return_type = TypeRef(start_token.span, "void", [])
        if self._match(TokenType.ARROW):
            return_type = self._parse_type_ref()
            
        self._consume(TokenType.LBRACE, "Expected '{' before function body")
        body = self._block()
        
        span = Span(start_token.span.start, self.tokens[self.current-1].span.end, start_token.span.line, 0)
        return FunctionDef(span, name, type_params, params, return_type, body, is_public)

    def _struct_def(self, is_public: bool = False) -> StructDef:
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
        return StructDef(span, name, type_params, fields, is_public)

    def _enum_def(self, is_public: bool = False) -> EnumDef:
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
            
            if not self._match(TokenType.COMMA):
                break
        
        self._consume(TokenType.RBRACE, "Expected '}' after enum variants")
        
        span = Span(start_token.span.start, self.tokens[self.current-1].span.end, start_token.span.line, 0)
        return EnumDef(span, name, type_params, variants, is_public)

    def _trait_def(self, is_public: bool = False) -> TraitDef:
        """Parse: trait Name { fn method(self) -> type; }"""
        start_token = self._consume(TokenType.TRAIT, "Expected 'trait'")
        name = self._consume(TokenType.IDENTIFIER, "Expected trait name").lexeme
        
        self._consume(TokenType.LBRACE, "Expected '{'")
        methods = []
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            methods.append(self._parse_trait_method())
        self._consume(TokenType.RBRACE, "Expected '}'")
        
        span = Span(start_token.span.start, self.tokens[self.current-1].span.end, start_token.span.line, 0)
        return TraitDef(span, name, methods, is_public)

    def _parse_trait_method(self) -> TraitMethod:
        """Parse method signature: fn name(self) -> type;"""
        start_token = self._consume(TokenType.FN, "Expected 'fn'")
        name = self._consume(TokenType.IDENTIFIER, "Expected method name").lexeme
        
        self._consume(TokenType.LPAREN, "Expected '('")
        params = []
        # First parameter must be 'self'
        if self._match(TokenType.SELF):
            params.append(("self", TypeRef(start_token.span, "Self", [])))
            if self._check(TokenType.COMMA):
                self._advance()
        
        # Additional parameters
        while not self._check(TokenType.RPAREN) and not self._is_at_end():
            param_name = self._consume(TokenType.IDENTIFIER, "Expected parameter").lexeme
            self._consume(TokenType.COLON, "Expected ':'")
            param_type = self._parse_type_ref()
            params.append((param_name, param_type))
            
            if not self._match(TokenType.COMMA):
                break
        
        self._consume(TokenType.RPAREN, "Expected ')'")
        
        return_type = TypeRef(start_token.span, "void", [])
        if self._match(TokenType.ARROW):
            return_type = self._parse_type_ref()
        
        self._consume(TokenType.SEMICOLON, "Expected ';' after method signature")
        
        span = Span(start_token.span.start, self.tokens[self.current-1].span.end, start_token.span.line, 0)
        return TraitMethod(span, name, params, return_type)

    def _impl_block(self) -> ImplBlock:
        """Parse: impl TraitName for TypeName { ... }"""
        start_token = self._consume(TokenType.IMPL, "Expected 'impl'")
        trait_name = self._consume(TokenType.IDENTIFIER, "Expected trait name").lexeme
        
        self._consume(TokenType.FOR, "Expected 'for'")
        type_name = self._consume(TokenType.IDENTIFIER, "Expected type name").lexeme
        
        # Parse type arguments if generic: impl Display for Box<int>
        type_args = self._parse_type_args()
        
        self._consume(TokenType.LBRACE, "Expected '{'")
        methods = []
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            methods.append(self._function_def())  # Reuse function parsing
        self._consume(TokenType.RBRACE, "Expected '}'")
        
        span = Span(start_token.span.start, self.tokens[self.current-1].span.end, start_token.span.line, 0)
        return ImplBlock(span, trait_name, type_name, type_args, methods)


    def _block(self) -> List[Stmt]:
        statements = []
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            try:
                if self._match(TokenType.LET):
                    statements.append(self._let_declaration())
                elif self._match(TokenType.RETURN):
                    statements.append(self._return_stmt())
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
        if self._match(TokenType.RETURN):
            return self._return_stmt()
        expr = self._expression()
        self._consume(TokenType.SEMICOLON, "Expected ';' after expression")
        return ExprStmt(expr.span, expr)

    def _return_stmt(self) -> ReturnStmt:
        start_token = self.tokens[self.current - 1]
        value = None
        if not self._check(TokenType.SEMICOLON):
            value = self._expression()
        
        self._consume(TokenType.SEMICOLON, "Expected ';' after return value")
        
        end_span = self.tokens[self.current-1].span
        span = Span(start_token.span.start, end_span.end, start_token.span.line, 0)
        return ReturnStmt(span, value)

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
        
        # Handle field access (e.g., p.x) and method calls (e.g., p.show())
        while self._match(TokenType.DOT):
            field_name = self._consume(TokenType.IDENTIFIER, "Expected field/method name after '.'").lexeme
            
            if self._check(TokenType.LPAREN):
                # Method call: expr.method(...)
                args = self._parse_args()
                span = Span(expr.span.start, self.tokens[self.current-1].span.end, expr.span.line, 0)
                expr = MethodCallExpr(span, expr, field_name, args)
            else:
                # Field access: expr.field
                span = Span(expr.span.start, self.tokens[self.current-1].span.end, expr.span.line, 0)
                expr = FieldAccessExpr(span, expr, field_name)
        
        return expr

    def _parse_args(self) -> List[Expr]:
        self._consume(TokenType.LPAREN, "Expected '('")
        args = []
        if not self._check(TokenType.RPAREN):
            while True:
                args.append(self._expression())
                if not self._match(TokenType.COMMA):
                    break
        self._consume(TokenType.RPAREN, "Expected ')'")
        return args


    def _primary(self) -> Expr:
        if self._match(TokenType.FALSE):
            return LiteralExpr(self.tokens[self.current-1].span, False, "bool")
        if self._match(TokenType.TRUE):
            return LiteralExpr(self.tokens[self.current-1].span, True, "bool")
        if self._match(TokenType.INTEGER):
            return LiteralExpr(self.tokens[self.current-1].span, self.tokens[self.current-1].value, "int")
        if self._match(TokenType.STRING):
            return LiteralExpr(self.tokens[self.current-1].span, self.tokens[self.current-1].value, "string")
        
        if self._match(TokenType.IDENTIFIER):
            identifier_token = self.tokens[self.current-1]
            
            # Check for qualified path: math::add or Option<int>::Some
            path = [identifier_token.lexeme]
            type_args = []
            
            # If the first segment has type args: Box<int>::new
            if self._check(TokenType.LT) and self._is_type_argument_context():
                type_args = self._parse_type_args()
            
            while self._match(TokenType.COLONCOLON):
                path.append(self._consume(TokenType.IDENTIFIER, "Expected identifier after '::'").lexeme)
                # If subsequent segments have type args (less common but possible in some languages)
                # For Forge, let's assume only one set of type args is allowed for now, 
                # either on the first segment (Box<int>::new) or the last (Option::Some<int> - though Forge doesn't support this yet)
                if not type_args and self._check(TokenType.LT) and self._is_type_argument_context():
                    type_args = self._parse_type_args()
            
            # Check if this is a struct instantiation (e.g., Box<int> { value: 10 } or math::Point { ... })
            if self._check(TokenType.LBRACE):
                is_struct = False
                # Look ahead
                if self.tokens[self.current + 1].type == TokenType.RBRACE:
                    is_struct = True # Empty struct Box<int> {}
                elif (self.tokens[self.current + 1].type == TokenType.IDENTIFIER and 
                      self.tokens[self.current + 2].type == TokenType.COLON):
                    is_struct = True # Box<int> { x: ... }
                
                if is_struct:
                    return self._struct_instantiation_path(path, type_args)
            
            # Check for function call: foo(...) or math::add(...)
            if self._check(TokenType.LPAREN):
                args = self._parse_args()
                span = Span(identifier_token.span.start, self.tokens[self.current-1].span.end, identifier_token.span.line, 0)
                return CallExpr(span, path, args)

            # Check for enum instantiation (e.g., Option<int>::Some(42))
            # If we have a path and it looks like an enum (handled by _enum_instantiation_path)
            # This is a bit ambiguous with VariableExpr if it's just a path.
            # In Forge, enum variants are accessed via ::.
            # If the last segment is a variant, it should be handled.
            # For now, if it's a path with ::, we'll try to resolve it in semantic analysis.
            # But if it has payload '(', it's a CallExpr or EnumInstantiation.
            
            return VariableExpr(identifier_token.span, path)
            
        if self._match(TokenType.SELF):
            return VariableExpr(self.tokens[self.current-1].span, "self")
        
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

    def _struct_instantiation_path(self, path: List[str], type_args: List[TypeRef] = None) -> StructInstantiationExpr:
        # Box<int> { value: 42 } or Point { x: 10, y: 20 }
        start_span = self.tokens[self.current-1].span # Approximate
        
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
        return StructInstantiationExpr(span, path, type_args or [], field_values)

    def _enum_instantiation_path(self, path: List[str], type_args: List[TypeRef] = None) -> EnumInstantiationExpr:
        # Option<int>::Some(42) or Option::None
        # Note: path already contains the variant name as the last element if it was parsed in _primary
        # But wait, _enum_instantiation was called when we saw ::.
        # If path is ["Option", "Some"], then enum_path is ["Option"] and variant is "Some".
        if len(path) < 2:
            raise self._error("Expected qualified path for enum instantiation")
            
        enum_path = path[:-1]
        variant_name = path[-1]
        
        start_span = self.tokens[self.current-1].span # Approximate
        
        payload = None
        # Check for payload
        if self._match(TokenType.LPAREN):
            payload = self._expression()
            self._consume(TokenType.RPAREN, "Expected ')' after payload")
        
        span = Span(start_span.start, self.tokens[self.current-1].span.end, start_span.line, 0)
        return EnumInstantiationExpr(span, enum_path, type_args or [], variant_name, payload)

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
        # <T, U> or <T: Display>
        if not self._match(TokenType.LT):
            return []
        
        params = []
        while not self._check(TokenType.GT) and not self._is_at_end():
            param_name_token = self._consume(TokenType.IDENTIFIER, "Expected type parameter name")
            param_name = param_name_token.lexeme
            
            # Parse trait bounds: T: Display + Clone
            bounds = []
            if self._match(TokenType.COLON):
                while True:
                    bound_name = self._consume(TokenType.IDENTIFIER, "Expected trait name in bound").lexeme
                    bounds.append(bound_name)
                    if not self._match(TokenType.PLUS):
                        break
            
            params.append(TypeParameter(param_name_token.span, param_name, bounds))
            
            if not self._match(TokenType.COMMA):
                break
        
        self._consume(TokenType.GT, "Expected '>' after type parameters")
        return params
    
    def _parse_type_ref(self) -> TypeRef:
        """Parse type reference: int, Box<int>, Option<Box<bool>>. Handles nested generics."""
        start_token = self._consume(TokenType.IDENTIFIER, "Expected type name")
        path = [start_token.lexeme]
        
        while self._match(TokenType.COLONCOLON):
            path.append(self._consume(TokenType.IDENTIFIER, "Expected identifier after '::'").lexeme)
        
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
        return TypeRef(span, path, type_args)
    
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
