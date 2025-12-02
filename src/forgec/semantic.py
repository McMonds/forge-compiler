from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from forgec.ast_nodes import (
    Program, FunctionDef, Stmt, LetStmt, ExprStmt,
    Expr, BinaryExpr, LiteralExpr, VariableExpr, IfExpr, CallExpr,
    StructDef, StructInstantiationExpr, FieldAccessExpr,
    EnumDef, EnumInstantiationExpr,
    MatchExpr, EnumPattern
)
from forgec.diagnostics import DiagnosticEngine, Span

@dataclass
class Symbol:
    name: str
    type: str
    span: Span

class SymbolTable:
    def __init__(self, parent: Optional['SymbolTable'] = None):
        self.symbols: Dict[str, Symbol] = {}
        self.parent = parent

    def define(self, name: str, type: str, span: Span) -> bool:
        if name in self.symbols:
            return False
        self.symbols[name] = Symbol(name, type, span)
        return True

    def resolve(self, name: str) -> Optional[Symbol]:
        symbol = self.symbols.get(name)
        if symbol:
            return symbol
        if self.parent:
            return self.parent.resolve(name)
        return None

class TypeChecker:
    def __init__(self, diagnostics: DiagnosticEngine):
        self.diagnostics = diagnostics
        self.global_scope = SymbolTable()
        self.current_scope = self.global_scope
        self.current_function_return_type: Optional[str] = None
        self.struct_schemas: Dict[str, List[tuple[str, str]]] = {}  # struct_name -> [(field_name, field_type), ...]
        self.enum_schemas: Dict[str, List] = {}  # enum_name -> [EnumVariant, ...]


    def check(self, program: Program):
        # First pass: Register all struct types
        for struct in program.structs:
            if struct.name in self.struct_schemas:
                self.diagnostics.error(f"Struct '{struct.name}' is already defined", struct.span)
            else:
                self.struct_schemas[struct.name] = struct.fields
        
        # Register all enum types
        for enum in program.enums:
            if enum.name in self.enum_schemas:
                self.diagnostics.error(f"Enum '{enum.name}' is already defined", enum.span)
            else:
                self.enum_schemas[enum.name] = enum.variants
        
        # Second pass: Define all functions in global scope
        for func in program.functions:
            # For now, function types are simplified
            func_type = f"fn({', '.join(t for _, t in func.params)}) -> {func.return_type}"
            if not self.global_scope.define(func.name, func_type, func.span):
                self.diagnostics.error(f"Function '{func.name}' is already defined", func.span)

        # Third pass: Check function bodies
        for func in program.functions:
            self._check_function(func)

    def _check_function(self, func: FunctionDef):
        self.current_function_return_type = func.return_type
        self._enter_scope()
        
        # Define parameters in scope
        for param_name, param_type in func.params:
            if not self.current_scope.define(param_name, param_type, func.span): # Span is approx
                self.diagnostics.error(f"Parameter '{param_name}' is already defined", func.span)

        for stmt in func.body:
            self._check_stmt(stmt)
        
        self._exit_scope()
        self.current_function_return_type = None

    def _check_stmt(self, stmt: Stmt):
        if isinstance(stmt, LetStmt):
            self._check_let(stmt)
        elif isinstance(stmt, ExprStmt):
            self._check_expr(stmt.expression)
        # Add other statement types

    def _check_let(self, stmt: LetStmt):
        init_type = self._check_expr(stmt.initializer)
        
        if stmt.type_annotation:
            if init_type != stmt.type_annotation:
                self.diagnostics.error(
                    f"Type mismatch: expected '{stmt.type_annotation}', got '{init_type}'",
                    stmt.initializer.span
                )
            final_type = stmt.type_annotation
        else:
            # Inference
            final_type = init_type
            if final_type == "void":
                 self.diagnostics.error("Cannot bind variable to void expression", stmt.span)

        if not self.current_scope.define(stmt.name, final_type, stmt.span):
            self.diagnostics.error(f"Variable '{stmt.name}' is already defined in this scope", stmt.span)

    def _check_expr(self, expr: Expr) -> str:
        if isinstance(expr, LiteralExpr):
            return expr.type_name
        
        if isinstance(expr, VariableExpr):
            symbol = self.current_scope.resolve(expr.name)
            if not symbol:
                self.diagnostics.error(f"Undefined variable '{expr.name}'", expr.span)
                return "error"
            return symbol.type

        if isinstance(expr, BinaryExpr):
            left_type = self._check_expr(expr.left)
            right_type = self._check_expr(expr.right)
            
            if left_type == "error" or right_type == "error":
                return "error"

            if expr.operator in ["+", "-", "*", "/"]:
                if left_type != "int" or right_type != "int":
                    self.diagnostics.error(f"Operator '{expr.operator}' requires integers", expr.span)
                    return "error"
                return "int"
            
            if expr.operator in ["==", "!=", "<", ">"]:
                if left_type != right_type:
                    self.diagnostics.error("Comparison operands must be of the same type", expr.span)
                    return "error"
                return "bool"

        if isinstance(expr, IfExpr):
            cond_type = self._check_expr(expr.condition)
            if cond_type != "bool":
                self.diagnostics.error(f"If condition must be bool, got '{cond_type}'", expr.condition.span)
            
            # Check branches
            # For now, we assume blocks return the type of their last expression if it's an expression stmt?
            # Or we need to implement block type checking properly.
            # Simplified: Check last stmt of block.
            then_type = self._check_block(expr.then_branch)
            else_type = "void"
            if expr.else_branch:
                else_type = self._check_block(expr.else_branch)
            
            if then_type != else_type:
                self.diagnostics.error(f"If branches must return same type. Then: '{then_type}', Else: '{else_type}'", expr.span)
                return "error"
            
            return then_type

        if isinstance(expr, StructInstantiationExpr):
            return self._check_struct_instantiation(expr)
        
        if isinstance(expr, FieldAccessExpr):
            return self._check_field_access(expr)

        if isinstance(expr, EnumInstantiationExpr):
            return self._check_enum_instantiation(expr)

        if isinstance(expr, MatchExpr):
            return self._check_match(expr)

        return "void"

    def _check_match(self, expr: MatchExpr) -> str:
        # Check scrutinee type
        scrutinee_type = self._check_expr(expr.scrutinee)
        
        # Verify scrutinee is an enum
        if scrutinee_type not in self.enum_schemas:
            self.diagnostics.error(f"Match scrutinee must be an enum, got '{scrutinee_type}'", expr.scrutinee.span)
            return "error"
        
        # Get enum variants
        variants = self.enum_schemas[scrutinee_type]
        covered_variants = set()
        
        # Check each arm
        arm_types = []
        for arm in expr.arms:
            if isinstance(arm.pattern, EnumPattern):
                # Verify pattern enum matches scrutinee
                if arm.pattern.enum_name != scrutinee_type:
                    self.diagnostics.error(
                        f"Pattern enum '{arm.pattern.enum_name}' does not match scrutinee type '{scrutinee_type}'",
                        arm.pattern.span
                    )
                    continue
                
                # Verify variant exists
                variant = next((v for v in variants if v.name == arm.pattern.variant_name), None)
                if not variant:
                    self.diagnostics.error(
                        f"Enum '{scrutinee_type}' has no variant '{arm.pattern.variant_name}'",
                        arm.pattern.span
                    )
                    continue
                
                # Track covered variant
                covered_variants.add(arm.pattern.variant_name)
                
                # If there's a binding, add it to scope for the arm body
                if arm.pattern.binding:
                    if not variant.payload_type:
                        self.diagnostics.error(
                            f"Variant '{arm.pattern.variant_name}' has no payload to bind",
                            arm.pattern.span
                        )
                    else:
                        # Create new scope for arm body with binding
                        self.current_scope = SymbolTable(self.current_scope)
                        self.current_scope.define(arm.pattern.binding, variant.payload_type, arm.pattern.span)
                
                # Check arm body
                arm_type = "void"
                for stmt in arm.body:
                    if isinstance(stmt, ExprStmt):
                        arm_type = self._check_expr(stmt.expression)
                    else:
                        self._check_stmt(stmt)
                
                arm_types.append(arm_type)
                
                # Pop scope if we created one
                if arm.pattern.binding:
                    self.current_scope = self.current_scope.parent
        
        # Exhaustiveness check
        variant_names = {v.name for v in variants}
        if covered_variants != variant_names:
            missing = variant_names - covered_variants
            self.diagnostics.error(
                f"Non-exhaustive match: missing variants {missing}",
                expr.span
            )
        
        # All arms should have the same type (simplified)
        if arm_types:
            return arm_types[0]
        return "void"

    def _check_enum_instantiation(self, expr: EnumInstantiationExpr) -> str:
        # Verify enum exists
        if expr.enum_name not in self.enum_schemas:
            self.diagnostics.error(f"Undefined enum '{expr.enum_name}'", expr.span)
            return "error"
        
        # Verify variant exists
        variants = self.enum_schemas[expr.enum_name]
        variant = next((v for v in variants if v.name == expr.variant_name), None)
        if not variant:
            self.diagnostics.error(f"Enum '{expr.enum_name}' has no variant '{expr.variant_name}'", expr.span)
            return "error"
        
        # Check payload
        if variant.payload_type and not expr.payload:
            self.diagnostics.error(f"Variant '{expr.variant_name}' requires a payload of type '{variant.payload_type}'", expr.span)
        elif not variant.payload_type and expr.payload:
            self.diagnostics.error(f"Variant '{expr.variant_name}' does not take a payload", expr.span)
        elif variant.payload_type and expr.payload:
            payload_type = self._check_expr(expr.payload)
            if payload_type != variant.payload_type:
                self.diagnostics.error(
                    f"Variant '{expr.variant_name}' expects payload type '{variant.payload_type}', got '{payload_type}'",
                    expr.payload.span
                )
        
        return expr.enum_name  # Return the enum type name


    def _check_struct_instantiation(self, expr: StructInstantiationExpr) -> str:
        # Verify struct exists
        if expr.struct_name not in self.struct_schemas:
            self.diagnostics.error(f"Undefined struct '{expr.struct_name}'", expr.span)
            return "error"
        
        schema = self.struct_schemas[expr.struct_name]
        provided_fields = {name: value for name, value in expr.field_values}
        
        # Check all required fields are provided
        for field_name, field_type in schema:
            if field_name not in provided_fields:
                self.diagnostics.error(f"Missing field '{field_name}' in struct instantiation", expr.span)
                continue
            
            # Type check the field value
            value_expr = provided_fields[field_name]
            value_type = self._check_expr(value_expr)
            
            if value_type != field_type:
                self.diagnostics.error(
                    f"Field '{field_name}' expects type '{field_type}', got '{value_type}'",
                    value_expr.span
                )
        
        # Check for extra fields
        for field_name, _ in expr.field_values:
            if not any(name == field_name for name, _ in schema):
                self.diagnostics.error(f"Struct '{expr.struct_name}' has no field '{field_name}'", expr.span)
        
        return expr.struct_name  # Return the struct type name

    def _check_field_access(self, expr: FieldAccessExpr) -> str:
        obj_type = self._check_expr(expr.object)
        
        if obj_type == "error":
            return "error"
        
        if obj_type not in self.struct_schemas:
            self.diagnostics.error(f"Cannot access field on non-struct type '{obj_type}'", expr.span)
            return "error"
        
        # Find field type
        schema = self.struct_schemas[obj_type]
        for field_name, field_type in schema:
            if field_name == expr.field_name:
                return field_type
        
        self.diagnostics.error(f"Struct '{obj_type}' has no field '{expr.field_name}'", expr.span)
        return "error"


    def _check_block(self, stmts: List[Stmt]) -> str:
        if not stmts:
            return "void"
        
        # Check all statements
        for stmt in stmts[:-1]:
            self._check_stmt(stmt)
            
        # Check last statement
        last_stmt = stmts[-1]
        if isinstance(last_stmt, ExprStmt):
            return self._check_expr(last_stmt.expression)
        else:
            self._check_stmt(last_stmt)
            return "void"

    def _enter_scope(self):
        self.current_scope = SymbolTable(self.current_scope)

    def _exit_scope(self):
        self.current_scope = self.current_scope.parent
