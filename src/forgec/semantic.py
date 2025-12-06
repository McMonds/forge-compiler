from typing import Dict, Optional, List, Any, Set
from dataclasses import dataclass
from forgec.ast_nodes import (
    Program, FunctionDef, Stmt, LetStmt, ExprStmt, ReturnStmt,
    Expr, BinaryExpr, LiteralExpr, VariableExpr, IfExpr, CallExpr,
    StructDef, StructInstantiationExpr, FieldAccessExpr,
    EnumDef, EnumVariant, EnumInstantiationExpr,
    MatchExpr, EnumPattern,
    TypeParameter, TypeRef,
    TraitDef, ImplBlock, TraitMethod,
    CallExpr, MethodCallExpr # NEW
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
        
        # Store full definitions instead of just fields
        self.struct_schemas: Dict[str, StructDef] = {}  # name -> StructDef
        self.enum_schemas: Dict[str, EnumDef] = {}      # name -> EnumDef
        self.function_schemas: Dict[str, FunctionDef] = {} # name -> FunctionDef
        
        # NEW: Generic type system  
        self.type_params: Dict[str, TypeParameter] = {}  # Currently in-scope type parameters
        self.generic_instances: Dict[str, Set[tuple]] = {} # name -> set of type arg tuples
        
        self.current_function_return_type: Optional[str] = None # Track return type for validation: "Box" -> {("int",), ("bool",)}
        self.current_self_type: Optional[str] = None # Track 'Self' type for impl blocks
        
        # NEW: Traits system
        self.traits: Dict[str, TraitDef] = {}  # trait_name -> TraitDef
        self.impls: Dict[tuple, ImplBlock] = {}  # (trait_name, type_name) -> ImplBlockmorphization


    def check(self, program: Program):
        # 1. Register all structs and enums first (types must be known)
        for struct in program.structs:
            if struct.name in self.struct_schemas:
                self.diagnostics.error(f"Struct '{struct.name}' is already defined", struct.span)
            else:
                self.struct_schemas[struct.name] = struct
            
        for enum in program.enums:
            if enum.name in self.enum_schemas:
                self.diagnostics.error(f"Enum '{enum.name}' is already defined", enum.span)
            else:
                self.enum_schemas[enum.name] = enum  # Store full EnumDef
        
        # Second pass: Define all functions in global scope
        for func in program.functions:
            # Build function type string with TypeRef support
            param_types = ', '.join(self._typeref_to_string(t) for _, t in func.params)
            return_type = self._typeref_to_string(func.return_type)
            func_type = f"fn({param_types}) -> {return_type}"
            
            if not self.global_scope.define(func.name, func_type, func.span):
                self.diagnostics.error(f"Function '{func.name}' is already defined", func.span)

        # Register Traits
        for trait in program.traits:
            if trait.name in self.traits:
                self.diagnostics.error(f"Trait '{trait.name}' is already defined", trait.span)
            self.traits[trait.name] = trait

        # Check Impl Blocks
        for impl in program.impls:
            self._check_impl_block(impl)

        # 3. Check functions
        for func in program.functions:
            if func.name in self.function_schemas:
                self.diagnostics.error(f"Function '{func.name}' is already defined", func.span)
            self.function_schemas[func.name] = func
            self._check_function(func)

    def _check_function(self, func: FunctionDef):
        # Enter type parameter scope if generic function
        old_type_params = self._enter_type_param_scope(func.type_params)
        
        # Validate return type
        return_type_str = self._check_type_ref(func.return_type)
        self.current_function_return_type = return_type_str
        
        # Enter function scope
        self.current_scope = SymbolTable(self.current_scope)
        
        # Set current return type
        previous_return_type = self.current_function_return_type
        self.current_function_return_type = self._typeref_to_string(func.return_type)
        
        # Define parameters
        for param_name, param_type_ref in func.params:
            param_type = self._typeref_to_string(param_type_ref)
            self.current_scope.define(param_name, param_type, func.span)
            
        # Check body
        self._check_block(func.body)
        
        # Restore return type
        self.current_function_return_type = previous_return_type
        
        self.current_scope = self.current_scope.parent
        # Exit type parameter scope
        self._exit_type_param_scope(old_type_params)

    def _check_stmt(self, stmt: Stmt):
        if isinstance(stmt, LetStmt):
            self._check_let(stmt)
        elif isinstance(stmt, ExprStmt):
            self._check_expr(stmt.expression)
        elif isinstance(stmt, ReturnStmt):
            self._check_return_stmt(stmt)

    def _check_return_stmt(self, stmt: ReturnStmt):
        if stmt.value:
            return_type = self._check_expr(stmt.value)
            if self.current_function_return_type == "void":
                self.diagnostics.error("Cannot return a value from a void function", stmt.span)
            elif return_type != self.current_function_return_type:
                self.diagnostics.error(
                    f"Type mismatch: expected return type '{self.current_function_return_type}', got '{return_type}'",
                    stmt.value.span
                )
        else:
            if self.current_function_return_type != "void":
                self.diagnostics.error(
                    f"Expected return value of type '{self.current_function_return_type}'",
                    stmt.span
                )
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
        """Wrapper to populate inferred_type"""
        type_name = self._resolve_expr_type(expr)
        expr.inferred_type = type_name
        return type_name

    def _resolve_expr_type(self, expr: Expr) -> str:
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

        if isinstance(expr, MatchExpr):
            return self._check_match(expr)

        if isinstance(expr, CallExpr):
            return self._check_call_expr(expr)

        if isinstance(expr, MethodCallExpr):
            return self._check_method_call_expr(expr)

        return "void"

    def _check_method_call_expr(self, expr: MethodCallExpr) -> str:
        # 1. Check receiver type
        receiver_type = self._check_expr(expr.receiver)
        if receiver_type == "error":
            return "error"
            
        # DEBUG
        # print(f"DEBUG: Checking method '{expr.method_name}' for receiver '{receiver_type}'")
        # print(f"DEBUG: Available impls: {list(self.impls.keys())}")
            
        # 2. Look for method in traits implemented for this type
        # We need to find an impl block that:
        # a) Implements a trait for 'receiver_type'
        # b) Has a method named 'expr.method_name'
        
        # Simplified lookup: Iterate all impls
        # In a real compiler, we'd have a better lookup structure (Type -> [Impls])
        
        found_method = None
        found_impl = None
        
        for (trait_name, type_name), impl in self.impls.items():
            # Check if type matches
            # Note: type_name in impls key might be generic string "Box<int>"
            if type_name == receiver_type:
                # Check if method exists in this impl
                for method in impl.methods:
                    if method.name == expr.method_name:
                        found_method = method
                        found_impl = impl
                        break
            if found_method:
                break
        
        if not found_method:
             self.diagnostics.error(
                 f"Method '{expr.method_name}' not found for type '{receiver_type}'",
                 expr.span
             )
             return "error"
             
        # 3. Check arguments
        # Note: 'self' is the first parameter in trait method, but it's implicit in method call
        # So we check remaining parameters against expr.arguments
        
        expected_params = found_method.params[1:] # Skip self
        
        if len(expected_params) != len(expr.arguments):
            self.diagnostics.error(
                f"Method '{expr.method_name}' expects {len(expected_params)} arguments, got {len(expr.arguments)}",
                expr.span
            )
            return "error"
            
        for (param_name, param_type_ref), arg_expr in zip(expected_params, expr.arguments):
            arg_type = self._check_expr(arg_expr)
            param_type = self._typeref_to_string(param_type_ref)
            
            if arg_type != param_type:
                self.diagnostics.error(
                    f"Argument type mismatch for '{param_name}'. Expected '{param_type}', got '{arg_type}'",
                    arg_expr.span
                )
        
        return self._typeref_to_string(found_method.return_type)

    def _check_call_expr(self, expr: CallExpr) -> str:
        # 1. Check if it's a function call
        # For now, we only support direct function calls by name
        # Future: closures, method calls on objects
        
        # Check if it's a method call (e.g. p.show())
        # The parser parses p.show() as FieldAccessExpr if it's a property access, 
        # but if it's a call, it might be parsed as CallExpr with callee "p.show" or similar?
        # Actually, our parser handles function calls like 'foo(args)' where foo is identifier.
        # Method calls like 'x.method()' are not yet fully supported in parser/AST as distinct from field access + call?
        # Let's check parser.py _primary -> _call
        # In parser, _primary handles identifiers. _call handles '('.
        # If we have x.y(), it parses x.y as FieldAccessExpr, then () as CallExpr?
        # No, CallExpr expects 'callee' to be a string in current AST.
        # Wait, CallExpr definition: callee: str. This limits us to simple function calls.
        # To support methods x.y(), we need to change CallExpr to accept Expr as callee, or handle it differently.
        # For this phase, let's assume we are calling functions or static methods: Trait::method(self, ...)
        # OR, we might have updated CallExpr in a previous step? Let's check AST.
        # AST says: callee: str.
        
        # So for now, we only support function calls: foo(x)
        # If we want to support x.show(), we need to parse it. 
        # But wait, the plan said "Static dispatch (no trait objects)".
        # So we can call Trait::method(x).
        # But we also want x.method() syntax ideally.
        # If the parser doesn't support x.method(), we can't do it yet.
        # Let's stick to function calls for now.
        
        func_name = expr.callee
        
        # Check if function exists
        # It could be a global function or a method in a trait/impl?
        # For now, just global functions.
        
        # We need to look up function definition.
        # We don't have a quick lookup for functions in TypeChecker yet?
        # We iterate in check(). We should store them.
        # Let's add self.functions map in __init__ and populate in check().
        
        # For now, let's just return "int" or "void" as placeholder if we can't find it, 
        # but we should try to find it.
        # Actually, we need to register functions in symbol table or a separate registry.
        
        # Let's assume we have self.function_schemas populated (I'll add it).
        
        if func_name not in self.function_schemas:
             self.diagnostics.error(f"Undefined function '{func_name}'", expr.span)
             return "error"
             
        func_def = self.function_schemas[func_name]
        
        # Check argument count
        if len(func_def.params) != len(expr.arguments):
            self.diagnostics.error(
                f"Function '{func_name}' expects {len(func_def.params)} arguments, got {len(expr.arguments)}",
                expr.span
            )
            return "error"
            
        # Check argument types
        for (param_name, param_type_ref), arg_expr in zip(func_def.params, expr.arguments):
            arg_type = self._check_expr(arg_expr)
            param_type = self._typeref_to_string(param_type_ref)
            
            if arg_type != param_type:
                self.diagnostics.error(
                    f"Argument type mismatch for '{param_name}'. Expected '{param_type}', got '{arg_type}'",
                    arg_expr.span
                )
        
        return self._typeref_to_string(func_def.return_type)

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
        
        enum_def = self.enum_schemas[expr.enum_name]
        
        # Validate type arguments (arity check)
        expected_params = len(enum_def.type_params)
        provided_args = len(expr.type_args)
        
        if expected_params != provided_args:
            self.diagnostics.error(
                f"Enum '{expr.enum_name}' expects {expected_params} type arguments, got {provided_args}",
                expr.span
            )
        
        # Create type substitution map
        type_subst = {}
        if enum_def.type_params and expr.type_args:
            for type_param, type_arg in zip(enum_def.type_params, expr.type_args):
                arg_type_str = self._check_type_ref(type_arg)
                type_subst[type_param.name] = arg_type_str
        
        # Track instantiation
        if expr.type_args:
            inst_key = tuple(self._typeref_to_string(arg) for arg in expr.type_args)
            if expr.enum_name not in self.generic_instances:
                self.generic_instances[expr.enum_name] = set()
            self.generic_instances[expr.enum_name].add(inst_key)
        
        # Verify variant exists
        variant = next((v for v in enum_def.variants if v.name == expr.variant_name), None)
        if not variant:
            self.diagnostics.error(f"Enum '{expr.enum_name}' has no variant '{expr.variant_name}'", expr.span)
            return "error"
        
        # Check payload
        if variant.payload_type and not expr.payload:
            expected_payload_type = self._substitute_type_in_ref(variant.payload_type, type_subst)
            self.diagnostics.error(
                f"Variant '{expr.variant_name}' requires a payload of type '{expected_payload_type}'",
                expr.span
            )
        elif not variant.payload_type and expr.payload:
            self.diagnostics.error(f"Variant '{expr.variant_name}' does not take a payload", expr.span)
        elif variant.payload_type and expr.payload:
            expected_payload_type = self._substitute_type_in_ref(variant.payload_type, type_subst)
            payload_type = self._check_expr(expr.payload)
            if payload_type != expected_payload_type:
                self.diagnostics.error(
                    f"Variant '{expr.variant_name}' expects payload type '{expected_payload_type}', got '{payload_type}'",
                    expr.payload.span
                )
        
        # Return monomorphized type name
        if expr.type_args:
            args_str = ', '.join(self._typeref_to_string(arg) for arg in expr.type_args)
            return f"{expr.enum_name}<{args_str}>"
        return expr.enum_name


    def _check_struct_instantiation(self, expr: StructInstantiationExpr) -> str:
        # Verify struct exists
        if expr.struct_name not in self.struct_schemas:
            self.diagnostics.error(f"Undefined struct '{expr.struct_name}'", expr.span)
            return "error"
        
        struct_def = self.struct_schemas[expr.struct_name]
        
        # Validate type arguments (arity check)
        expected_params = len(struct_def.type_params)
        provided_args = len(expr.type_args)
        
        if expected_params != provided_args:
            self.diagnostics.error(
                f"Struct '{expr.struct_name}' expects {expected_params} type arguments, got {provided_args}",
                expr.span
            )
            # Continue checking even with wrong arity
        
        # Create type substitution map for checking fields
        type_subst = {}
        if struct_def.type_params and expr.type_args:
            for type_param, type_arg in zip(struct_def.type_params, expr.type_args):
                # Validate type argument
                arg_type_str = self._check_type_ref(type_arg)
                type_subst[type_param.name] = arg_type_str
        
        # Track instantiation
        if expr.type_args:
            inst_key = tuple(self._typeref_to_string(arg) for arg in expr.type_args)
            if expr.struct_name not in self.generic_instances:
                self.generic_instances[expr.struct_name] = set()
            self.generic_instances[expr.struct_name].add(inst_key)
        
        provided_fields = {name: value for name, value in expr.field_values}
        
        # Check all required fields are provided
        for field_name, field_type_ref in struct_def.fields:
            if field_name not in provided_fields:
                self.diagnostics.error(f"Missing field '{field_name}' in struct instantiation", expr.span)
                continue
            
            # Get expected field type (with substitution if generic)
            expected_type = self._substitute_type_in_ref(field_type_ref, type_subst)
            
            # Type check the field value
            value_expr = provided_fields[field_name]
            value_type = self._check_expr(value_expr)
            
            if value_type != expected_type:
                self.diagnostics.error(
                    f"Field '{field_name}' expects type '{expected_type}', got '{value_type}'",
                    value_expr.span
                )
        
        # Check for extra fields
        for field_name, _ in expr.field_values:
            if not any(name == field_name for name, _ in struct_def.fields):
                self.diagnostics.error(f"Struct '{expr.struct_name}' has no field '{field_name}'", expr.span)
        
        # Return monomorphized type name
        if expr.type_args:
            args_str = ', '.join(self._typeref_to_string(arg) for arg in expr.type_args)
            return f"{expr.struct_name}<{args_str}>"
        return expr.struct_name

    def _check_field_access(self, expr: FieldAccessExpr) -> str:
        obj_type = self._check_expr(expr.object)
        
        if obj_type == "error":
            return "error"
        
        if obj_type not in self.struct_schemas:
            self.diagnostics.error(f"Cannot access field on non-struct type '{obj_type}'", expr.span)
            return "error"
        
        # Find field type
        schema = self.struct_schemas[obj_type]
        for field_name, field_type in schema.fields:
            if field_name == expr.field_name:
                return self._typeref_to_string(field_type)
        
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
    
    # --- Generic Type System Helpers ---
    
    def _typeref_to_string(self, type_ref: TypeRef) -> str:
        """Convert TypeRef to string representation like 'Box<int>' or 'Option<T>'"""
        if type_ref.name == "Self" and self.current_self_type:
            return self.current_self_type
            
        if not type_ref.type_args:
            return type_ref.name
        
        args_str = ', '.join(self._typeref_to_string(arg) for arg in type_ref.type_args)
        return f"{type_ref.name}<{args_str}>"
    
    def _check_type_ref(self, type_ref: TypeRef) -> str:
        """
        Validate a TypeRef and return its string representation.
        Checks that types exist and type arguments have correct arity.
        """
        # Is it a type parameter?
        if type_ref.name in self.type_params:
            if type_ref.type_args:
                self.diagnostics.error(
                    f"Type parameter '{type_ref.name}' cannot have type arguments",
                    type_ref.span
                )
            return type_ref.name
        
        # Is it a primitive type?
        if type_ref.name in {"int", "bool", "void"}:
            if type_ref.type_args:
                self.diagnostics.error(
                    f"Primitive type '{type_ref.name}' cannot have type arguments",
                    type_ref.span
                )
            return type_ref.name
        
        # Is it a struct?
        if type_ref.name in self.struct_schemas:
            struct_def = self.struct_schemas[type_ref.name]
            expected_params = len(struct_def.type_params)
            provided_args = len(type_ref.type_args)
            
            if expected_params != provided_args:
                self.diagnostics.error(
                    f"Struct '{type_ref.name}' expects {expected_params} type arguments, got {provided_args}",
                    type_ref.span
                )
                # Continue checking args anyway
            
            # Recursively check type arguments
            for arg in type_ref.type_args:
                self._check_type_ref(arg)
            
            # Track instantiation for monomorphization
            if type_ref.type_args:
                inst_key = tuple(self._typeref_to_string(arg) for arg in type_ref.type_args)
                if type_ref.name not in self.generic_instances:
                    self.generic_instances[type_ref.name] = set()
                self.generic_instances[type_ref.name].add(inst_key)
            
            return self._typeref_to_string(type_ref)
        
        # Is it an enum?
        if type_ref.name in self.enum_schemas:
            # Check arity
            enum_def = self.enum_schemas[type_ref.name]
            expected_arity = len(enum_def.type_params)
            actual_arity = len(type_ref.type_args)
            
            if expected_arity != actual_arity:
                self.diagnostics.error(
                    f"Enum '{type_ref.name}' expects {expected_arity} type arguments, but got {actual_arity}",
                    type_ref.span
                )
            
            # Check arguments recursively
            for arg in type_ref.type_args:
                self._check_type_ref(arg)
                
            # Track instantiation for monomorphization
            if type_ref.type_args:
                if type_ref.name not in self.generic_instances:
                    self.generic_instances[type_ref.name] = set()
                
                # Convert args to string tuple for storage
                arg_tuple = tuple(self._typeref_to_string(arg) for arg in type_ref.type_args)
                self.generic_instances[type_ref.name].add(arg_tuple)
                
            return self._typeref_to_string(type_ref)
            
        self.diagnostics.error(f"Undefined type '{type_ref.name}'", type_ref.span)
        return "error"

    def _check_impl_block(self, impl: ImplBlock):
        """Verify impl block matches trait definition"""
        # Check trait exists
        if impl.trait_name not in self.traits:
            self.diagnostics.error(f"Undefined trait '{impl.trait_name}'", impl.span)
            return
        
        trait_def = self.traits[impl.trait_name]
        
        # Check type exists (struct or enum)
        if impl.type_name not in self.struct_schemas and impl.type_name not in self.enum_schemas:
             # Allow primitive types? For now, no.
             self.diagnostics.error(f"Undefined type '{impl.type_name}'", impl.span)
             return
             
        # Build type name (with generics if present)
        if impl.type_args:
            type_full_name = f"{impl.type_name}<{', '.join(str(t) for t in impl.type_args)}>"
        else:
            type_full_name = impl.type_name
        
        # Check all required methods are implemented
        trait_methods = {m.name: m for m in trait_def.methods}
        impl_methods = {m.name: m for m in impl.methods}
        
        # Set Self type for method checking
        previous_self_type = self.current_self_type
        self.current_self_type = type_full_name
        
        for method_name, trait_method in trait_methods.items():
            if method_name not in impl_methods:
                self.diagnostics.error(
                    f"Missing method '{method_name}' in implementation of '{impl.trait_name}' for '{type_full_name}'",
                    impl.span
                )
                continue
            
            # Check signature matches
            impl_method = impl_methods[method_name]
            self._check_method_signature_match(trait_method, impl_method, impl.span)
            
            # Also check the function body itself!
            # We need to enter a scope that includes 'self'
            # For now, just check it as a normal function but we might need special handling for 'self'
            self._check_function(impl_method)
            
        # Restore Self type
        self.current_self_type = previous_self_type
        
        # Store impl
        impl_key = (impl.trait_name, type_full_name)
        self.impls[impl_key] = impl

    def _check_method_signature_match(self, trait_method: TraitMethod, impl_method: FunctionDef, span):
        """Verify impl method signature matches trait method"""
        # Check parameter count
        if len(trait_method.params) != len(impl_method.params):
            self.diagnostics.error(
                f"Method '{trait_method.name}' expects {len(trait_method.params)} parameters, got {len(impl_method.params)}", 
                span
            )
            return
        
        # Check parameter types (skip 'self' which is first param)
        # Note: trait_method.params is [(name, TypeRef)], impl_method.params is [(name, TypeRef)]
        for i, ((t_name, t_type), (i_name, i_type)) in enumerate(zip(trait_method.params, impl_method.params)):
            if i == 0 and t_name == "self":
                # Skip self check for now, assume valid if present
                continue
                
            if self._typeref_to_string(t_type) != self._typeref_to_string(i_type):
                self.diagnostics.error(
                    f"Parameter '{t_name}' type mismatch in method '{trait_method.name}'. Expected '{t_type}', got '{i_type}'", 
                    span
                )
        
        # Check return type
        if self._typeref_to_string(trait_method.return_type) != self._typeref_to_string(impl_method.return_type):
            self.diagnostics.error(
                f"Return type mismatch in method '{trait_method.name}'. Expected '{trait_method.return_type}', got '{impl_method.return_type}'", 
                span
            )
    
    def _enter_type_param_scope(self, type_params: List[TypeParameter]):
        """Enter scope with given type parameters"""
        old_params = self.type_params.copy()
        for tp in type_params:
            if tp.name in self.type_params:
                self.diagnostics.error(
                    f"Type parameter '{tp.name}' is already defined",
                    tp.span
                )
            self.type_params[tp.name] = tp
        return old_params
    
    def _substitute_type_in_ref(self, type_ref: TypeRef, type_subst: Dict[str, str]) -> str:
        """Substitute type parameters in a TypeRef. Returns string type name."""
        # If this is a type parameter, substitute it
        if type_ref.name in type_subst:
            return type_subst[type_ref.name]
        
        # Otherwise just convert to string (handles concrete types)
        return self._typeref_to_string(type_ref)
    
    def _exit_type_param_scope(self, old_params: Dict[str, TypeParameter]):
        """Restore previous type parameter scope"""
        self.type_params = old_params
