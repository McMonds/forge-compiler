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
    TraitDef, ImplBlock, TraitMethod,
    CallExpr, MethodCallExpr,
    ModDecl, UseDecl # NEW
)
from forgec.diagnostics import DiagnosticEngine, Span

@dataclass
class Symbol:
    name: str
    type: str
    span: Span
    is_public: bool = False
    full_name: Optional[str] = None # NEW: For cross-module lookup
    module_scope: Optional['SymbolTable'] = None # For modules

class SymbolTable:
    def __init__(self, parent: Optional['SymbolTable'] = None):
        self.symbols: Dict[str, Symbol] = {}
        self.parent = parent

    def define(self, name: str, type: str, span: Span, is_public: bool = False, full_name: Optional[str] = None) -> bool:
        if name in self.symbols:
            return False
        self.symbols[name] = Symbol(name, type, span, is_public, full_name)
        return True

    def define_alias(self, name: str, symbol: Symbol) -> bool:
        if name in self.symbols:
            return False
        self.symbols[name] = symbol
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
        self.current_module_prefix: str = "" # NEW: Track current module for visibility checks
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
        
        # NEW: Modules
        self.modules: Dict[str, ModDecl] = {}

    def check(self, program: Program, prefix: str = ""):
        old_prefix = self.current_module_prefix
        self.current_module_prefix = prefix
        
        # Pass 1: Register all types and modules across the entire tree
        self._pass_register_types_and_modules(program, prefix)
        
        # Pass 2: Register all functions across the entire tree
        self._pass_register_functions(program, prefix)
        
        # Pass 3: Resolve all imports across the entire tree
        self._pass_resolve_imports(program)
        
        # Pass 4: Register Traits and check Impl blocks
        self._pass_check_traits_and_impls(program, prefix)
        
        # Pass 5: Check all function bodies
        self._pass_check_bodies(program)
        
        self.current_module_prefix = old_prefix

    def _pass_register_types_and_modules(self, program: Program, prefix: str = ""):
        # Register structs
        for struct in program.structs:
            full_name = f"{prefix}{struct.name}" if prefix else struct.name
            if full_name in self.struct_schemas:
                self.diagnostics.error(f"Struct '{full_name}' is already defined", struct.span)
            else:
                self.struct_schemas[full_name] = struct
            
        # Register enums
        for enum in program.enums:
            full_name = f"{prefix}{enum.name}" if prefix else enum.name
            if full_name in self.enum_schemas:
                self.diagnostics.error(f"Enum '{full_name}' is already defined", enum.span)
            else:
                self.enum_schemas[full_name] = enum
        
        # Register modules and recurse
        for mod in program.modules:
            full_name = f"{prefix}{mod.name}" if prefix else mod.name
            if not self.current_scope.define(mod.name, "module", mod.span, mod.is_public, full_name):
                self.diagnostics.error(f"Module '{mod.name}' is already defined", mod.span)
                continue
            
            self.modules[mod.name] = mod
            
            if mod.body:
                # Create scope for the module
                mod_scope = SymbolTable(parent=self.current_scope)
                
                # Store the module scope in the symbol
                mod_symbol = self.current_scope.resolve(mod.name)
                if mod_symbol:
                    mod_symbol.module_scope = mod_scope
                
                # Recurse into module
                old_scope = self.current_scope
                self.current_scope = mod_scope
                new_prefix = f"{prefix}{mod.name}::"
                self._pass_register_types_and_modules(mod.body, new_prefix)
                self.current_scope = old_scope

    def _pass_register_functions(self, program: Program, prefix: str = ""):
        # Register functions in current scope
        for func in program.functions:
            full_name = f"{prefix}{func.name}" if prefix else func.name
            param_types = ', '.join(self._typeref_to_string(t) for _, t in func.params)
            return_type = self._typeref_to_string(func.return_type)
            func_type = f"fn({param_types}) -> {return_type}"
            
            if not self.current_scope.define(func.name, func_type, func.span, func.is_public, full_name):
                self.diagnostics.error(f"Function '{func.name}' is already defined", func.span)
            
            self.function_schemas[full_name] = func
            
        # Recurse into modules
        for mod in program.modules:
            mod_symbol = self.current_scope.resolve(mod.name)
            if mod_symbol and mod_symbol.module_scope and mod.body:
                old_scope = self.current_scope
                self.current_scope = mod_symbol.module_scope
                new_prefix = f"{prefix}{mod.name}::"
                self._pass_register_functions(mod.body, new_prefix)
                self.current_scope = old_scope

    def _pass_resolve_imports(self, program: Program):
        # Resolve imports in current scope
        for imp in program.imports:
            self._check_use_decl(imp)
            
        # Recurse into modules
        for mod in program.modules:
            mod_symbol = self.current_scope.resolve(mod.name)
            if mod_symbol and mod_symbol.module_scope and mod.body:
                old_scope = self.current_scope
                self.current_scope = mod_symbol.module_scope
                self._pass_resolve_imports(mod.body)
                self.current_scope = old_scope

    def _pass_check_traits_and_impls(self, program: Program, prefix: str = ""):
        # Register Traits
        for trait in program.traits:
            full_name = f"{prefix}{trait.name}" if prefix else trait.name
            if full_name in self.traits:
                self.diagnostics.error(f"Trait '{full_name}' is already defined", trait.span)
            self.traits[full_name] = trait

        # Check Impl Blocks
        for impl in program.impls:
            self._check_impl_block(impl)
            
        # Recurse into modules
        for mod in program.modules:
            mod_symbol = self.current_scope.resolve(mod.name)
            if mod_symbol and mod_symbol.module_scope and mod.body:
                old_scope = self.current_scope
                self.current_scope = mod_symbol.module_scope
                new_prefix = f"{prefix}{mod.name}::"
                self._pass_check_traits_and_impls(mod.body, new_prefix)
                self.current_scope = old_scope

    def _pass_check_bodies(self, program: Program):
        # Check function bodies
        for func in program.functions:
            self._check_function(func)
            
        # Recurse into modules
        for mod in program.modules:
            mod_symbol = self.current_scope.resolve(mod.name)
            if mod_symbol and mod_symbol.module_scope and mod.body:
                old_scope = self.current_scope
                self.current_scope = mod_symbol.module_scope
                self._pass_check_bodies(mod.body)
                self.current_scope = old_scope

        
    def _check_use_decl(self, imp: UseDecl):
        # Path is a string like "math::add"
        path_segments = imp.path.split("::")
        
        # Resolve the path
        symbol = self._resolve_path(path_segments, span=imp.span)
        if not symbol:
            self.diagnostics.error(f"Undefined import path '{imp.path}'", imp.span)
            return
            
        # Define alias in current scope
        # e.g., use math::add -> 'add' refers to math::add symbol
        name = path_segments[-1]
        if not self.current_scope.define_alias(name, symbol):
            self.diagnostics.error(f"Symbol '{name}' is already defined in this scope", imp.span)

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

    def _resolve_path(self, path: List[str], check_visibility: bool = True, span: Optional[Span] = None) -> Optional[Symbol]:
        """Resolve a qualified path like math::add or x"""
        scope = self.current_scope
        symbol = None
        
        for i, segment in enumerate(path):
            symbol = scope.resolve(segment)
            if not symbol:
                return None
            
            # Visibility check:
            # If we are traversing into a module (i > 0),
            # we must check if the symbol is public, UNLESS it's in the current module.
            if check_visibility and i > 0:
                # Get the module part of the full name
                if symbol.full_name and "::" in symbol.full_name:
                    symbol_module = symbol.full_name.rsplit("::", 1)[0] + "::"
                else:
                    symbol_module = ""
                
                if symbol_module != self.current_module_prefix:
                    if not symbol.is_public:
                        error_span = span if span else symbol.span
                        self.diagnostics.error(f"Symbol '{segment}' is private", error_span)
            
            # If there are more segments, the current symbol must be a module
            if i < len(path) - 1:
                if symbol.type != "module" or not symbol.module_scope:
                    # Error: segment is not a module
                    return None
                scope = symbol.module_scope
        
        return symbol

    def _resolve_expr_type(self, expr: Expr) -> str:
        if isinstance(expr, LiteralExpr):
            return expr.type_name
        
        if isinstance(expr, VariableExpr):
            symbol = self._resolve_path(expr.path, span=expr.span)
            if not symbol:
                self.diagnostics.error(f"Undefined variable '{'::'.join(expr.path)}'", expr.span)
                return "error"
            
            # Store resolved name for IR generation
            expr.resolved_name = symbol.full_name if symbol.full_name else "::".join(expr.path)
            
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
        # ... (unchanged for now, methods are different)
        return "void" # Placeholder to avoid too much change at once

    def _check_call_expr(self, expr: CallExpr) -> str:
        func_path = expr.callee_path
        func_name = "::".join(func_path)
        
        # Resolve path to function symbol
        symbol = self._resolve_path(func_path, span=expr.span)
        
        if not symbol or not symbol.type.startswith("fn("):
             self.diagnostics.error(f"Undefined function '{func_name}'", expr.span)
             return "error"
             
        # For now, we still need the FunctionDef to check arguments
        # We should probably store FunctionDef in Symbol or have a way to look it up
        # Use the symbol's full name if available, otherwise fallback to the path name
        lookup_name = symbol.full_name if symbol.full_name else func_name
        
        if lookup_name not in self.function_schemas:
             self.diagnostics.error(f"Function schema not found for '{lookup_name}'", expr.span)
             return "error"
             
        func_def = self.function_schemas[lookup_name]
        
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
        
        # Store the resolved name for IR generation
        expr.resolved_name = lookup_name
        
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
        enum_name = "::".join(expr.enum_path)
        
        # Resolve path to enum symbol
        symbol = self._resolve_path(expr.enum_path, span=expr.span)
        if symbol and symbol.full_name:
            enum_name = symbol.full_name
            
        # Store resolved name for IR generation
        expr.resolved_name = enum_name
        
        # Verify enum exists
        if enum_name not in self.enum_schemas:
            self.diagnostics.error(f"Undefined enum '{enum_name}'", expr.span)
            return "error"
        
        enum_def = self.enum_schemas[enum_name]
        
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
        struct_name = "::".join(expr.struct_path)
        
        # Resolve path to struct symbol
        symbol = self._resolve_path(expr.struct_path, span=expr.span)
        if symbol and symbol.full_name:
            struct_name = symbol.full_name
            
        # Store resolved name for IR generation
        expr.resolved_name = struct_name
        
        # Verify struct exists
        if struct_name not in self.struct_schemas:
            self.diagnostics.error(f"Undefined struct '{struct_name}'", expr.span)
            return "error"
        
        struct_def = self.struct_schemas[struct_name]
        
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
        name = "::".join(type_ref.path)
        if name == "Self" and self.current_self_type:
            return self.current_self_type
            
        if not type_ref.type_args:
            return name
        
        args_str = ', '.join(self._typeref_to_string(arg) for arg in type_ref.type_args)
        return f"{name}<{args_str}>"
    
    def _check_type_ref(self, type_ref: TypeRef) -> str:
        """
        Validate a TypeRef and return its string representation.
        Checks that types exist and type arguments have correct arity.
        """
        # Resolve path to type symbol
        symbol = self._resolve_path(type_ref.path, span=type_ref.span)
        if symbol and symbol.full_name:
            name = symbol.full_name
        else:
            name = "::".join(type_ref.path)
            
        # Store resolved name for IR generation
        type_ref.resolved_name = name
        
        # Is it a type parameter?
        if name in self.type_params:
            if type_ref.type_args:
                self.diagnostics.error(
                    f"Type parameter '{name}' cannot have type arguments",
                    type_ref.span
                )
            return name
        
        # Is it a primitive type?
        if name in {"int", "bool", "void"}:
            if type_ref.type_args:
                self.diagnostics.error(
                    f"Primitive type '{name}' cannot have type arguments",
                    type_ref.span
                )
            return name
        
        # Is it a struct?
        if name in self.struct_schemas:
            struct_def = self.struct_schemas[name]
            expected_params = len(struct_def.type_params)
            provided_args = len(type_ref.type_args)
            
            if expected_params != provided_args:
                self.diagnostics.error(
                    f"Struct '{name}' expects {expected_params} type arguments, got {provided_args}",
                    type_ref.span
                )
                # Continue checking args anyway
            
            # Recursively check type arguments
            for arg in type_ref.type_args:
                self._check_type_ref(arg)
            
            # Track instantiation for monomorphization
            if type_ref.type_args:
                inst_key = tuple(self._typeref_to_string(arg) for arg in type_ref.type_args)
                if name not in self.generic_instances:
                    self.generic_instances[name] = set()
                self.generic_instances[name].add(inst_key)
            
            return self._typeref_to_string(type_ref)
        
        # Is it an enum?
        if name in self.enum_schemas:
            # Check arity
            enum_def = self.enum_schemas[name]
            expected_arity = len(enum_def.type_params)
            actual_arity = len(type_ref.type_args)
            
            if expected_arity != actual_arity:
                self.diagnostics.error(
                    f"Enum '{name}' expects {expected_arity} type arguments, but got {actual_arity}",
                    type_ref.span
                )
            
            # Check arguments recursively
            for arg in type_ref.type_args:
                self._check_type_ref(arg)
                
            # Track instantiation for monomorphization
            if type_ref.type_args:
                if name not in self.generic_instances:
                    self.generic_instances[name] = set()
                
                # Convert args to string tuple for storage
                arg_tuple = tuple(self._typeref_to_string(arg) for arg in type_ref.type_args)
                self.generic_instances[name].add(arg_tuple)
                
            return self._typeref_to_string(type_ref)
            
        self.diagnostics.error(f"Undefined type '{name}'", type_ref.span)
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
