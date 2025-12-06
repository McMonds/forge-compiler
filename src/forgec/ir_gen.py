from typing import Dict, Any
from llvmlite import ir
from forgec.ast_nodes import (
    Program, FunctionDef, Stmt, LetStmt, ExprStmt, ReturnStmt,
    Expr, BinaryExpr, LiteralExpr, VariableExpr, IfExpr, CallExpr, MethodCallExpr,
    StructDef, StructInstantiationExpr, FieldAccessExpr,
    EnumDef, EnumInstantiationExpr,
    MatchExpr, EnumPattern,
    TypeParameter, TypeRef,
    ImplBlock
)
from forgec.diagnostics import DiagnosticEngine

class IRGenerator:
    def __init__(self, program: Program, diagnostics):
        self.diagnostics = diagnostics
        self.module = ir.Module(name="main")
        self.builder = None
        self.current_func = None
        self.named_values = {}  # Variable name -> ir.Value
        
        # Standard types
        self.int_type = ir.IntType(32)
        self.bool_type = ir.IntType(1)
        self.void_type = ir.VoidType()
        
        # Struct types
        self.struct_types: Dict[str, ir.Type] = {}  # struct_name -> LLVM struct type
        self.struct_schemas: Dict[str, list] = {}  # struct_name -> [(field_name, field_type), ...]
        
        # Enum types (tagged unions)
        self.enum_types: Dict[str, ir.Type] = {}  # enum_name -> LLVM struct type { i8 tag, payload }
        self.enum_schemas: Dict[str, list] = {}  # enum_name -> [EnumVariant, ...]
        
        # NEW: Monomorphization infrastructure
        self.mono_cache: Dict[tuple, ir.Type] = {}  # (type_name, tuple(type_args)) -> ir.Type
        # Example: ("Box", ("int",)) -> %Box_int = { i32 }
        
        # Store original definitions for monomorphization
        # Store original definitions for monomorphization
        self.struct_defs: Dict[str, StructDef] = {s.name: s for s in program.structs}
        self.enum_defs: Dict[str, EnumDef] = {e.name: e for e in program.enums}
        
        self.trait_impls: Dict[tuple, ImplBlock] = {}
        self.current_self_type: Optional[ir.Type] = None # Track 'Self' type for impl blocks
        self.enum_defs: Dict[str, EnumDef] = {e.name: e for e in program.enums}
        
        # NEW: Traits system
        self.trait_impls: Dict[tuple, ImplBlock] = {}  # (trait_name, type_name) -> ImplBlock

    def generate(self, program: Program) -> str:
        # Register struct types
        for struct in program.structs:
            self._register_struct_type(struct)
        
        # Register enum types
        for enum in program.enums:
            self._register_enum_type(enum)
            
        # Register impl blocks
        for impl in program.impls:
            # Store for resolution: (Trait, Type) -> Impl
            # Note: For generic types, we might need more complex key, but for now use string representation
            if impl.type_args:
                type_name = f"{impl.type_name}<{', '.join(str(t) for t in impl.type_args)}>"
            else:
                type_name = impl.type_name
            self.trait_impls[(impl.trait_name, type_name)] = impl
            
            # Generate code for methods
            self._gen_impl_block(impl)
        
        for func in program.functions:
            self._gen_function(func)
        return str(self.module)

    def _gen_impl_block(self, impl: ImplBlock):
        """Generate code for methods in an impl block"""
        # For now, we only support non-generic impls or we rely on monomorphization
        # If it's a generic impl (impl<T> Display for Box<T>), we skip generation here
        # and generate on demand during monomorphization (similar to generic structs)
        # BUT for Phase III simplified, we might assume concrete types or handle simple generics
        
        # Name mangling: {TypeName}_{TraitName}_{MethodName}
        # e.g. Point_Display_show
        
        type_name = impl.type_name
        if impl.type_args:
            # If generic, we might skip if it's a template, or generate if it's a concrete instantiation
            # For now, let's assume concrete instantiation or skip
            pass 
            
        # Resolve Self type
        if type_name in self.struct_types:
            self.current_self_type = self.struct_types[type_name]
        elif type_name in self.enum_types:
            self.current_self_type = self.enum_types[type_name]
        else:
            # Fallback or error? For now, assume it's handled elsewhere or primitive
            pass
            
        for method in impl.methods:
            mangled_name = f"{type_name}_{impl.trait_name}_{method.name}"
            self._gen_function(method, name_override=mangled_name)
            
        self.current_self_type = None

    def _register_struct_type(self, struct: StructDef):
        # Skip generic structs - they'll be monomorphized on demand
        if struct.type_params:
            return
        
        # Non-generic struct: create LLVM type
        field_types = []
        field_schema = []
        for field_name, field_type_ref in struct.fields:
            # Convert TypeRef to LLVM type
            field_type_ir = self._typeref_to_ir_type(field_type_ref)
            field_types.append(field_type_ir)
            field_schema.append((field_name, field_type_ref))  # Store for later
        
        # Create LLVM struct type
        struct_type = ir.LiteralStructType(field_types)
        self.struct_types[struct.name] = struct_type

    def _register_enum_type(self, enum: EnumDef):
        # Store schema for later use
        self.enum_schemas[enum.name] = enum.variants
        
        # Determine the largest payload type
        # For now, we'll use i32 as the payload type (supports int)
        # In a real compiler, we'd determine the union of all payload types
        payload_type = self.int_type
        
        # Tagged union: { i8 tag, i32 payload }
        # tag identifies which variant is active
        enum_type = ir.LiteralStructType([ir.IntType(8), payload_type])
        self.enum_types[enum.name] = enum_type


    def _gen_function(self, func: FunctionDef, name_override: str = None):
        # Determine function type
        arg_types = []
        for _, arg_type_ref in func.params:
            arg_types.append(self._typeref_to_ir_type(arg_type_ref))
            
        ret_type = self._typeref_to_ir_type(func.return_type)
        func_type = ir.FunctionType(ret_type, arg_types)
        
        # Create function
        func_name = name_override if name_override else func.name
        ir_func = ir.Function(self.module, func_type, name=func_name)
        
        # Create entry block
        block = ir_func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)
        self.current_func = ir_func
        self.func_symtab = {}

        # Register args
        for i, (arg_name, _) in enumerate(func.params):
            arg = ir_func.args[i]
            arg.name = arg_name
            # Allocate space for arg to be mutable (if we want mutable args)
            # For now, just map name to value
            ptr = self.builder.alloca(arg.type, name=arg_name)
            self.builder.store(arg, ptr)
            self.func_symtab[arg_name] = ptr

        # Generate body
        for stmt in func.body:
            self._gen_stmt(stmt)

        # Add implicit return void if needed
        if ret_type == self.void_type and not block.is_terminated:
            self.builder.ret_void()

    def _gen_stmt(self, stmt: Stmt):
        if isinstance(stmt, LetStmt):
            self._gen_let(stmt)
        elif isinstance(stmt, ExprStmt):
            self._gen_expr(stmt.expression)
        elif isinstance(stmt, ReturnStmt):
            self._gen_return_stmt(stmt)

    def _gen_return_stmt(self, stmt: ReturnStmt):
        if stmt.value:
            val = self._gen_expr(stmt.value)
            self.builder.ret(val)
        else:
            self.builder.ret_void()

    def _gen_let(self, stmt: LetStmt):
        # Calculate initializer value
        init_val = self._gen_expr(stmt.initializer)
        
        # Determine type
        typ = init_val.type
            
        ptr = self.builder.alloca(typ, name=stmt.name)
        self.builder.store(init_val, ptr)
        self.func_symtab[stmt.name] = ptr

    def _gen_expr(self, expr: Expr) -> ir.Value:
        if isinstance(expr, LiteralExpr):
            if expr.type_name == "int":
                return ir.Constant(self.int_type, expr.value)
            elif expr.type_name == "bool":
                return ir.Constant(self.bool_type, 1 if expr.value else 0)
        
        if isinstance(expr, VariableExpr):
            ptr = self.func_symtab.get(expr.name)
            if not ptr:
                # Should have been caught by semantic analysis
                raise Exception(f"Codegen error: Undefined variable {expr.name}")
            return self.builder.load(ptr, name=expr.name)

        if isinstance(expr, BinaryExpr):
            lhs = self._gen_expr(expr.left)
            rhs = self._gen_expr(expr.right)
            
            if expr.operator == "+":
                return self.builder.add(lhs, rhs, name="addtmp")
            elif expr.operator == "-":
                return self.builder.sub(lhs, rhs, name="subtmp")
            elif expr.operator == "*":
                return self.builder.mul(lhs, rhs, name="multmp")
            elif expr.operator == "/":
                return self.builder.sdiv(lhs, rhs, name="divtmp")
            # Comparisons
            elif expr.operator == "==":
                return self.builder.icmp_signed("==", lhs, rhs, name="eqtmp")
            elif expr.operator == "!=":
                return self.builder.icmp_signed("!=", lhs, rhs, name="neqtmp")
            elif expr.operator == "<":
                return self.builder.icmp_signed("<", lhs, rhs, name="lttmp")
            elif expr.operator == ">":
                return self.builder.icmp_signed(">", lhs, rhs, name="gttmp")

        if isinstance(expr, StructInstantiationExpr):
            return self._gen_struct_instantiation(expr)
        
        if isinstance(expr, FieldAccessExpr):
            return self._gen_field_access(expr)

        if isinstance(expr, EnumInstantiationExpr):
            return self._gen_enum_instantiation(expr)

        if isinstance(expr, MatchExpr):
            return self._gen_match(expr)

        if isinstance(expr, IfExpr):
            return self._gen_if(expr)

        if isinstance(expr, CallExpr):
            return self._gen_call_expression(expr)

        if isinstance(expr, CallExpr):
            return self._gen_call_expression(expr)

        if isinstance(expr, MethodCallExpr):
            return self._gen_method_call_expr(expr)

        return ir.Constant(self.int_type, 0) # Fallback

    def _gen_struct_instantiation(self, expr: StructInstantiationExpr) -> ir.Value:
        # Get the appropriate struct type (monomorphized if generic)
        if expr.type_args:
            # Generic instantiation: Box<int> { value: 42 }
            struct_type = self._monomorphize_struct(expr.struct_name, expr.type_args)
            struct_def = self.struct_defs[expr.struct_name]
            schema = struct_def.fields  # Use original definition fields
        else:
            # Non-generic: Point { x: 10, y: 20 }
            struct_type = self.struct_types[expr.struct_name]
            schema = self.struct_defs[expr.struct_name].fields
        
        # Create an undefined struct value
        struct_val = ir.Constant(struct_type, ir.Undefined)
        
        # Insert each field value
        field_map = {name: value_expr for name, value_expr in expr.field_values}
        for idx, (field_name, _) in enumerate(schema):
            if field_name in field_map:
                field_value = self._gen_expr(field_map[field_name])
                struct_val = self.builder.insert_value(struct_val, field_value, idx, name=f"{expr.struct_name}.{field_name}")
        
        return struct_val

    def _gen_field_access(self, expr: FieldAccessExpr) -> ir.Value:
        # Get the struct value
        obj = self._gen_expr(expr.object)
        
        # Use inferred type from semantic analysis
        struct_full_name = expr.object.inferred_type
        if not struct_full_name:
             raise Exception(f"Codegen error: No inferred type for field access on '{expr.object}'")
             
        # Extract base name (e.g. "Box" from "Box<int>")
        struct_name = struct_full_name.split('<')[0]
        
        if struct_name not in self.struct_defs:
             raise Exception(f"Codegen error: Unknown struct '{struct_name}'")
             
        schema = self.struct_defs[struct_name]
        
        # Find field index
        try:
            field_idx = next(i for i, (name, _) in enumerate(schema.fields) if name == expr.field_name)
        except StopIteration:
            raise Exception(f"Codegen error: Field '{expr.field_name}' not found in '{struct_name}'")
        
        # Extract the field
        return self.builder.extract_value(obj, field_idx, name=expr.field_name)

    def _gen_enum_instantiation(self, expr: EnumInstantiationExpr) -> ir.Value:
        # Get the appropriate enum type (monomorphized if generic)
        if expr.type_args:
            # Generic instantiation: Option<int>::Some(42)
            enum_type = self._monomorphize_enum(expr.enum_name, expr.type_args)
            enum_def = self.enum_defs[expr.enum_name]
            variants = enum_def.variants
        else:
            # Non-generic enum
            enum_type = self.enum_types[expr.enum_name]
            variants = self.enum_schemas[expr.enum_name]
        
        # Find variant index (tag)
        tag = next(i for i, v in enumerate(variants) if v.name == expr.variant_name)
        
        # Create tagged union: { i8 tag, payload }
        tag_constant = ir.Constant(ir.IntType(8), tag)
        tagged_union = self.builder.insert_value(ir.Constant(enum_type, ir.Undefined), tag_constant, 0)
        
        # Insert payload if exists
        variant = next(v for v in variants if v.name == expr.variant_name)
        if variant.payload_type and expr.payload:
            payload_val = self._gen_expr(expr.payload)
            tagged_union = self.builder.insert_value(tagged_union, payload_val, 1)
        else:
            # Unit variant - insert zero payload
            # The type of the zero payload must match the second element of the enum_type
            # which is the largest payload type determined during monomorphization.
            payload_ir_type = enum_type.elements[1]
            zero = ir.Constant(payload_ir_type, 0)
            tagged_union = self.builder.insert_value(tagged_union, zero, 1)
        
        return tagged_union

    def _gen_call_expression(self, expr: CallExpr) -> ir.Value:
        # 1. Look up function
        if expr.callee not in self.module.globals:
             # Should have been caught by semantic analysis
             raise Exception(f"Codegen error: Undefined function {expr.callee}")
             
        func = self.module.globals[expr.callee]
        
        # 2. Generate arguments
        args = [self._gen_expr(arg) for arg in expr.arguments]
        
        # 3. Call
        return self.builder.call(func, args)

    def _gen_method_call_expr(self, expr: MethodCallExpr) -> ir.Value:
        # 1. Generate receiver
        receiver_val = self._gen_expr(expr.receiver)
        
        # 2. Determine receiver type
        # We rely on Semantic Analysis having populated inferred_type
        receiver_type = expr.receiver.inferred_type
        if not receiver_type:
             raise Exception(f"Codegen error: No inferred type for receiver in method call '{expr.method_name}'")
             
        # 3. Find trait implementation
        # We need to find the mangled name: {TypeName}_{TraitName}_{MethodName}
        # We iterate trait_impls to find the trait that this type implements and has this method
        
        target_func_name = None
        
        for (trait_name, type_name), impl in self.trait_impls.items():
            if type_name == receiver_type:
                # Check if method is in this impl
                for method in impl.methods:
                    if method.name == expr.method_name:
                        # Found it!
                        target_func_name = f"{type_name}_{trait_name}_{method.name}"
                        break
            if target_func_name:
                break
                
        if not target_func_name:
             raise Exception(f"Codegen error: Method '{expr.method_name}' not found for type '{receiver_type}'")
             
        # 4. Look up function
        if target_func_name not in self.module.globals:
             raise Exception(f"Codegen error: Missing function '{target_func_name}'")
             
        func = self.module.globals[target_func_name]
        
        # 5. Generate arguments (prepend receiver)
        args = [receiver_val] + [self._gen_expr(arg) for arg in expr.arguments]
        
        # 6. Call
        return self.builder.call(func, args)

    def _gen_match(self, expr: MatchExpr) -> ir.Value:
        # 1. Generate scrutinee
        scrutinee_val = self._gen_expr(expr.scrutinee)
        
        # 2. Extract tag
        # scrutinee is {i8, payload}
        tag_val = self.builder.extract_value(scrutinee_val, 0, name="match.tag")
        
        # 3. Create merge block
        merge_block = self.current_func.append_basic_block(name="match.merge")
        
        # 4. Create switch
        # We need a default block, but if exhaustive, it's unreachable
        default_block = self.current_func.append_basic_block(name="match.default")
        switch = self.builder.switch(tag_val, default_block)
        
        # Default block is unreachable
        self.builder.position_at_start(default_block)
        self.builder.unreachable()
        
        # 5. Generate arms
        incoming_values = []
        incoming_blocks = []
        
        # We need to know the enum type to map variants to tags
        # We can infer it from scrutinee type, but we don't track types in IR gen perfectly yet.
        # However, we have self.enum_schemas and self.enum_types.
        # We need to find which enum this is.
        # For now, let's assume we can find it by matching type structure?
        # Or better, we can look up the enum name from the pattern!
        # The first pattern must be an EnumPattern with the enum name.
        
        first_pattern = expr.arms[0].pattern
        if not isinstance(first_pattern, EnumPattern):
            raise Exception("Only enum patterns supported")
            
        enum_name = first_pattern.enum_name
        variants = self.enum_schemas[enum_name]
        
        for arm in expr.arms:
            pattern = arm.pattern
            if not isinstance(pattern, EnumPattern):
                continue
                
            # Find tag for this variant
            tag = next(i for i, v in enumerate(variants) if v.name == pattern.variant_name)
            
            # Create block for this arm
            arm_block = self.current_func.append_basic_block(name=f"match.{pattern.variant_name}")
            switch.add_case(ir.Constant(ir.IntType(8), tag), arm_block)
            
            self.builder.position_at_start(arm_block)
            
            # Handle binding
            if pattern.binding:
                # Extract payload
                payload_val = self.builder.extract_value(scrutinee_val, 1, name=f"match.payload.{pattern.binding}")
                
                # Store in symtab (allocate stack space)
                # We need to know the type of the payload
                # We can get it from the variant definition
                variant = variants[tag]
                # Map type name to IR type
                payload_type = self.int_type # Default
                if variant.payload_type == "bool":
                    payload_type = self.bool_type
                # TODO: Support other types
                
                ptr = self.builder.alloca(payload_type, name=pattern.binding)
                self.builder.store(payload_val, ptr)
                
                # Add to scope (shadowing)
                # We need to save old value if any?
                # For simplicity, we just overwrite in func_symtab and restore later?
                # Or better, IR generation doesn't need to restore because we are in a new block?
                # But func_symtab is global for function.
                # We should save/restore.
                old_binding = self.func_symtab.get(pattern.binding)
                self.func_symtab[pattern.binding] = ptr
            
            # Generate body
            body_val = self._gen_block(arm.body)
            
            # Branch to merge
            self.builder.branch(merge_block)
            
            incoming_values.append(body_val)
            incoming_blocks.append(self.builder.block)
            
            # Restore binding
            if pattern.binding:
                if old_binding:
                    self.func_symtab[pattern.binding] = old_binding
                else:
                    del self.func_symtab[pattern.binding]
        
        # 6. Merge block
        self.builder.position_at_start(merge_block)
        
        # Phi node for result
        # Assuming all arms return same type (int for now)
        phi = self.builder.phi(self.int_type, name="match.result")
        for val, block in zip(incoming_values, incoming_blocks):
            phi.add_incoming(val, block)
            
        return phi
    
    # --- Monomorphization Helpers ---
    
    def _typeref_to_ir_type(self, type_ref: TypeRef, type_subst: Dict[str, ir.Type] = None) -> ir.Type:
        """Convert TypeRef to LLVM IR type, handling generics via monomorphization"""
        if type_subst is None:
            type_subst = {}
        
        # Check if it's a type parameter that should be substituted
        if type_ref.name in type_subst:
            return type_subst[type_ref.name]
            
        # Check for Self type
        if type_ref.name == "Self" and self.current_self_type:
            return self.current_self_type
        
        # Primitive types
        if type_ref.name == "int":
            return self.int_type
        elif type_ref.name == "bool":
            return self.bool_type
        elif type_ref.name == "void":
            return self.void_type
        
        # Generic struct
        if type_ref.name in self.struct_defs:
            if type_ref.type_args:
                return self._monomorphize_struct(type_ref.name, type_ref.type_args)
            else:
                # Non-generic struct (or generic with no args - error, but handled elsewhere)
                return self.struct_types.get(type_ref.name, self.int_type)
        
        # Generic enum
        if type_ref.name in self.enum_defs:
            if type_ref.type_args:
                return self._monomorphize_enum(type_ref.name, type_ref.type_args)
            else:
                return self.enum_types.get(type_ref.name, self.int_type)
        
        # Fallback
        return self.int_type
    
    def _monomorphize_struct(self, struct_name: str, type_args: list) -> ir.Type:
        """Generate concrete LLVM struct type for generic instantiation like Box<int>"""
        # Create cache key
        type_arg_strs = tuple(self._typeref_to_string(arg) for arg in type_args)
        cache_key = (struct_name, type_arg_strs)
        
        # Check cache
        if cache_key in self.mono_cache:
            return self.mono_cache[cache_key]
        
        # Get original generic definition
        struct_def = self.struct_defs[struct_name]
        
        # Create type substitution map: T -> int, U -> bool, etc.
        type_subst = {}
        for type_param, type_arg in zip(struct_def.type_params, type_args):
            ir_type = self._typeref_to_ir_type(type_arg)
            type_subst[type_param.name] = ir_type
        
        # Generate field types with substitution
        field_types = []
        for field_name, field_type_ref in struct_def.fields:
            field_ir_type = self._typeref_to_ir_type(field_type_ref, type_subst)
            field_types.append(field_ir_type)
        
        # Create monomorphized LLVM type
        mono_name = f"{struct_name}_{'_'.join(type_arg_strs)}"
        mono_type = ir.LiteralStructType(field_types)
        
        # Cache it
        self.mono_cache[cache_key] = mono_type
        
        return mono_type
    
    def _monomorphize_enum(self, enum_name: str, type_args: list) -> ir.Type:
        """Generate concrete LLVM enum type (tagged union) for generic instantiation"""
        # Create cache key
        type_arg_strs = tuple(self._typeref_to_string(arg) for arg in type_args)
        cache_key = (enum_name, type_arg_strs)
        
        # Check cache
        if cache_key in self.mono_cache:
            return self.mono_cache[cache_key]
        
        # Get original generic definition
        enum_def = self.enum_defs[enum_name]
        
        # Create type substitution map
        type_subst = {}
        for type_param, type_arg in zip(enum_def.type_params, type_args):
            ir_type = self._typeref_to_ir_type(type_arg)
            type_subst[type_param.name] = ir_type
        
        # For enums, we need to find the largest payload type after substitution
        # Tagged union: { i8 tag, <largest_payload_type> }
        tag_type = ir.IntType(8)
        max_payload_type = self.int_type  # Default
        
        for variant in enum_def.variants:
            if variant.payload_type:
                payload_ir_type = self._typeref_to_ir_type(variant.payload_type, type_subst)
                # Simple heuristic: use the "largest" (just pick any for now)
                max_payload_type = payload_ir_type
        
        # Create tagged union
        mono_type = ir.LiteralStructType([tag_type, max_payload_type])
        
        # Cache it
        self.mono_cache[cache_key] = mono_type
        
        return mono_type
    
    def _typeref_to_string(self, type_ref: TypeRef) -> str:
        """Convert TypeRef to string for cache keys"""
        if not type_ref.type_args:
            return type_ref.name
        args_str = '_'.join(self._typeref_to_string(arg) for arg in type_ref.type_args)
        return f"{type_ref.name}_{args_str}"

    def _gen_if(self, expr: IfExpr) -> ir.Value:
        # 1. Generate condition
        cond_val = self._gen_expr(expr.condition)
        
        # 2. Create blocks
        then_block = self.current_func.append_basic_block(name="then")
        else_block = self.current_func.append_basic_block(name="else")
        merge_block = self.current_func.append_basic_block(name="merge")
        
        # 3. Conditional branch
        # Ensure condition is i1 (bool)
        if cond_val.type != self.bool_type:
            # This should be caught by semantic analysis, but for safety:
            cond_val = self.builder.icmp_signed("!=", cond_val, ir.Constant(self.int_type, 0), name="tobool")
            
        self.builder.cbranch(cond_val, then_block, else_block)
        
        # 4. Generate 'then' block
        self.builder.position_at_start(then_block)
        then_val = self._gen_block(expr.then_branch)
        self.builder.branch(merge_block)
        # Capture the block we ended up in (might be different if nested ifs)
        then_end_block = self.builder.block
        
        # 5. Generate 'else' block
        self.builder.position_at_start(else_block)
        if expr.else_branch:
            else_val = self._gen_block(expr.else_branch)
        else:
            else_val = ir.Constant(self.int_type, 0) # Default for if without else (should be unit/void)
        self.builder.branch(merge_block)
        else_end_block = self.builder.block
        
        # 6. Merge block (Phi node)
        self.builder.position_at_start(merge_block)
        phi = self.builder.phi(self.int_type, name="ifresult")
        phi.add_incoming(then_val, then_end_block)
        phi.add_incoming(else_val, else_end_block)
        
        return phi

    def _gen_block(self, stmts: list[Stmt]) -> ir.Value:
        # Generate all statements in the block
        # Return the value of the last statement if it's an expression, else 0
        last_val = ir.Constant(self.int_type, 0)
        for stmt in stmts:
            if isinstance(stmt, ExprStmt):
                last_val = self._gen_expr(stmt.expression)
            else:
                self._gen_stmt(stmt)
        return last_val
