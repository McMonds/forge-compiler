from typing import Dict, Any
from llvmlite import ir
from forgec.ast_nodes import (
    Program, FunctionDef, Stmt, LetStmt, ExprStmt,
    Expr, BinaryExpr, LiteralExpr, VariableExpr, IfExpr, CallExpr,
    StructDef, StructInstantiationExpr, FieldAccessExpr,
    EnumDef, EnumInstantiationExpr,
    MatchExpr, EnumPattern
)
from forgec.diagnostics import DiagnosticEngine

class IRGenerator:
    def __init__(self, diagnostics: DiagnosticEngine):
        self.diagnostics = diagnostics
        self.module = ir.Module(name="forge_module")
        self.builder = None
        self.func_symtab: Dict[str, ir.Value] = {}
        self.current_func = None

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

    def generate(self, program: Program) -> str:
        # Register struct types
        for struct in program.structs:
            self._register_struct_type(struct)
        
        # Register enum types
        for enum in program.enums:
            self._register_enum_type(enum)
        
        for func in program.functions:
            self._gen_function(func)
        return str(self.module)

    def _register_struct_type(self, struct: StructDef):
        # Store schema for later use
        self.struct_schemas[struct.name] = struct.fields
        
        # Map field types to LLVM types
        field_types = []
        for _, type_name in struct.fields:
            if type_name == "int":
                field_types.append(self.int_type)
            elif type_name == "bool":
                field_types.append(self.bool_type)
            else:
                # Could be another struct
                if type_name in self.struct_types:
                    field_types.append(self.struct_types[type_name])
                else:
                    raise Exception(f"Unknown type: {type_name}")
        
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


    def _gen_function(self, func: FunctionDef):
        # Define function type
        # Simplified: assuming int args and return for now, or void
        arg_types = [self.int_type for _ in func.params] # TODO: Map types correctly
        
        ret_type = self.void_type
        if func.return_type == "int":
            ret_type = self.int_type
        elif func.return_type == "bool":
            ret_type = self.bool_type

        func_type = ir.FunctionType(ret_type, arg_types)
        ir_func = ir.Function(self.module, func_type, name=func.name)
        
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
            ptr = self.builder.alloca(self.int_type, name=arg_name)
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

        return ir.Constant(self.int_type, 0) # Fallback

    def _gen_struct_instantiation(self, expr: StructInstantiationExpr) -> ir.Value:
        struct_type = self.struct_types[expr.struct_name]
        schema = self.struct_schemas[expr.struct_name]
        
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
        
        # Determine the struct type from the object
        # We need to know which struct this is to find the field index
        # For now, we'll extract from the variable if it's a VariableExpr
        if isinstance(expr.object, VariableExpr):
            # Load the struct from memory
            ptr = self.func_symtab.get(expr.object.name)
            if ptr:
                obj = self.builder.load(ptr)
        
        # Find the field index
        # We need the struct type name - this is a limitation of our current approach
        # In a real compiler, we'd track types through the IR generation
        # For now, we'll infer from the loaded value's type
        struct_type = obj.type
        
        # Find which struct this type corresponds to
        struct_name = None
        for name, stype in self.struct_types.items():
            if stype == struct_type:
                struct_name = name
                break
        
        if not struct_name:
            raise Exception(f"Could not determine struct type for field access")
        
        schema = self.struct_schemas[struct_name]
        field_idx = next(i for i, (name, _) in enumerate(schema) if name == expr.field_name)
        
        # Extract the field
        return self.builder.extract_value(obj, field_idx, name=expr.field_name)

    def _gen_enum_instantiation(self, expr: EnumInstantiationExpr) -> ir.Value:
        enum_type = self.enum_types[expr.enum_name]
        variants = self.enum_schemas[expr.enum_name]
        
        # Find variant index (tag)
        tag = next(i for i, v in enumerate(variants) if v.name == expr.variant_name)
        
        # Create enum value (tagged union)
        enum_val = ir.Constant(enum_type, ir.Undefined)
        
        # Set tag (first field)
        enum_val = self.builder.insert_value(enum_val, ir.Constant(ir.IntType(8), tag), 0, name=f"{expr.enum_name}.tag")
        
        # Set payload (second field) if exists
        if expr.payload:
            payload_val = self._gen_expr(expr.payload)
            enum_val = self.builder.insert_value(enum_val, payload_val, 1, name=f"{expr.enum_name}.{expr.variant_name}")
        else:
            # For unit variants, set payload to 0 (unused)
            enum_val = self.builder.insert_value(enum_val, ir.Constant(self.int_type, 0), 1, name=f"{expr.enum_name}.{expr.variant_name}")
        
        return enum_val


        return enum_val

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
