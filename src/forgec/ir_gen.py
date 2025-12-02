from typing import Dict, Any
from llvmlite import ir
from forgec.ast_nodes import (
    Program, FunctionDef, Stmt, LetStmt, ExprStmt,
    Expr, BinaryExpr, LiteralExpr, VariableExpr, IfExpr, CallExpr,
    StructDef, StructInstantiationExpr, FieldAccessExpr
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

    def generate(self, program: Program) -> str:
        # Register struct types
        for struct in program.structs:
            self._register_struct_type(struct)
        
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
