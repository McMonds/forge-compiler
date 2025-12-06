from dataclasses import dataclass
from typing import List, Optional, Union, Any
from forgec.diagnostics import Span

@dataclass
class ASTNode:
    span: Span

# --- Type System ---

@dataclass
class TypeParameter(ASTNode):
    """Generic type parameter like 'T' in Box<T>"""
    name: str
    # Future: bounds like T: Display

@dataclass  
class TypeRef(ASTNode):
    """Reference to a type (possibly generic)"""
    name: str                           # e.g., "Box", "int", "T"
    type_args: List['TypeRef']          # e.g., [TypeRef("int")] for Box<int>
    
    def __str__(self) -> str:
        if not self.type_args:
            return self.name
        args_str = ", ".join(str(arg) for arg in self.type_args)
        return f"{self.name}<{args_str}>"

# Expressions
@dataclass
class Expr(ASTNode):
    pass

@dataclass
class LiteralExpr(Expr):
    value: Any
    type_name: str  # "int", "bool"

@dataclass
class VariableExpr(Expr):
    name: str

@dataclass
class BinaryExpr(Expr):
    left: Expr
    operator: str
    right: Expr

@dataclass
class CallExpr(Expr):
    callee: str
    arguments: List[Expr]

@dataclass
class IfExpr(Expr):
    condition: Expr
    then_branch: List['Stmt'] # Block is a list of statements
    else_branch: Optional[List['Stmt']]

# Statements / Declarations
@dataclass
class Stmt(ASTNode):
    pass

@dataclass
class LetStmt(Stmt):
    name: str
    initializer: Expr
    type_annotation: Optional[str] = None

@dataclass
class ExprStmt(Stmt):
    expression: Expr

@dataclass
class FunctionDef(Stmt):
    name: str
    type_params: List[TypeParameter]       # NEW: Generic parameters like <T>
    params: List[tuple[str, TypeRef]]       # Changed from str to TypeRef
    return_type: TypeRef                   # Changed from str to TypeRef
    body: List[Stmt]
    is_public: bool = False                # For module system

# Struct-related nodes
@dataclass
class StructDef(Stmt):
    name: str
    type_params: List[TypeParameter]      # NEW: Generic parameters like <T, U>
    fields: List[tuple[str, TypeRef]]      # Changed from str to TypeRef

@dataclass
class StructInstantiationExpr(Expr):
    struct_name: str
    type_args: List[TypeRef]               # NEW: Type arguments like Box<int>
    field_values: List[tuple[str, Expr]]   # (field_name, value)

@dataclass
class FieldAccessExpr(Expr):
    object: Expr  # The struct instance
    field_name: str

# Enum-related nodes
@dataclass
class EnumVariant(ASTNode):
    name: str
    payload_type: Optional[TypeRef]        # Changed from str to TypeRef

@dataclass
class EnumDef(Stmt):
    name: str
    type_params: List[TypeParameter]       # NEW: Generic parameters
    variants: List[EnumVariant]

@dataclass
class EnumInstantiationExpr(Expr):
    enum_name: str
    type_args: List[TypeRef]               # NEW: Type arguments
    variant_name: str
    payload: Optional[Expr]  # None for unit variants

# Pattern matching nodes
@dataclass
class Pattern(ASTNode):
    pass

@dataclass
class EnumPattern(Pattern):
    enum_name: str
    variant_name: str
    binding: Optional[str]  # Variable name to bind payload to (e.g., "x" in Some(x))

@dataclass
class MatchArm(ASTNode):
    pattern: Pattern
    body: List[Stmt]  # Block of statements for this arm

@dataclass
class MatchExpr(Expr):
    scrutinee: Expr  # The value being matched
    arms: List[MatchArm]

@dataclass
class Program(ASTNode):
    functions: List[FunctionDef]
    structs: List[StructDef]  # Add struct definitions to program
    enums: List[EnumDef]  # Add enum definitions to program

