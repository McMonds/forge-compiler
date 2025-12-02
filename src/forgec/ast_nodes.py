from dataclasses import dataclass
from typing import List, Optional, Union, Any
from forgec.diagnostics import Span

@dataclass
class ASTNode:
    span: Span

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
    params: List[tuple[str, str]] # (name, type)
    return_type: str
    body: List[Stmt]

# Struct-related nodes
@dataclass
class StructDef(Stmt):
    name: str
    fields: List[tuple[str, str]]  # [(field_name, type_name), ...]

@dataclass
class StructInstantiationExpr(Expr):
    struct_name: str
    field_values: List[tuple[str, Expr]]  # [(field_name, value_expr), ...]

@dataclass
class FieldAccessExpr(Expr):
    object: Expr  # The struct instance
    field_name: str

# Enum-related nodes
@dataclass
class EnumVariant:
    name: str
    payload_type: Optional[str]  # None for unit variants (e.g., None in Option)

@dataclass
class EnumDef(Stmt):
    name: str
    variants: List[EnumVariant]

@dataclass
class EnumInstantiationExpr(Expr):
    enum_name: str
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

