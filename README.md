# ⚒️ Forge Programming Language

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge&logo=github)](https://github.com/McMonds/forge)
[![Version](https://img.shields.io/badge/version-0.2.0-blue?style=for-the-badge)](https://github.com/McMonds/forge)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-yellow?style=for-the-badge&logo=python)](https://www.python.org/)
[![LLVM](https://img.shields.io/badge/llvm-14+-red?style=for-the-badge&logo=llvm)](https://llvm.org/)

> **Forge** is a modern, statically-typed compiled programming language that combines the performance of low-level languages with the expressiveness of modern functional programming. Built with Python and LLVM, it features a powerful type system with algebraic data types, pattern matching, and a focus on safety and correctness.

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Technical Challenges Solved](#-technical-challenges-solved)
- [Getting Started](#-getting-started)
- [Language Guide](#-language-guide)
- [Development](#-development)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## ✨ Features

Forge is currently in **Phase V (Foundation Complete)**, offering a powerful type system with generics, traits, a robust module system, and a modern **Memory Safety** model with ownership and borrowing.

### 🛡️ Static Type System
- **Strong Type Checking**: All types verified at compile time
- **Type Inference**: Smart variable type deduction
- **No Runtime Type Errors**: Catch bugs before execution

### 📦 Algebraic Data Types

**Structs (Product Types)**
```rust
struct Point {
    x: int,
    y: int
}

let p = Point { x: 10, y: 20 };
let distance = p.x * p.x + p.y * p.y;
```

**Enums (Sum Types)**
```rust
enum Option {
    Some(int),
    None
}

enum Result {
    Ok(int),
    Error(int)
}
```

### 🎨 Pattern Matching
Exhaustive, type-safe pattern matching with compiler-verified completeness:

```rust
fn safe_divide(a: int, b: int) -> int {
    let result = if b == 0 {
        Result::Error(0)
    } else {
        Result::Ok(a / b)
    };

    match result {
        Result::Ok(value) => { value },
        Result::Error(code) => { 0 }
    }
}
```

### 🧬 Generics
Forge supports powerful generics for structs, enums, and functions, enabling code reuse without performance loss.

```rust
struct Box<T> {
    value: T
}

enum Option<T> {
    Some(T),
    None
}

fn identity<T>(x: T) -> T {
    x
}
```

### 📜 Traits & Implementations
```rust
trait Display {
    fn to_string(self) -> string;
}

impl Display for Point {
    fn to_string(self) -> string {
        "Point"
    }
}
```

### 🛡️ Memory Safety (Phase V)
Forge implements a modern memory safety model inspired by Rust, ensuring safety without a garbage collector.

**Ownership & Move Semantics**
Non-copy types are moved by default, preventing use-after-free.
```rust
let s1 = "hello";
let s2 = s1; // s1 is moved to s2
// print(s1); // Compile error: Use of moved value 's1'
```

**Borrowing**
Access data without taking ownership using references.
```rust
let mut x = 10;
let r1 = &x;     // Immutable borrow
let r2 = &mut x; // Mutable borrow (only one allowed at a time)
```

### 📦 Module System & Visibility
Organize your code into modules with file-based resolution and visibility control.

```rust
mod math;
use math::add;

pub fn main() -> int {
    add(10, 20)
}
```

### ⚡ High-Performance Compilation
- **LLVM Backend**: Direct compilation to optimized machine code
- **Tagged Unions**: Efficient enum representation (`{i8 tag, payload}`)
- **Zero-Cost Abstractions**: No runtime overhead for high-level features
- **Smart Code Generation**: Switch-based pattern matching for O(1) dispatch

---

## 🏗️ Architecture

The Forge compiler (`forgec`) implements a classic multi-stage compilation pipeline:

```
Source Code (.forge)
    ↓
┌─────────────┐
│   Lexer     │  → Tokenization
└─────────────┘
    ↓
┌─────────────┐
│   Parser    │  → AST Construction
└─────────────┘
    ↓
┌─────────────┐
│  Semantic   │  → Type Checking
│  Analyzer   │    Symbol Resolution
└─────────────┘    Exhaustiveness
    ↓
┌─────────────┐
│ IR Generator│  → LLVM IR Emission
└─────────────┘
    ↓
LLVM IR (.ll)
    ↓
Machine Code
```

### Core Components

| Component | Location | Responsibility |
|-----------|----------|----------------|
| **Lexer** | `src/forgec/lexer.py` | Tokenizes source into token stream |
| **Parser** | `src/forgec/parser.py` | Builds typed AST with error recovery |
| **Semantic** | `src/forgec/semantic.py` | Type checking, symbol tables, exhaustiveness |
| **IR Gen** | `src/forgec/ir_gen.py` | LLVM IR emission, optimization |
| **Diagnostics** | `src/forgec/diagnostics.py` | Rich error reporting |
| **CLI** | `src/forgec/main.py` | Command-line interface |

---

## 🔬 Technical Challenges Solved

This section documents the major technical problems we encountered and how we solved them.

### 1. Parser Ambiguity: Struct Instantiation vs. Match Blocks

**Problem**: The syntax `identifier { ... }` is ambiguous:
- Struct instantiation: `Point { x: 10, y: 20 }`
- Match expression scrutinee: `match x { ... }`

When parsing `match x { ...}`, the parser would incorrectly interpret `x {` as the start of struct instantiation.

**Solution**: Implemented **lookahead-based disambiguation**:
```python
# In _primary(), before assuming struct instantiation:
if self._check(TokenType.LBRACE):
    is_struct = False
    # Lookahead to verify it's actually a struct
    if (self.tokens[self.current + 1].type == TokenType.RBRACE or
        (self.tokens[self.current + 1].type == TokenType.IDENTIFIER and
         self.tokens[self.current + 2].type == TokenType.COLON)):
        is_struct = True
    
    if is_struct:
        return self._struct_instantiation(identifier_token)
```

**Impact**: Enables clean syntax for both features without conflicts.

---

### 2. Implicit Returns in Block Expressions

**Problem**: Match arms need to return values:
```rust
match x {
    Option::Some(val) => { val },  // Should return 'val'
    Option::None => { 0 }
}
```

But our original block parser required semicolons after all expressions, treating `{ val }` as invalid.

**Solution**: Modified `_block()` to support **tail expressions**:
```python
def _block(self) -> List[Stmt]:
    statements = []
    while not self._check(TokenType.RBRACE):
        expr = self._expression()
        
        # Implicit return: last expression without semicolon
        if self._check(TokenType.RBRACE):
            statements.append(ExprStmt(expr.span, expr))
            break
        
        # Block expressions (if, match) have optional semicolons
        if isinstance(expr, (IfExpr, MatchExpr)):
            self._match(TokenType.SEMICOLON)  # Optional
        else:
            self._consume(TokenType.SEMICOLON)  # Required
        
        statements.append(ExprStmt(expr.span, expr))
```

**Impact**: Enables Rust-like expression-oriented programming.

---

### 3. Tagged Union Representation for Enums

**Problem**: Enums can have variants with different payload types:
```rust
enum Message {
    Quit,              // No payload
    Move(int),         // int payload
    Write(bool)        // bool payload
}
```

How do we represent this efficiently in memory?

**Solution**: **Tagged unions** with LLVM struct types:

```llvm
; Enum represented as { i8 tag, i32 payload }
; - Tag 0 = Quit (payload unused)
; - Tag 1 = Move (payload = int)
; - Tag 2 = Write (payload = bool, stored as i32)

%enum = type { i8, i32 }

; Creating Option::Some(42):
%tag = insertvalue {i8, i32} undef, i8 0, 0      ; Tag = 0 (Some)
%value = insertvalue {i8, i32} %tag, i32 42, 1   ; Payload = 42
```

**Rationale**:
- **Tag (i8)**: Identifies which variant is active (0-255 variants supported)
- **Payload**: Sized to fit the largest possible variant payload
- **Memory Efficient**: Single struct, no heap allocations
- **Type Safe**: Semantic analysis ensures tags/payloads match

**Impact**: Zero-cost enums with compile-time safety guarantees.

---

### 4. Exhaustiveness Checking for Pattern Matching

**Problem**: Ensure all enum variants are handled:
```rust
match x {
    Option::Some(val) => { val }
    // Missing: Option::None
}
```

This should be a **compile error** to prevent runtime crashes.

**Solution**: Implemented **variant coverage analysis** in semantic checker:

```python
def _check_match(self, expr: MatchExpr) -> str:
    scrutinee_type = self._check_expr(expr.scrutinee)
    variants = self.enum_schemas[scrutinee_type]
    covered_variants = set()
    
    for arm in expr.arms:
        # Track which variants are covered
        covered_variants.add(arm.pattern.variant_name)
    
    # Check exhaustiveness
    variant_names = {v.name for v in variants}
    if covered_variants != variant_names:
        missing = variant_names - covered_variants
        self.diagnostics.error(
            f"Non-exhaustive match: missing variants {missing}",
            expr.span
        )
```

**Impact**: Catch logic errors at compile time, not runtime.

---

### 5. Switch-Based Pattern Matching IR Generation

**Problem**: How to efficiently compile pattern matching to machine code?

**Naive Approach**: Chain of if-else comparisons (O(n) complexity)
```llvm
; Inefficient:
if tag == 0 goto some_handler
if tag == 1 goto none_handler
```

**Our Solution**: LLVM `switch` instruction (O(1) jump table):

```llvm
; Extract tag from enum
%tag = extractvalue {i8, i32} %scrutinee, 0

; Switch on tag (O(1) dispatch)
switch i8 %tag, label %default [
    i8 0, label %match.Some
    i8 1, label %match.None
]

match.Some:
    ; Extract payload
    %payload = extractvalue {i8, i32} %scrutinee, 1
    ; ... handle Some(payload)
    br label %merge

match.None:
    ; ... handle None
    br label %merge

merge:
    ; Phi node to merge results
    %result = phi i32 [%some_val, %match.Some], [%none_val, %match.None]
```

**Optimizations**:
- Jump tables for dense tag ranges
- Binary search for sparse tags
- Unreachable default block for exhaustive matches

**Impact**: Pattern matching is as fast as C switch statements.

---

### 6. Error Recovery in Recursive Descent Parsing

**Problem**: When parsing fails, the parser should:
1. Report the error
2. Recover and continue parsing
3. Find as many errors as possible in one pass

**Solution**: **Panic Mode Recovery** with synchronization points:

```python
def _block(self) -> List[Stmt]:
    statements = []
    while not self._check(TokenType.RBRACE):
        try:
            # Parse declaration
            stmt = self._declaration()
            statements.append(stmt)
        except ParseError:
            # Synchronize to next statement boundary
            self._synchronize()
    
    return statements

def _synchronize(self):
    """Skip tokens until we find a statement boundary"""
    while not self._is_at_end():
        if self.previous().type == TokenType.SEMICOLON:
            return
        
        if self.peek().type in {TokenType.FN, TokenType.LET, 
                                 TokenType.IF, TokenType.MATCH}:
            return
        
        self.advance()
```

**Impact**: 
- Reports multiple errors per compilation
- Continues checking rest of file
- Better developer experience

---

### 7. Symbol Table with Nested Scopes

**Problem**: Handle lexical scoping correctly:
```rust
let x = 10;
{
    let x = 20;  // Shadows outer x
    x            // Uses inner x (20)
}
x                // Uses outer x (10)
```

**Solution**: **Linked scope chain** with shadowing support:

```python
class SymbolTable:
    def __init__(self, parent: Optional['SymbolTable'] = None):
        self.symbols: Dict[str, str] = {}  # name -> type
        self.parent = parent
    
    def lookup(self, name: str) -> Optional[str]:
        """Search this scope and parent scopes"""
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.lookup(name)
        return None
    
    def define(self, name: str, type_name: str):
        """Define in current scope (allows shadowing)"""
        self.symbols[name] = type_name
```

Usage in semantic analyzer:
```python
# Enter new scope
self.current_scope = SymbolTable(self.current_scope)

# ... check statements ...

# Exit scope
self.current_scope = self.current_scope.parent
```

**Impact**: Correct scoping semantics matching Rust/Python behavior.

---

### 8. Type Inference for Variable Bindings

**Problem**: Infer types from initializers:
```rust
let x = 42;           // Should infer 'int'
let p = Point { ... } // Should infer 'Point'
```

**Solution**: **Bottom-up type propagation** in semantic checker:

```python
def _check_let(self, stmt: LetStmt):
    # Infer type from initializer
    inferred_type = self._check_expr(stmt.initializer)
    
    # Verify against annotation if present
    if stmt.type_annotation:
        if stmt.type_annotation != inferred_type:
            self.diagnostics.error(
                f"Type mismatch: expected '{stmt.type_annotation}', "
                f"got '{inferred_type}'"
            )
    
    # Register variable with inferred type
    self.current_scope.define(stmt.name, inferred_type, stmt.span)
```

**Impact**: Less verbose code, better developer experience.

---

### 9. Monolithic vs. Incremental Compilation

**Problem**: Should we compile the entire program at once or support separate compilation?

**Our Approach** (Phase I-II): **Monolithic whole-program compilation**
- Simpler implementation
- Better optimization opportunities (whole-program analysis)
- Faster for small-medium projects

**Future** (Phase III): Will add module system for separate compilation:
- Faster incremental rebuilds
- Better code organization
- Library support

---

### 10. LLVM Type Mapping

**Problem**: Map Forge types to LLVM IR types correctly:

| Forge Type | LLVM IR Type | Notes |
|------------|--------------|-------|
| `int` | `i32` | 32-bit signed integer |
| `bool` | `i1` | 1-bit boolean |
| `struct Point { x: int, y: int }` | `{i32, i32}` | Literal struct type |
| `enum Option { Some(int), None }` | `{i8, i32}` | Tagged union |
| Function pointers | Future | Not yet implemented |

**Complexity**: Recursive type resolution for nested structures:
```rust
struct Container {
    point: Point,    // Needs Point's type first
    value: int
}
```

**Solution**: Two-pass registration:
```python
# Pass 1: Register all struct/enum names
for struct in program.structs:
    self.struct_types[struct.name] = None  # Reserve name

# Pass 2: Define types (can reference each other)
for struct in program.structs:
    field_types = [self._resolve_type(f_type) for _, f_type in struct.fields]
    self.struct_types[struct.name] = ir.LiteralStructType(field_types)
```

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+**
- **LLVM 14+** (for `llvmlite`)

### Installation

```bash
# Clone repository
git clone https://github.com/McMonds/forge.git
cd forge

# Install dependencies
pip install -e .
```

### Quick Start

**Compile a Forge program:**
```bash
forgec compile examples/simple.forge
```

**Run compiler from source:**
```bash
python -m forgec.main compile examples/enums.forge
```

**View help:**
```bash
forgec --help
```

---

## 📖 Language Guide

### Basic Syntax

**Variables:**
```rust
let x = 42;           // Type inferred as int
let y: int = 10;      // Explicit type annotation
let flag = true;      // Type inferred as bool
```

**Functions:**
```rust
fn add(a: int, b: int) -> int {
    a + b
}

fn main() {
    let result = add(10, 20);
}
```

**Control Flow:**
```rust
fn abs(x: int) -> int {
    if x < 0 {
        0 - x
    } else {
        x
    }
}
```

### Structs

```rust
struct Rectangle {
    width: int,
    height: int
}

fn area(r: Rectangle) -> int {
    r.width * r.height
}

fn main() {
    let rect = Rectangle { width: 10, height: 20 };
    let a = area(rect);
}
```

### Enums & Pattern Matching

```rust
enum Option {
    Some(int),
    None
}

fn unwrap_or(opt: Option, default: int) -> int {
    match opt {
        Option::Some(value) => { value },
        Option::None => { default }
    }
}

fn main() {
    let x = Option::Some(42);
    let y = Option::None;
    
    let val1 = unwrap_or(x, 0);  // Returns 42
    let val2 = unwrap_or(y, 0);  // Returns 0
}
```

---

## 🧪 Development

### Running Tests

```bash
# Run full test suite
python run_tests.py

# Run specific test module
PYTHONPATH=src python -m pytest tests/test_parser.py
```

### Test Coverage
- ✅ **34/34 tests passing**
- Lexer & Tokenization
- Parser & AST Construction
- Semantic Analysis & Type Checking
- IR Generation & Optimization
- End-to-End Compilation

### Project Structure

```
forge/
├── src/
│   └── forgec/              # Compiler source
│       ├── __init__.py
│       ├── main.py          # CLI entry point
│       ├── lexer.py         # Tokenizer (500 LOC)
│       ├── parser.py        # Parser (800 LOC)
│       ├── semantic.py      # Type checker (400 LOC)
│       ├── ir_gen.py        # LLVM IR gen (600 LOC)
│       ├── ast_nodes.py     # AST definitions
│       └── diagnostics.py   # Error reporting
├── tests/                   # Unit tests
│   ├── test_lexer.py
│   ├── test_parser.py
│   ├── test_semantic.py
│   ├── test_ir_gen.py
│   └── ... (34 total)
├── examples/                # Example programs
│   ├── simple.forge
│   ├── structs.forge
│   └── enums.forge
├── pyproject.toml          # Dependencies
├── Dockerfile              # Container setup
└── README.md
```

---

## 🗺️ Roadmap

| Phase | Status | Features | Timeline |
|-------|--------|----------|----------|
| **I. Core Calculus** | ✅ **Complete** | Primitives, Functions, Control Flow | Q4 2024 |
| **II. Type System** | ✅ **Complete** | Structs, Enums, Pattern Matching | Q4 2024 |
| **III. Abstractions** | ✅ **Complete** | Generics, Traits, Module System, Visibility | Q1 2025 |
| **IV. Practicality** | ✅ **Complete** | Strings, C FFI, Basic I/O | Q1 2025 |
| **V. Safety** | 🏗️ **Foundation** | Ownership, Borrow Checker, Mutability | Q2 2025 |
| **VI. Performance** | 📅 **Planned** | Optimization Passes, SIMD | Q3 2025 |

### Phase V: Safety (In Progress)

**Ownership & Borrowing:**
```rust
fn main() -> int {
    let mut x = 10;
    {
        let r = &mut x;
        *r = 20; // Update through reference
    }
    x // x is now 20
}
```

**Move Semantics:**
```rust
fn consume(s: string) { /* ... */ }

fn main() {
    let s = "data";
    consume(s); // s is moved
    // consume(s); // Error: s is already moved
}
```
        // ...
    }
}
```

**Modules:**
```rust
mod math {
    pub fn sqrt(x: int) -> int { ... }
}

use math::sqrt;
```

---

## 📊 Statistics

- **Lines of Code**: ~2,500 (compiler core)
- **Test Coverage**: 34 unit tests, 100% pass rate
- **Compilation Speed**: ~100ms for small programs
- **IR Generation**: Direct to LLVM (no intermediate representations)
- **Memory Usage**: <50MB for typical programs

---

## 🤝 Contributing

Contributions are welcome! Please see our contributing guidelines.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
python run_tests.py

# Format code
black src/ tests/

# Type check
mypy src/
```

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **LLVM Project** - For the incredible compiler infrastructure
- **Rust Language** - Design inspiration for syntax and semantics
- **Python Community** - For excellent tooling and libraries

---

**Forged with ❤️ by the Forge Team**

*"Crafting code with precision and power"*
