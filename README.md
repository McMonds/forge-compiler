# ⚒️ Forge Compiler

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge&logo=github)](https://github.com/monk/forge)
[![Version](https://img.shields.io/badge/version-0.2.0-blue?style=for-the-badge)](https://github.com/monk/forge)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-yellow?style=for-the-badge&logo=python)](https://www.python.org/)
[![LLVM](https://img.shields.io/badge/llvm-14+-red?style=for-the-badge&logo=llvm)](https://llvm.org/)

> **Forge** is a modern, statically-typed compiled language designed to bridge the gap between low-level performance and high-level expressiveness. Built with Python and LLVM, it offers a clean syntax inspired by Rust and Python, featuring a powerful type system with algebraic data types and pattern matching.

---

## ✨ Key Features

Forge is currently in **Phase II (The Expressive System)**, delivering a robust set of features for building complex applications.

### 🛡️ Strong Static Typing
Forge enforces type safety at compile time, preventing common runtime errors.
- **Primitives**: `int` (32-bit signed), `bool` (true/false).
- **Inference**: Smart type inference for variable bindings.

### 📦 Algebraic Data Types (ADTs)
Define expressive data structures that can represent complex states.

**Structs (Product Types)**
```rust
struct Point {
    x: int,
    y: int
}

let p = Point { x: 10, y: 20 };
let x_val = p.x;
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
Safely handle control flow and destructure data with exhaustive pattern matching.

```rust
fn safe_divide(a: int, b: int) -> int {
    let result = if b == 0 {
        Option::None
    } else {
        Option::Some(a / b)
    };

    match result {
        Option::Some(val) => { val },
        Option::None => { 0 } // Handle error case
    }
}
```

### ⚡ High-Performance Compilation
- **LLVM Backend**: Compiles directly to optimized machine code via LLVM IR.
- **Zero-Cost Abstractions**: Enums and structs are compiled to efficient memory representations (tagged unions).

---

## 🛠️ Architecture

The Forge compiler (`forgec`) is built as a modular pipeline:

1.  **Lexer**: Tokenizes source code into a stream of tokens (`src/forgec/lexer.py`).
2.  **Parser**: Recursive descent parser with robust error recovery, building a typed AST (`src/forgec/parser.py`).
3.  **Semantic Analyzer**: Performs symbol resolution, type checking, and exhaustiveness verification (`src/forgec/semantic.py`).
4.  **IR Generator**: Emits optimized LLVM IR using `llvmlite`, handling memory layout and control flow (`src/forgec/ir_gen.py`).

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+**
- **LLVM 14+** (Required for `llvmlite`)

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/monk/forge.git
cd forge
pip install -e .
```

### Usage

**Compile a file:**
```bash
forgec compile examples/enums.ae
```

**Run from source:**
```bash
python -m forgec.main compile examples/simple.ae
```

**View Help:**
```bash
forgec --help
```

---

## 🧪 Development & Testing

We maintain a comprehensive test suite covering all compiler stages.

**Run all tests:**
```bash
python run_tests.py
```

**Current Test Coverage:**
- ✅ Lexer & Parser (Error recovery, precedence)
- ✅ Semantic Analysis (Type checking, scoping)
- ✅ IR Generation (Structs, Enums, Control Flow)
- ✅ End-to-End Compilation

---

## 🗺️ Roadmap

| Phase | Status | Features |
|-------|--------|----------|
| **I. Core Calculus** | ✅ Done | Primitives, Functions, Control Flow, Let Bindings |
| **II. Expressive System** | ✅ Done | Structs, Enums, Pattern Matching, Type Inference |
| **III. Abstractions** | ⏳ Next | Generics (`Struct<T>`), Traits (`impl Display`), Modules |
| **IV. Safety** | 🔮 Future | Borrow Checker, Memory Safety, Concurrency |

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

*Forged with ❤️ by the Forge Team.*
