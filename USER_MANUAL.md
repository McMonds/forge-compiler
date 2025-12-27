# Forge User Manual

**Version 0.2.0** | **Last Updated: December 2024**

Welcome to **Forge**, a modern statically-typed compiled programming language designed for building robust, efficient software. This manual will guide you through installation, language features, and practical usage.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Language Fundamentals](#2-language-fundamentals)
3. [Data Types](#3-data-types)
4. [Control Flow](#4-control-flow)
5. [Functions](#5-functions)
6. [Structs](#6-structs)
7. [Enums](#7-enums)
8. [Pattern Matching](#8-pattern-matching)
9. [Using the Compiler](#9-using-the-compiler)
10. [Examples](#10-examples)
11. [Troubleshooting](#11-troubleshooting)
12. [Best Practices](#12-best-practices)
13. [Advanced Features](#13-advanced-features)
14. [Generics](#14-generics)
15. [Traits](#15-traits)
16. [Module System](#16-module-system)

---

## 1. Getting Started

### 1.1 Prerequisites

Before installing Forge, ensure you have:

- **Python 3.10 or higher**
  ```bash
  python --version  # Should show 3.10+
  ```

- **LLVM 14 or higher** (required for `llvmlite`)
  ```bash
  # Ubuntu/Debian
  sudo apt install llvm-14 llvm-14-dev
  
  # macOS (Homebrew)
  brew install llvm@14
  
  # Windows
  # Download from https://releases.llvm.org/
  ```

### 1.2 Installation

**From Source (Recommended):**

```bash
# Clone the repository
git clone https://github.com/McMonds/forge-compiler.git
cd forge-compiler

# Install in development mode
pip install -e .

# Verify installation
forgec --help
```

**Using pip (Future):**
```bash
# Not yet available on PyPI
pip install forgec
```

### 1.3 Quick Test

Create your first Forge program:

```rust
// hello.forge
fn main() {
    let message = 42;
    message
}
```

Compile it:
```bash
forgec compile hello.forge
```

This generates `hello.ll` (LLVM IR file).

---

## 2. Language Fundamentals

### 2.1 Comments

```rust
// Single-line comment

/*
   Multi-line comments
   not yet supported
*/
```

### 2.2 Variables

Variables in Forge are **immutable by default** (like Rust).

**Basic Declaration:**
```rust
let x = 42;              // Type inferred as 'int'
let flag = true;         // Type inferred as 'bool'
```

**Explicit Type Annotation:**
```rust
let count: int = 100;
let is_active: bool = false;
```

**Variable Shadowing:**
```rust
let x = 10;
let x = 20;    // Shadows the previous 'x'
```

### 2.3 Type System

Forge is **statically typed** with strong type checking at compile time.

**Primitive Types:**

| Type | Description | Example |
|------|-------------|---------|
| `int` | 32-bit signed integer | `42`, `-100`, `0` |
| `bool` | Boolean | `true`, `false` |

**Type Inference:**
```rust
let x = 42;          // Inferred as int
let valid = true;    // Inferred as bool
```

---

## 3. Data Types

### 3.1 Integers

```rust
let positive = 100;
let negative = -50;
let zero = 0;
```

**Arithmetic Operations:**
```rust
let sum = 10 + 20;        // 30
let diff = 50 - 30;       // 20
let product = 5 * 6;      // 30
let quotient = 20 / 4;    // 5
```

**Comparison Operations:**
```rust
let a = 10;
let b = 20;

let equal = a == b;       // false
let not_equal = a != b;   // true
let less = a < b;         // true
let greater = a > b;      // false
```

### 3.2 Booleans

```rust
let is_valid = true;
let is_error = false;

// Boolean expressions in conditions
if is_valid {
    // ...
}
```

---

## 4. Control Flow

### 4.1 If Expressions

In Forge, `if` is an **expression** that returns a value.

**Basic If:**
```rust
let x = 10;
let result = if x > 5 {
    1
} else {
    0
};
// result = 1
```

**Nested If:**
```rust
fn classify(n: int) -> int {
    if n < 0 {
        -1
    } else {
        if n == 0 {
            0
        } else {
            1
        }
    }
}
```

**If-Else Chain:**
```rust
let score = 85;
let grade = if score >= 90 {
    5  // A
} else {
    if score >= 80 {
        4  // B
    } else {
        if score >= 70 {
            3  // C
        } else {
            2  // D or F
        }
    }
};
```

### 4.2 Blocks

Blocks are sequences of statements enclosed in `{}`. The last expression in a block is its **return value**.

```rust
let result = {
    let a = 10;
    let b = 20;
    a + b  // Returns 30
};
```

**With Semicolons:**
```rust
let x = {
    let temp = 5;
    temp * 2;  // Semicolon = statement, returns void
};
// x has type 'void' (not recommended)
```

---

## 5. Functions

### 5.1 Function Definition

```rust
fn function_name(param1: type1, param2: type2) -> return_type {
    // function body
}
```

**Example:**
```rust
fn add(a: int, b: int) -> int {
    a + b
}
```

### 5.2 Calling Functions

```rust
fn main() {
    let sum = add(10, 20);  // sum = 30
    let double = multiply(5, 2);
}

fn multiply(x: int, y: int) -> int {
    x * y
}
```

### 5.3 Return Values

The last expression in a function is **implicitly returned**:

```rust
fn square(n: int) -> int {
    n * n  // Implicitly returned
}
```

**Explicit Return (Future):**
```rust
// 'return' keyword not yet implemented
```

### 5.4 Void Functions

Functions that don't return a value:

```rust
fn print_number(n: int) -> void {
    // No return value
}
```

---

## 6. Structs

Structs are **product types** that group related data.

### 6.1 Defining Structs

```rust
struct Point {
    x: int,
    y: int
}

struct Rectangle {
    width: int,
    height: int
}
```

### 6.2 Creating Struct Instances

```rust
let origin = Point { x: 0, y: 0 };
let point = Point { x: 10, y: 20 };
```

**Field Order Doesn't Matter:**
```rust
let p1 = Point { x: 5, y: 10 };
let p2 = Point { y: 10, x: 5 };  // Same as p1
```

### 6.3 Accessing Fields

```rust
let point = Point { x: 10, y: 20 };

let x_val = point.x;  // 10
let y_val = point.y;  // 20
```

### 6.4 Struct Functions

```rust
struct Circle {
    radius: int
}

fn area(c: Circle) -> int {
    // Approximation: π ≈ 3
    3 * c.radius * c.radius
}

fn main() {
    let circle = Circle { radius: 5 };
    let a = area(circle);  // a = 75
}
```

### 6.5 Nested Structs (Future)

```rust
// Not yet implemented
struct Container {
    point: Point,
    value: int
}
```

---

## 7. Enums

Enums are **sum types** (tagged unions) that represent a value that can be one of several variants.

### 7.1 Defining Enums

**Unit Variants (No Payload):**
```rust
enum Direction {
    North,
    South,
    East,
    West
}
```

**Variants with Payloads:**
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

### 7.2 Creating Enum Instances

Use the `::` operator to access variants:

```rust
let success = Result::Ok(42);
let failure = Result::Error(404);

let value = Option::Some(100);
let nothing = Option::None;
```

### 7.3 Enum Representation

Internally, enums are represented as **tagged unions**:

```llvm
// LLVM IR representation
%Result = type { i8, i32 }
// ^^ { tag (0=Ok, 1=Error), payload }
```

---

## 8. Pattern Matching

Pattern matching provides **exhaustive, type-safe** destructuring of enums.

### 8.1 Basic Match

```rust
fn unwrap_or_zero(opt: Option) -> int {
    match opt {
        Option::Some(value) => { value },
        Option::None => { 0 }
    }
}
```

### 8.2 Match with Payloads

**Extracting Values:**
```rust
enum Result {
    Ok(int),
    Error(int)
}

fn handle_result(res: Result) -> int {
    match res {
        Result::Ok(val) => { val },
        Result::Error(code) => {
            // Handle error
            0 - code
        }
    }
}
```

### 8.3 Exhaustiveness Checking

All variants must be handled:

```rust
// ✅ Valid - All variants covered
match x {
    Option::Some(v) => { v },
    Option::None => { 0 }
}

// ❌ Compile Error - Missing Option::None
match x {
    Option::Some(v) => { v }
}
// Error: Non-exhaustive match: missing variants {'None'}
```

### 8.4 Match as Expression

Match expressions return values:

```rust
let result = match compute() {
    Result::Ok(value) => { value * 2 },
    Result::Error(code) => { 0 }
};
```

### 8.5 Binding Variables

Extract payload into a variable:

```rust
match option {
    Option::Some(x) => {
        // 'x' is bound to the payload
        let doubled = x * 2;
        doubled
    },
    Option::None => { 0 }
}
```

---

## 9. Using the Compiler

### 9.1 Command-Line Interface

```bash
forgec compile <source_file>
```

**Options:**
```bash
forgec compile input.forge         # Basic compilation
forgec compile input.forge -v      # Verbose output (future)
forgec --help                      # Show help
```

### 9.2 Compilation Process

```
input.forge
    ↓
[Lexer] → Tokens
    ↓
[Parser] → AST
    ↓
[Semantic] → Type-checked AST
    ↓
[IR Gen] → LLVM IR
    ↓
output.ll
```

### 9.3 Output Files

**LLVM IR (`.ll`):**
```bash
forgec compile program.forge
# Creates: program.ll
```

**Reading LLVM IR:**
```bash
cat program.ll
```

**Compiling to Binary (Future):**
```bash
# Not yet implemented
# Will use LLVM tools (llc, clang)
llc program.ll -o program.s       # Assembly
clang program.s -o program        # Executable
```

### 9.4 Error Messages

Forge provides detailed error messages:

```rust
// Error example
fn main() {
    let x = 10;
    let y = x + true;  // Type error
}
```

**Compiler Output:**
```
ERROR: Type mismatch in binary operation '+':
  Left: int
  Right: bool
  at line 3, column 13

    let y = x + true;
                ^^^^
```

---

## 10. Examples

### 10.1 Hello World (Numeric)

```rust
// hello.forge
fn main() {
    42
}
```

```bash
forgec compile hello.forge
```

### 10.2 Fibonacci

```rust
// fibonacci.forge
fn fib(n: int) -> int {
    if n < 2 {
        n
    } else {
        fib(n - 1) + fib(n - 2)
    }
}

fn main() {
    let result = fib(10);
    result
}
```

### 10.3 Point Distance

```rust
// geometry.forge
struct Point {
    x: int,
    y: int
}

fn distance_squared(p1: Point, p2: Point) -> int {
    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;
    dx * dx + dy * dy
}

fn main() {
    let origin = Point { x: 0, y: 0 };
    let point = Point { x: 3, y: 4 };
    let dist_sq = distance_squared(origin, point);
    dist_sq  // Returns 25
}
```

### 10.4 Option Type

```rust
// option.forge
enum Option {
    Some(int),
    None
}

fn safe_divide(a: int, b: int) -> Option {
    if b == 0 {
        Option::None
    } else {
        Option::Some(a / b)
    }
}

fn unwrap_or(opt: Option, default: int) -> int {
    match opt {
        Option::Some(value) => { value },
        Option::None => { default }
    }
}

fn main() {
    let result1 = safe_divide(10, 2);
    let result2 = safe_divide(10, 0);
    
    let val1 = unwrap_or(result1, -1);  // val1 = 5
    let val2 = unwrap_or(result2, -1);  // val2 = -1
    
    val1 + val2  // Returns 4
}
```

### 10.5 Result Type

```rust
// result.forge
enum Result {
    Ok(int),
    Error(int)
}

fn checked_subtract(a: int, b: int) -> Result {
    if a < b {
        Result::Error(1)  // Error code
    } else {
        Result::Ok(a - b)
    }
}

fn main() {
    let res1 = checked_subtract(10, 5);
    let res2 = checked_subtract(5, 10);
    
    let val = match res1 {
        Result::Ok(v) => { v },
        Result::Error(code) => { 0 }
    };
    
    val  // Returns 5
}
```

---

## 11. Troubleshooting

### 11.1 Common Errors

**1. Type Mismatch:**
```rust
let x = 10;
let y = true;
let z = x + y;  // ❌ Error: Cannot add int and bool
```

**Solution:** Ensure types match in operations.

---

**2. Undefined Variable:**
```rust
fn main() {
    let result = undefined_var;  // ❌ Error: Undefined variable
}
```

**Solution:** Declare all variables before use.

---

**3. Non-Exhaustive Match:**
```rust
match option {
    Option::Some(v) => { v }
    // ❌ Error: Missing Option::None
}
```

**Solution:** Handle all enum variants.

---

**4. Missing Field in Struct:**
```rust
struct Point { x: int, y: int }

let p = Point { x: 10 };  // ❌ Error: Missing field 'y'
```

**Solution:** Provide all required fields.

---

**5. Type Annotation Mismatch:**
```rust
let x: int = true;  // ❌ Error: Expected int, got bool
```

**Solution:** Ensure annotation matches initializer type.

---

### 11.2 Installation Issues

**Problem: `llvmlite` installation fails**

**Solution:**
```bash
# Install LLVM development files
sudo apt install llvm-14-dev  # Ubuntu
brew install llvm@14          # macOS

# Set LLVM_CONFIG environment variable
export LLVM_CONFIG=/usr/bin/llvm-config-14
pip install llvmlite
```

---

**Problem: `forgec: command not found`**

**Solution:**
```bash
# Ensure pip install completed
pip install -e .

# Check if in PATH
which forgec

# Run directly
python -m forgec.main compile input.forge
```

---

### 11.3 Debugging Tips

**1. Check LLVM IR Output:**
```bash
forgec compile program.forge
cat program.ll  # View generated IR
```

**2. Verify Syntax:**
- Ensure semicolons after statements (except last expression in block)
- Match braces for blocks, structs, matches
- Use `::` for enum variants

**3. Type Checking:**
- Run compiler to see type errors
- Use explicit type annotations for clarity

---

## 12. Best Practices

### 12.1 Code Style

**Naming Conventions:**
```rust
// Types: PascalCase
struct UserAccount { }
enum RequestStatus { }

// Variables/Functions: snake_case
let user_count = 10;
fn calculate_total() -> int { }
```

**Indentation:**
- Use 4 spaces (not tabs)
- Align braces consistently

```rust
fn example() {
    if condition {
        // 4 spaces
        statement;
    }
}
```

### 12.2 Type Annotations

**When to Use:**
```rust
// ✅ Good: Clear intent
let timeout: int = 5000;

// ✅ Good: Readable
fn parse_number(s: String) -> int { }

// ⚠️ Verbose but OK
let count: int = 0;

// ✅ Preferred: Inference
let count = 0;
```

### 12.3 Pattern Matching

**Prefer Match Over Nested Ifs:**

```rust
// ❌ Verbose
let value = if x == 1 {
    10
} else {
    if x == 2 {
        20
    } else {
        0
    }
};

// ✅ Better (with enums)
enum State { One, Two, Other }

let value = match state {
    State::One => { 10 },
    State::Two => { 20 },
    State::Other => { 0 }
};
```

### 12.4 Error Handling

**Use Result Type:**
```rust
enum Result {
    Ok(int),
    Error(int)
}

fn divide(a: int, b: int) -> Result {
    if b == 0 {
        Result::Error(1)
    } else {
        Result::Ok(a / b)
    }
}
```

### 12.5 Function Design

**Keep Functions Small:**
```rust
// ✅ Good: Single responsibility
fn calculate_area(width: int, height: int) -> int {
    width * height
}

fn validate_dimensions(width: int, height: int) -> bool {
    width > 0 && height > 0
}
```

**Use Descriptive Names:**
```rust
// ❌ Bad
fn calc(x: int, y: int) -> int { }

// ✅ Good
fn calculate_distance(x: int, y: int) -> int { }
```

---

## 13. Advanced Features

### 13.1 Generics

Generics allow you to write flexible, reusable code that works with multiple types.

**Generic Structs:**
```rust
struct Box<T> {
    value: T
}

let int_box = Box<int> { value: 42 };
let bool_box = Box<bool> { value: true };
```

**Generic Functions:**
```rust
fn identity<T>(x: T) -> T {
    x
}

let n = identity<int>(10);
```

**Generic Enums:**
```rust
enum Option<T> {
    Some(T),
    None
}

let opt = Option<int>::Some(100);
```

### 13.2 Traits

Traits define shared behavior that types can implement.

**Defining a Trait:**
```rust
trait Display {
    fn show(self) -> int;
}
```

**Implementing a Trait:**
```rust
struct Point { x: int, y: int }

impl Display for Point {
    fn show(self) -> int {
        self.x + self.y
    }
}
```

**Using Traits:**
```rust
let p = Point { x: 10, y: 20 };
let val = p.show();  // Static dispatch
```

---

## Appendix A: Language Grammar

### A.1 Keywords

```
fn      let     if      else    match   struct  enum
int     bool    true    false   void    trait   impl
for     self    return
```

### A.2 Operators

**Arithmetic:**
```
+  -  *  /
```

**Comparison:**
```
==  !=  <  >
```

**Structural:**
```
::  .  =>  ->  :
```

---

## Appendix B: Future Features

These features are planned but not yet implemented:

- **Modules:** `mod utils; use utils::helper;`
- **Strings:** Native string type
- **Arrays:** `[int; 10]`
- **Loops:** `for`, `while`
- **Mutable Variables:** `mut` keyword
- **Floating Point:** `f32`, `f64`

---

## Support

- **GitHub:** https://github.com/McMonds/forge-compiler
- **Issues:** https://github.com/McMonds/forge-compiler/issues
- **Discussions:** https://github.com/McMonds/forge-compiler/discussions

---

## 14. Generics

Generics allow you to write code that is flexible and reusable across different types.

### 14.1 Generic Structs

```rust
struct Box<T> {
    value: T
}

fn main() {
    let int_box = Box<int> { value: 42 };
    let bool_box = Box<bool> { value: true };
}
```

### 14.2 Generic Enums

```rust
enum Option<T> {
    Some(T),
    None
}

fn main() {
    let x = Option<int>::Some(10);
    let y = Option<bool>::None;
}
```

### 14.3 Generic Functions

```rust
fn identity<T>(x: T) -> T {
    x
}

fn main() {
    let a = identity<int>(5);
    let b = identity<bool>(true);
}
```

---

## 15. Traits

Traits define shared behavior that types can implement.

### 15.1 Defining a Trait

```rust
trait Display {
    fn show(self) -> int;
}
```

### 15.2 Implementing a Trait

```rust
struct Point {
    x: int,
    y: int
}

impl Display for Point {
    fn show(self) -> int {
        self.x + self.y
    }
}
```

### 15.3 Method Calls

```rust
fn main() {
    let p = Point { x: 10, y: 20 };
    let s = p.show(); // Returns 30
}
```

---

## 16. Module System

The module system helps you organize code into multiple files and control visibility.

### 16.1 Declaring Modules

Use the `mod` keyword to declare a submodule. Forge will look for a file named `name.forge` or `name/mod.forge`.

```rust
// main.forge
mod math;

fn main() -> int {
    math::add(1, 2)
}
```

### 16.2 Visibility with `pub`

By default, all items are private to their module. Use `pub` to make them accessible from outside.

```rust
// math.forge
pub fn add(a: int, b: int) -> int {
    a + b
}

fn secret_helper() {
    // Private to math module
}
```

### 16.3 Importing with `use`

Use the `use` keyword to bring items into the current scope.

```rust
mod math;
use math::add;

fn main() -> int {
    add(10, 20)
}
```

---

---

## 17. C FFI

Forge allows you to call functions from C libraries using `extern` blocks.

### 17.1 Declaring External Functions

```rust
extern "C" {
    fn puts(s: string) -> int;
    fn malloc(size: int) -> string;
}
```

### 17.2 Calling External Functions

External functions are called like any other Forge function.

```rust
fn main() {
    puts("Hello from C!");
}
```

---

## 18. Strings

Forge supports heap-allocated string literals.

```rust
fn main() {
    let message = "Hello, Forge!";
}
```

Strings are currently passed as `i8*` pointers to C functions.

---

## 19. Standard Library

Forge is building a standard library in the `std` namespace.

### 19.1 `std::io`

Provides basic input/output functions.

```rust
use std::io::print;

fn main() {
    print("Hello, World!");
}
```

### 19.2 `std::mem`

Provides basic memory management.

```rust
use std::mem::alloc;

fn main() {
    let ptr = alloc(1024);
}
```

---

**Forge User Manual v0.4.0**  
*Last Updated: December 2024*
