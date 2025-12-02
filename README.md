# Forge Compiler (`forgec`)

Forge is a statically-typed programming language designed for correctness and developer tooling. This repository contains the reference compiler, `forgec`, which targets LLVM IR.

## Project Structure

- `src/forgec`: Source code for the compiler (Lexer, Parser, Semantic Analysis, IR Gen).
- `tests`: Unit and integration tests.
- `examples`: Example Forge source files.

## Prerequisites

- **Python 3.10+**
- **LLVM** (Required for `llvmlite`)

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -e .
    ```
    Or using a virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .
    ```

2.  **Run Tests**:
    ```bash
    python run_tests.py
    ```

## Usage

Compile a Forge source file to LLVM IR:

```bash
python -m forgec.main examples/simple.ae
```

This will generate:
- `examples/simple.ll`: The LLVM IR output.
- `examples/simple.json`: Visualization data (if `--visualize` is used).

### Visualization

To generate data for the dashboard:

```bash
python -m forgec.main examples/simple.ae --visualize
```

## Docker Support

To ensure a consistent environment (especially for Windows users), you can run the compiler in a Docker container.

1.  **Build the Image**:
    ```bash
    docker build -t forgec .
    ```

2.  **Run the Compiler**:
    ```bash
    docker run --rm -v $(pwd)/examples:/app/examples forgec examples/simple.ae
    ```
    *Note: We mount the `examples` directory so the compiler can read the file and write the output back to your host machine.*

3.  **Run Tests in Docker**:
    ```bash
    docker run --rm --entrypoint python forgec run_tests.py
    ```


## Project Structure

- `compiler/src/aetherc`: Source code for the compiler (Lexer, Parser, Semantic Analysis, IR Gen).
- `compiler/tests`: Unit and integration tests.
- `compiler/examples`: Example Aether source files.

## Prerequisites

- **Python 3.10+**
- **LLVM** (Required for `llvmlite`)

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -e .
    ```
    Or using a virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .
    ```

2.  **Run Tests**:
    ```bash
    python run_tests.py
    ```

## Usage

Compile an Aether source file to LLVM IR:

```bash
python -m aetherc.main compile examples/simple.ae
```

This will generate:
- `examples/simple.ll`: The LLVM IR output.
- `examples/simple.json`: Visualization data (if `--visualize` is used).

### Visualization

To generate data for the dashboard:

```bash
python -m aetherc.main compile examples/simple.ae --visualize
```

## Docker Support

To ensure a consistent environment (especially for Windows users), you can run the compiler in a Docker container.

1.  **Build the Image**:
    ```bash
    docker build -t aetherc .
    ```

2.  **Run the Compiler**:
    ```bash
    docker run --rm -v $(pwd)/examples:/app/examples aetherc compile examples/simple.ae
    ```
    *Note: We mount the `examples` directory so the compiler can read the file and write the output back to your host machine.*

3.  **Run Tests in Docker**:
    ```bash
    docker run --rm --entrypoint python aetherc run_tests.py
    ```
