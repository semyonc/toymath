# ToyMath

A symbolic mathematics system implemented as a Jupyter kernel that parses LaTeX expressions, performs symbolic manipulation using Prolog-style logic, and outputs results as LaTeX.

## Features

- **LaTeX I/O**: Input and output mathematical expressions in LaTeX format
- **Symbolic Manipulation**: Fixed-point iteration engine for expression transformations
- **Processing Commands**: Explicit control over evaluation with `mul!`, `add!`, and more
- **Prolog-style Logic**: Pattern matching and unification for symbolic operations
- **Jupyter Integration**: Full Jupyter kernel implementation
- **Fraction Operations**: Complete support for symbolic and numeric fractions
- **Power Operations**: Negative powers, fraction powers, and power cancellation

## Architecture

ToyMath uses a **notation graph** (DAG) to represent mathematical expressions and a **fixed-point iteration** model for evaluation:

```
Input (LaTeX) → Parser → Notation Graph → Processor → Output (LaTeX)
```

Key components:
- **`notation.py`**: DAG representation (Symbol, Func, Notation)
- **`processor.py`**: Fixed-point iteration engine (MathProcessor, Calculator)
- **`cmd_*.py`**: Transformation commands (mul!, add!, etc.)
- **`LatexParser.py`**: LaTeX → notation parsing
- **`LatexWriter.py`**: Notation → LaTeX output

## Installation

### Requirements
- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) - Fast Python package manager
- Jupyter Lab >= 4.2

### Setup

1. Clone the repository:
```bash
git clone https://github.com/semyonc/toymath.git
cd toymath
```

2. Create virtual environment and install dependencies:
```bash
# Create venv with uv
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies with uv
uv pip install -e .

# Or install from requirements.txt
uv pip install -r requirements.txt
```

4. Install as Jupyter kernel:
```bash
jupyter kernelspec install kernel_spec --user
```

## Quick Start (with uv)

```bash
# Clone and setup
git clone https://github.com/semyonc/toymath.git && cd toymath

# Install with uv (handles venv creation and dependencies)
uv sync

# Or manually
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt

# Run tests
uv run pytest engine/

# Launch Jupyter
uv run jupyter lab
```

## Usage

### Jupyter Notebook

Launch Jupyter Lab and select the ToyMath kernel:
```bash
uv run jupyter lab
```

Example notebook operations:
```latex
\frac{1}{2} + \frac{1}{3}  % → 5/6 (automatic numeric evaluation)
mul! x^{-1}x               % → 1 (explicit multiplication with power cancellation)
mul! (a/b)^2               % → a²/b² (explicit power expansion)
add! \frac{a}{b}+\frac{c}{d} % → \frac{ad+bc}{bd} (explicit fraction addition)
```

> **Note**: Commands ending with `!` (like `mul!` and `add!`) provide explicit control over evaluation.
> See [Processing Commands](#processing-commands) for details.

### Console Mode

```bash
uv run python console.py
```

## Processing Commands

ToyMath uses **explicit processing commands** (marked with `!`) to control when and how expressions are evaluated. This design choice provides fine-grained control over the evaluation process and makes the evaluation model transparent.

### Why Processing Commands?

Unlike traditional computer algebra systems that automatically evaluate expressions, ToyMath uses an **explicit evaluation model**:

1. **Transparency**: You see exactly when and how transformations occur
2. **Control**: Choose which operations to expand and which to leave symbolic
3. **Debugging**: Step through transformations to understand the evaluation process
4. **Composability**: Chain commands together for complex transformations

### Core Processing Commands

| Command | Purpose | Example | Result |
|---------|---------|---------|--------|
| `mul!` | Multiply expressions and expand products | `mul! x^{-1}x` | `1` |
| `add!` | Add expressions and combine like terms | `add! \frac{a}{b} + \frac{c}{d}` | `\frac{ad+bc}{bd}` |

### Arithmetic Command Details

#### `mul!` - Multiplication Command

**Purpose**: Explicitly evaluate multiplication operations including:
- Power expansion: `(a)^n` → `a·a·...·a` (n times)
- Fraction multiplication: `\frac{a}{b} × \frac{c}{d}` → `\frac{ac}{bd}`
- Power cancellation: `x^{-1} × x` → `1`
- Distribution: `a(b+c)` → `ab + ac`

**Examples**:
```latex
mul! xy                    % → xy (expands product)
mul! x^3                   % → xxx
mul! (a/b)(c/d)           % → \frac{ac}{bd}
mul! x^{-2}x^2            % → 1
mul! \frac{a}{b}(x+y)     % → \frac{ax+ay}{b}
```

**Why explicit?** Automatic expansion can lead to unwieldy expressions. With `mul!`, you control when products are expanded.

#### `add!` - Addition Command

**Purpose**: Explicitly evaluate addition operations including:
- Fraction addition: `\frac{a}{b} + \frac{c}{d}` → `\frac{ad+bc}{bd}`
- Like term collection: `2x + 3x` → `5x`
- Simplification: `a + 0` → `a`

**Examples**:
```latex
add! \frac{1}{2} + \frac{1}{3}  % → \frac{5}{6}
add! a + \frac{x}{y}            % → \frac{ay+x}{y}
add! 2x + 3x                    % → 5x
```

**Why explicit?** You can choose to keep sums unexpanded (e.g., `a+b`) or combine them into a single fraction.

### Utility Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `goal` | Execute Prolog-style goals | `goal solve(x, a+b=c)` |
| `rules` | Display active transformation rules | `rules` |
| `clear` | Clear state/context | `clear` |
| `closure` | Compute transitive closure | `closure R` |
| `dump` | Debug: dump notation graph | `dump expr` |
| `track` | Enable/disable execution tracing | `track` |
| `echo-on` | Enable echo mode | `echo-on` |
| `echo-off` | Disable echo mode | `echo-off` |
| `debug` | Debug mode control | `debug` |

### Command Composition

Commands can be nested to build complex transformations:

```latex
% Expand then add fractions
add! {mul! (a+b)(c+d)} + x

% Multiple operations
mul! (add! x+y)^2
```

### Automatic vs. Explicit Evaluation

**Automatic** (no command):
- Numeric constants are evaluated: `\frac{1}{2} + \frac{1}{3}` → `\frac{5}{6}`
- Variables remain symbolic: `x + y` → `x + y`

**Explicit** (with command):
- `mul!` / `add!` force evaluation and transformations
- Gives control over expansion depth
- Enables step-by-step simplification

### Fixed-Point Iteration

Commands execute until reaching a **fixed point** (no more changes):

```latex
mul! x^3  →  {mul! xx}x  →  {mul! x}xx  →  xxx  (fixed point)
```

Each iteration applies transformations once; nested commands trigger new iterations.

## Running Tests

```bash
# Core tests (40 tests)
uv run pytest engine/unittests.py

# Fraction operations tests (53 tests)
uv run pytest engine/unittests_frac.py

# All tests
uv run pytest engine/
```

## Project Structure

```
toymath/
├── engine/           # Core engine
│   ├── cmd_*.py     # Transformation commands
│   ├── notation.py  # Notation graph structures
│   ├── processor.py # Evaluation engine
│   ├── comparer.py  # Pattern matching
│   └── ...
├── examples/        # Example notebooks
├── obsolete/        # Deprecated code
└── pyproject.toml   # Project configuration
```

## Documentation

- [CLAUDE.md](CLAUDE.md) - Developer guide and architecture
- [NOTATION.md](NOTATION.md) - Notation graph system details

## Development

### Adding Commands

Create `engine/cmd_yourname.py`:

```python
class YourCommand(object):
    arity = 1

    def exec(self, processor, sym, f):
        # Transform expression
        pass

def create_actions():
    return {'yourname': YourCommand()}
```

### Code Style

The project uses:
- **Black** for code formatting
- **isort** for import sorting
- **Ruff** for linting
- **pytest** for testing

Development tools with uv:
```bash
# Format code
uv run black engine/

# Sort imports
uv run isort engine/

# Lint with ruff
uv run ruff check engine/

# Type checking (if mypy is installed)
uv run mypy engine/
```

### Updating Dependencies

```bash
# Update all packages to latest versions
uv pip install --upgrade -r requirements.txt

# Freeze current environment
uv pip list --format=freeze > requirements.txt
```

## License

MIT License

## Author

Semyon C
