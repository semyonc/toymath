# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

ToyMath is a symbolic mathematics system implemented as a Jupyter kernel that parses LaTeX expressions, performs symbolic manipulation using Prolog-style logic, and outputs results as LaTeX.

## Core Architecture

See [NOTATION.md](NOTATION.md) for detailed notation graph system documentation.

### Execution Flow

1. **Input** → LaTeX expressions in Jupyter or console
2. **Parsing** → `LatexParser.py` → Symbol/Func representation
3. **Notation** → `notation.py` stores expressions as DAG
4. **Processing** → MathProcessor runs **fixed-point iteration**:
   - Walks notation graph → applies transformations → builds new graph
   - Commands from `cmd_*.py` execute during walk
   - Continues until no changes (`s_equal` comparison)
5. **Output** → `LatexWriter.py` converts to LaTeX

### Key Files

| File | Purpose |
|------|---------|
| `processor.py` | MathProcessor, Calculator - fixed-point iteration engine |
| `notation.py` | Symbol, Func, Notation - DAG representation |
| `replicator.py` | Base class for graph walking (visitor pattern) |
| `cmd_mul.py` | Multiplication command with fraction/power support |
| `cmd_add.py` | Addition command with fraction support |
| `frac_utils.py` | Shared fraction utilities |
| `value.py` | IntegerValue, FracValue, FloatValue |
| `comparer.py` | Pattern matching, unification |
| `prolog.py` | Unification/pattern rules (logic layer) |
| `agent_do.py` | `do!` Jupyter endpoint: OpenRouter agent driving the primitives |
| `lexer.py` | Tokenizer used by LatexParser |

### Fixed-Point Iteration Model

```python
while True:
    calculator = Calculator(notation, output_notation, actions, model)
    result = calculator(expression)
    if s_equal(result, output_notation, expression, notation):
        break  # Converged
    notation, expression = output_notation, result
    output_notation = Notation()
```

**Key principles:**
- Transformations must be **monotonic** (no oscillation)
- Operations must be **idempotent** (applying twice = applying once)
- Nested commands process across multiple iterations
- Fresh notation graph each iteration (immutability)
- **Never mutate input notation**; always build results into `processor.output_notation`

## Running the Project

```bash
# Activate venv first (or set PYTHONPATH=. to avoid import errors)
source .venv/bin/activate

python console.py                    # Console mode
jupyter notebook                     # Kernel mode (pick the LaTeX kernel)

# Install dependencies
uv pip install -r requirements.txt
```

## Running Tests

```bash
pytest engine/unittests.py           # Core tests (40)
pytest engine/unittests_frac.py      # Fraction tests (50)
pytest engine/unittests_primitives.py  # Verified-derivation primitives
pytest engine/unittests_do.py        # do! endpoint (offline scripted agent)
TOYMATH_LIVE_TESTS=1 pytest engine/unittests_do.py  # + live OpenRouter test
```

## Adding Commands

Create `engine/cmd_yourname.py`:

```python
class YourCommand(object):
    arity = 1

    def exec(self, processor, sym, f):
        # f.args[1] = arguments tuple
        # Use processor.output_notation to build result
        # Use chainexpr() to create nested commands
        pass

def create_actions():
    return {'yourname': YourCommand()}
```

## Notation Structures

| Type | Structure | Example |
|------|-----------|---------|
| INDEX | `(base, (sub, sup_l, power, sup_r))` | `x^2` → `(x, (None, None, 2, None))` |
| P_LIST | `(factor1, factor2, ...)` | `xy` → `(x, y)` |
| S_LIST | `(term1, +term2, ...)` | `x+y` → `(x, +y)` |
| GROUP | `(inner,)` with `br` prop | `{x}` → `(x,)` br="{}" |
| FUNC | `(name, args)` | `\frac{a}{b}` |

## Fraction Operations

**Dual-path system:**
- **Numeric** (`\frac{1}{2}`) → Preprocessor converts to `FracValue` → arithmetic evaluated immediately
- **Symbolic** (`\frac{a}{b}`) → Stays as Symbol → `cmd_mul.py`/`cmd_add.py` handle via rules

**Key utilities (`frac_utils.py`):**
- `is_frac(notation, sym)` - detects symbolic fractions (NOT FracValue)
- `get_numerator/get_denominator` - extract parts
- `normalize_frac` - canonicalize: `\frac{x}{1}→x`, sign normalization, nested flattening

**Multiplication rules (`cmd_mul.py`):**
- Rule 2: `\frac{a}{b} × \frac{c}{d}` → nested `\mul!` commands
- Rule 3: `scalar × \frac{a}{b}` → numerator multiplication
- Rule 4: `\frac{a}{b} × (sum)` → distribution

**Addition rules (`cmd_add.py`):**
- Rule 1: `\frac{a}{b} + \frac{c}{d}` → cross-multiplication
- Rule 5: `scalar + \frac{a}{b}` → common denominator

## Power Operations

**Implemented in `cmd_mul.py`:**

**Fraction base powers:**
```
(\frac{a}{b})^n → \frac{a^n}{b^n}     (n > 0)
(\frac{a}{b})^n → \frac{b^|n|}{a^|n|} (n < 0)
(\frac{a}{b})^0 → 1
```

**Symbolic negative powers:**
```
x^{-n} → \frac{1}{x^n}
```

**Implementation pattern:**
1. `detect_power()` - Direct INDEX detection (no patterns for negative exponents)
2. `unwrap_base()` - Peel GROUP/MINUS wrappers
3. `power_fraction()` - Expand fraction powers with idempotence check
4. `negative_power()` - Convert to fraction form

**Why direct detection:** `NotationParam.N` only matches positive integers. Use direct INDEX notation inspection for negative/zero exponents.

## Development Patterns

### Unwrapping Wrappers
Expressions often wrapped in GROUP/MINUS. Always unwrap before type checks:
```python
def unwrap(notation, sym):
    f = notation.getf(sym, Notation.GROUP)
    if f: sym = f.args[0]
    f = notation.getf(sym, Notation.MINUS)
    if f: sym = f.args[0]
    return sym
```

### Creating Nested Commands
Use `chainexpr()` to defer evaluation to next iteration:
```python
from cmd_mul import chainexpr, Mul
nested = chainexpr(Mul.MUL, notation, plist, None)
```

### Idempotence Checks
Prevent re-expansion of already-processed expressions:
```python
# Skip if numerator/denominator already have powers
if notation.getf(x, Notation.INDEX) or notation.getf(y, Notation.INDEX):
    return None
```

### Getting Values
Use `get_value()` from processor.py to extract numeric values through wrappers:
```python
from processor import get_value
n = get_value(power, notation)
if isinstance(n, IntegerValue): ...
```

## File Organization

```
engine/
  cmd_*.py        - Command implementations
  *Parser.py      - Input parsing
  *Writer.py      - Output generation
  processor.py    - Core evaluation engine
  notation.py     - DAG data structures
  frac_utils.py   - Fraction utilities
  value.py        - Numeric value types
  unittests*.py   - Test suites
```

## Important Notes

- Symbols use name-based equality: `Symbol('x') == Symbol('x')`
- Use Replicator to copy expressions between notation contexts
- Parser generates `parser.out`/`parsetab.py` (cached, ignore)
- Commands auto-discovered via `register_actions()` on startup
- The `do!` agent endpoint requires the `OPEN_ROUTER` key in `.env`
  (model via `OPENROUTER_MODEL`, default `anthropic/claude-sonnet-5`)
