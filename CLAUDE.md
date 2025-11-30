# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ToyMath is a symbolic mathematics system implemented as a Jupyter kernel that parses LaTeX mathematical expressions, performs symbolic manipulation and unification using Prolog-style logic, and outputs results as LaTeX. The system combines mathematical notation processing with logic programming capabilities and includes experimental LLM-based pattern matching.

## Core Architecture

**For detailed information about the notation graph system and processing model, see [NOTATION.md](NOTATION.md).**

### Execution Flow

1. **Input** → LaTeX expressions entered in Jupyter or console
2. **Parsing** → `LatexParser.py` (PLY-based yacc parser) + `lexer.py` → creates Symbol/Func representation
3. **Notation** → `notation.py` stores expressions as a directed acyclic graph (DAG)
4. **Processing** → MathProcessor runs **fixed-point iteration loop**:
   - Repeatedly: walks notation graph → applies transformations → builds new graph
   - Uses `Calculator` (extends `Replicator`) to walk and transform
   - Commands from `cmd_*.py` execute during walk
   - Continues until no changes detected (`s_equal` comparison)
   - Ensures convergence to canonical form
5. **Prolog Engine** → `prolog.py` provides unification, pattern matching, and rule-based inference
6. **Output** → `LatexWriter.py` walks notation graph and converts to LaTeX for display

### Key Components

**Kernel Layer:**
- `toymathkernel.py`: IPython kernel implementation (MathKernel class)
- `console.py`: CLI interface for interactive use
- `engine/mathShell.py`: Core shell that orchestrates parsing, processing, and output

**Notation System:** (see [NOTATION.md](NOTATION.md) for details)
- `notation.py`: Symbol and Func classes represent mathematical terms as a graph
- Symbols are nodes with names; Funcs are edges linking symbols to arguments
- Notation object manages the graph via `self.rel` dictionary (symbol → func mapping)
- All processing uses a recursive "walking" pattern that traverses and transforms the graph

**Parser/Writer:**
- `LatexParser.py`: Yacc-based parser converts LaTeX → internal representation
- `LatexWriter.py`: Converts internal representation → LaTeX output
- Both work with Notation to track symbol metadata

**Processing:** (see [NOTATION.md](NOTATION.md) for walking pattern details)
- `replicator.py`: Base class implementing recursive graph walking (visitor pattern)
- `processor.py`: Contains MathProcessor and Calculator (see evaluation model below)
- `prolog.py`: PrologModel provides unification, pattern matching, and goal resolution
- `comparer.py`: UnifyComparer and pattern matching for symbolic expressions
- `replacer.py`: Extends Replicator to apply variable substitutions during walk
- All transformers inherit from Replicator and walk: source notation → new notation

### Evaluation Model: Fixed-Point Iteration

**MathProcessor** orchestrates all expression transformations using a **fixed-point iteration loop**:

```python
while True:
    calculator = Calculator(notation, output_notation, actions, prologModel)
    new_result = calculator(expression)

    if s_equal(new_result, new_notation, old_result, old_notation):
        break  # Fixed point reached

    # Continue: new result becomes input for next iteration
    notation = output_notation
    expression = new_result
    output_notation = Notation()  # Fresh graph for next iteration
```

**Key concepts**:

1. **Fixed-Point Convergence**: Loop continues until expression stops changing
   - Each iteration: walks notation graph, applies rules, builds new graph
   - Terminates when `s_equal()` detects identical structure between iterations
   - Ensures expressions reach canonical/simplified form

2. **Multi-Iteration Processing**: Complex transformations happen incrementally
   - Nested commands like `\frac{\mul!{ab}}{\mul!{cd}}` process over multiple iterations
   - Iteration 1: Outer operation creates nested commands
   - Iteration 2+: Nested commands expand/simplify
   - Continues until all transformations complete

3. **Calculator as Command Executor**:
   - Extends `Replicator` (inherits graph walking)
   - Adds `enter_command()` to execute commands from `cmd_*.py` files
   - Commands registered via `register_actions()` which scans `engine/cmd_*.py`
   - Each command has `exec()` method that performs transformations

4. **Immutability**: Fresh `Notation` created each iteration
   - Source graph never modified
   - Results built in new graph
   - Enables clean transformations and rollback

5. **Termination Guarantee**:
   - Transformations must be **monotonic** (simplify or expand, not oscillate)
   - **Idempotent** operations crucial (e.g., reduction shouldn't expand)
   - Infinite loops indicate implementation bugs

**Example** - `\mul!{(a+b)(c+d)}`:
- Iteration 1: Expands to `ac+ad+bc+bd`
- Iteration 2: Simplifies/canonicalizes terms
- Iteration 3: No changes → **fixed point** → stop

This model ensures all transformations converge to canonical forms suitable for pattern matching and unification.

**LLM Integration:**
- `llm_comparer.py`: LLMComparer uses OpenAI API for semantic pattern matching
- Allows matching expressions that are mathematically equivalent but structurally different
- Used when patterns have `\operatorname{llm}` operator

**Commands:**
- Commands defined in `engine/cmd_*.py` files (add, mul, debug, dump, echo, etc.)
- Loaded dynamically by processor; extend system functionality

## Running Tests

```bash
# Run all unit tests
pytest engine/unittests.py

# Run LLM-based tests (requires OpenAI API key in .env)
pytest engine/unittests_llm.py

# Run specific test
pytest engine/unittests.py::TestScenario::test_pattern1
```

## Development Workflow

### Console Mode
```bash
python console.py
```
Interactive console with colored prompts for testing expressions directly.

### Jupyter Kernel Mode
```bash
jupyter notebook
# Select "LaTex" kernel when creating a new notebook
```
The kernel is registered via `toymathkernel.py` as kernel spec.

### Environment Setup

**Package Manager:** This project uses `uv` for fast, reliable Python package management.

```bash
# Install/update dependencies
uv pip install -r requirements.txt

# Upgrade a specific package
uv pip install --upgrade <package-name>

# Add a new dependency
uv pip install <package-name>
```

Key dependencies: jupyter, ipykernel, ipywidgets, openai, ply, pytest, python-dotenv

For LLM features, create `.env` file with:
```
OPENAI_API_KEY=your_key_here
```

## Code Patterns

### Adding a New Command

Commands implement transformations that execute within the fixed-point iteration loop.

1. Create `engine/cmd_yourname.py`:
```python
from notation import Notation

class YourCommand(object):
    arity = 1  # number of required arguments

    def exec(self, processor, sym, f):
        # f.args[0] = parameters (list or None)
        # f.args[1] = arguments tuple

        # Access processor.notation (input) and processor.output_notation (output)
        # Walk the input graph, build result in output graph

        # Can recursively invoke other commands via chainexpr():
        # from cmd_mul import chainexpr, Mul
        # nested = chainexpr(Mul.MUL, processor.output_notation, expr, None)

        # Return Notation.NONE to suppress output
        # Return a symbol to display result
        pass

def create_actions():
    return {'yourname': YourCommand()}
```

2. The processor automatically loads commands from `engine/cmd_*.py` files via `register_actions()`

3. Use in LaTeX as `\yourname{arg}` or `\yourname[params]{arg}`

**Important convergence requirements**:
- Operations must be **monotonic** (don't oscillate between forms)
- Simplification operations should be **idempotent** (same result if applied again)
- Nested commands OK (they'll be processed in subsequent iterations)
- Each command invocation gets evaluated in the next MathProcessor iteration

### Working with Notation

Symbol creation and lookup:
```python
notation = Notation()
sym = notation.setf(Symbol('x'), args)  # Create symbol with args
f = notation.getf(sym, Notation.FUNC)   # Get function metadata
```

Notation types defined in `notation.py`: FUNC, GROUP, PLUS, MINUS, MUL, DIV, POWER, INDEX, SETQ, etc.

### Prolog-Style Rules

The system uses Prolog-like unification for pattern matching. See `prolog.py`:
- `Term` represents predicates with arguments
- `Rule` represents head :- body relationships
- `PrologModel.unify()` performs unification with substitutions
- Goals can include callbacks for dynamic behavior

### Fraction Operations

ToyMath handles fraction operations through a **dual-path system** that processes numeric and symbolic fractions differently:

#### Numeric Fractions (e.g., `\frac{1}{2}`, `\frac{2}{3}`)

**Processing Path:**
1. **Parsing** → `LatexParser` creates `Symbol` with `\frac` structure
2. **Preprocessing** → `Preprocessor.enter_oper()` (preprocessor.py:43-46) converts to `FracValue`
   - Only for fractions where both numerator and denominator are `IntegerValue`
   - Example: `\frac{1}{2}` → `FracValue(1, 2)`
3. **Arithmetic** → `FracValue` class (value.py:198-295) handles operations:
   - `FracValue.add()` (lines 260-264): Fraction addition using cross-multiplication
   - `FracValue.mul()` (lines 272-274): Fraction multiplication
   - Auto-reduces via GCD in constructor (lines 211-213)
4. **Output** → `LaTexWriter` renders `FracValue` back as `\frac{num}{denom}` LaTeX

**Key Properties:**
- `FracValue` objects are NOT recognized by `is_frac()` from `frac_utils.py`
- Arithmetic operations return new `FracValue` objects
- GCD reduction happens automatically during construction
- Numeric fractions are fully evaluated during preprocessing/processing

#### Symbolic Fractions (e.g., `\frac{a}{b}`, `\frac{x}{y}`)

**Processing Path:**
1. **Parsing** → Remain as `Symbol` objects with `\frac` structure
2. **Preprocessing** → No conversion (stay as symbols)
3. **Operations** → Handled by `cmd_add.py` and `cmd_mul.py`:
   - **Multiplication Rules (cmd_mul.py:80-161)**:
     - Rule 2: `\frac{x1}{y1} · \frac{x2}{y2}` → nested `\mul!` commands
     - Rule 3: `a · \frac{x}{y}` → numerator multiplication
     - Rule 4: `\frac{x}{y} · (sum)` → numerator multiplication
   - **Addition Rules (cmd_add.py:86-214)**:
     - Rule 1: `\frac{x1}{y1} + \frac{x2}{y2}` → cross-multiplication with nested `\add!`/`\mul!`
     - Rule 5: `a + \frac{x}{y}` → common denominator with nested commands
4. **Fixed-Point Iteration** → Nested commands expand over multiple iterations
5. **Output** → Final symbolic fraction expression

**Key Properties:**
- Detected by `is_frac()` which checks `f.sym.name in FRAC_SYMBOL_NAMES`
- Generate nested `\add!`/`\mul!` commands that evaluate in subsequent iterations
- Do NOT auto-reduce (symbolic expressions remain expanded)
- Follow monotonic/idempotent rules for convergence

#### Shared Utilities (`frac_utils.py`)

**Module Purpose:** Common fraction detection and manipulation for symbolic fractions

**Key Functions:**
- `FRAC_SYMBOL_NAMES`: Set of all fraction operator names (`\frac`, `\dfrac`, `\tfrac`, `\cfrac`)
- `is_frac(notation, sym)`: Detects if symbol is a fraction operator
  - **IMPORTANT**: Returns `False` for `FracValue` objects (numeric fractions)
  - Only recognizes `Symbol` objects with fraction operator names
- `get_numerator(notation, sym)`: Extracts numerator, unwrapping GROUP
- `get_denominator(notation, sym)`: Extracts denominator, unwrapping GROUP
- `normalize_frac(notation, num, denom, chainexpr_fn, mul_symbol)`: Creates canonical fraction
  - Identity simplification: `\frac{x}{1}` → `x`
  - Sign normalization: Negative always in numerator
  - Nested fraction flattening: `\frac{\frac{a}{b}}{c}` → `\frac{a}{\mul!{bc}}`
  - Always uses `\frac` (canonical style)

#### Mixed Expressions

**Numeric Fraction × Variable** (e.g., `\frac{2}{3}x`):
- Preprocessor converts `\frac{2}{3}` → `FracValue(2, 3)`
- Parser creates P_LIST: `[FracValue(2,3), Symbol('x')]`
- Works correctly in sums: `\frac{2}{3}x + \frac{1}{5}x` → `\frac{13}{15}x`

**Parsing Syntax Important Notes:**
- `\frac 2 3 x` (spaces, no braces) → `(\frac{2}{3}) * x` ✓ CORRECT
- `\frac{2}{3}x` (braces, no space) → Can parse as `\frac{2}{3x}` ✗ AMBIGUOUS
- Always use spaces between fraction and variables for clarity

#### Implementation Files

**Fraction Multiplication (Rules 2-4):**
- `engine/cmd_mul.py`: `multiply_fractions()` method (lines 80-161)
- Handles: frac×frac, scalar×frac, frac×sum

**Fraction Addition (Rules 1, 5):**
- `engine/cmd_add.py`: `add_fractions()` method (lines 86-214)
- Handles: frac+frac, scalar+frac
- Processes terms pairwise in S_LIST

**Shared Utilities:**
- `engine/frac_utils.py`: Detection, extraction, normalization functions

**Value Arithmetic:**
- `engine/value.py`: `FracValue` class with arithmetic operations

**Preprocessing:**
- `engine/preprocessor.py`: Numeric fraction → FracValue conversion (lines 43-46)

#### Testing Fraction Operations

```python
# Test numeric fractions (auto-evaluated)
"\frac 1 2 + \frac 1 3"  # → \frac{5}{6}

# Test symbolic fractions (symbolic manipulation)
"\frac a b + \frac c d"  # → \frac{\add!{\mul!{ad} + \mul!{bc}}}{\mul!{bd}}

# Test mixed (numeric frac with variable)
"\frac 2 3 x + \frac 1 5 x"  # → \frac{13}{15}x

# Run fraction tests
pytest engine/unittests.py::TestScenario::test_frac1
```

## File Organization

- `engine/`: Core math processing engine
  - `cmd_*.py`: Command implementations
  - `*Parser.py`, `*Writer.py`: Input/output
  - `prolog.py`, `comparer.py`: Logic engine
  - `unittests*.py`: Test suites
- `examples/`: Jupyter notebooks demonstrating features
- `bin/`: Virtual environment (not in git)
- `obsolete/`: Old/deprecated code

## Development Plans

- `FRACTION_OPERATIONS_PLAN.md`: Detailed plan for implementing fraction support in `add!` and `mul!` commands

## Important Notes

- **Notation Architecture**: See [NOTATION.md](NOTATION.md) for graph structure and walking pattern
- **Evaluation Model**: MathProcessor uses fixed-point iteration; transformations must converge
  - Operations should be **monotonic** (simplify or expand consistently, not oscillate)
  - Reduction operations must be **idempotent** (applying twice = applying once)
  - Nested commands processed across multiple iterations
  - Fresh notation graph created each iteration (immutability)
- Parser generates `parser.out` and `parsetab.py` from `LatexParser.py` (cached, safe to ignore)
- Symbols use name-based equality; `Symbol('x') == Symbol('x')`
- Notation objects are context-specific; use Replicator to copy expressions between notations
- All transformations follow immutable pattern: walk source notation → build new notation
- Commands in `cmd_*.py` auto-discovered via `register_actions()` on startup
- The `!` operator implements cut (like Prolog cut)
- LLM features are experimental and require API access
