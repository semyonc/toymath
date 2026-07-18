# Notation: Mathematical Expression Graph Representation

## Overview

The `Notation` class in `engine/notation.py` represents mathematical expressions as a **directed acyclic graph (DAG)** where:
- **Nodes** are symbols (instances of `Symbol` class)
- **Edges** are function relationships (instances of `Func` class) stored in the `self.rel` dictionary
- Each symbol serves as a unique key that can be referenced by other expressions

This graph-based representation allows complex mathematical expressions to be broken down into reusable sub-expressions, avoiding duplication and enabling efficient symbolic manipulation.

## Core Components

### 1. Symbol Class
```python
class Symbol(object):
    def __init__(self, name=None, **kwargs):
        # Auto-generated names like '_n1', '_n2' if name is None
        # Named symbols like 'x', 'y', '+', 'func', etc.
```

**Characteristics:**
- Symbols are identified by their `name` (string)
- Equality is name-based: `Symbol('x') == Symbol('x')`
- Auto-generated symbols get names like `_n1`, `_n2`, etc.
- Symbols can have properties stored in `self.props`

### 2. Func Class
```python
class Func(object):
    def __init__(self, sym, args, **kwargs):
        self.sym = sym      # Function symbol (e.g., Symbol('+'), Symbol('frac'))
        self.args = args    # Arguments (tuple or list)
        self.props = kwargs # Additional properties
```

**Characteristics:**
- Represents a function/operation applied to arguments
- Arguments can be:
  - **Tuple** `()`: Positional arguments (rendered with parentheses)
  - **List** `[]`: Named/bracketed arguments (rendered with brackets)
- Arguments can be symbols, values, or nested structures

### 3. Notation Class
```python
class Notation(object):
    def __init__(self):
        self.rel = defaultdict()  # Symbol → Func mapping
```

**Key Methods:**
- `setf(f, args, **kwargs)`: Create a new auto-generated symbol with function `f` and arguments
- `repf(sym, func)`: Associate a specific symbol with a function
- `getf(sym, f)`: Retrieve function `f` for symbol `sym` if it exists
- `get(sym)`: Get any function associated with symbol

## Graph Structure Examples

### Example 1: `x + 2y`

**Input:** `x + 2y`

**Parsed Symbol:** `_n3`

**Graph Representation:**
```
_n1: p-list [2, y]      # Represents "2y" (implicit multiplication as product list)
_n2: + (_n1)            # Represents "+ 2y" (unary plus with _n1 as argument)
_n3: s-list [x, _n2]    # Represents "x + 2y" (sum list with x and _n2)
```

**Explanation:**
- `_n1` represents `2y` as a product list (`p-list`) containing the number 2 and symbol y
- `_n2` wraps `_n1` with a `+` operator (representing the additive term)
- `_n3` is the root, representing a sum list (`s-list`) containing `x` and `_n2`

### Example 2: `2x`

**Input:** `2x`

**Parsed Symbol:** `_n1`

**Graph Representation:**
```
_n1: p-list [2, x]      # Represents "2x" (product list)
```

**Explanation:**
- Simple multiplication is represented as a product list (`p-list`)
- No intermediate nodes needed

### Example 3: `\frac{a}{b}`

**Input:** `\frac{a}{b}`

**Parsed Symbol:** `_n4`

**Graph Representation:**
```
_n2: group {"br": "{}"} (a)     # Numerator: 'a' grouped with braces
_n3: group {"br": "{}"} (b)     # Denominator: 'b' grouped with braces
_n4: \frac (_n2, _n3)           # Fraction operation with two arguments
```

**Explanation:**
- `_n2` and `_n3` represent grouped arguments (LaTeX braces `{}`)
- `_n4` represents the `\frac` operation with numerator and denominator as arguments
- Props like `{"br": "{}"}` track bracket types

### Example 4: `f(x)`

**Input:** `f(x)`

**Parsed Symbol:** `_n6`

**Graph Representation:**
```
_n5: group {"br": "()"} (x)     # Argument 'x' grouped with parentheses
_n6: p-list [f, _n5]            # Function application as product list
```

**Explanation:**
- `_n5` represents `x` grouped with parentheses
- `_n6` represents function application as a product of `f` and its argument group

### Example 5: Matrix environments

Standard non-alignment matrix environments normalize before grammar dispatch.
For example, `\begin{bmatrix}a & b \\ c & d\end{bmatrix}` is stored as:

```text
_n1: \bmatrix [[a, b], [c, d]]
```

Rows and columns are nested lists, while the function-symbol name preserves
the environment kind. `bmatrix`, `Bmatrix`, `vmatrix`, `Vmatrix`, and
`smallmatrix` write back as AMS environments because MathJax does not expose
equivalent plain-TeX commands. `matrix` and `pmatrix` retain ToyMath's existing
plain-command canonical output. `cases` uses the same row/column shape but is
piecewise scalar syntax, not a matrix-valued object.

The shared parser normalization deliberately excludes `array` and starred
matrix variants: their alignment preambles require explicit notation metadata
to round-trip without loss.

## Built-in Symbol Types

The `Notation` class defines several predefined symbols for common operations:

| Symbol | Name | Purpose |
|--------|------|---------|
| `PLUS` | `+` | Addition |
| `MINUS` | `-` | Subtraction |
| `GROUP` | `group` | Grouping with brackets |
| `FUNC` | `func` | Function application |
| `INDEX` | `index` | Indexing/subscript |
| `FACTORIAL` | `factorial` | Postfix factorial `(operand,)` |
| `BINOM` | `\binom` | Binomial coefficient `(n, k)` |
| `P_LIST` | `p-list` | Product list (multiplication) |
| `S_LIST` | `s-list` | Sum list (addition) |
| `C_LIST` | `c-list` | Comma-separated list (arguments, parameters, or relation systems) |
| `A_LIST` | `a-list` | Logical AND list (conjunction) |
| `O_LIST` | `o-list` | Logical OR list (disjunction) |
| `SETQ` | `setq` | Assignment |
| `COMP` | `comp` | Comparison |
| `FRAC` | `\frac` | Fraction (division with numerator/denominator) |

## Graph Properties

### Advantages

1. **No Duplication**: Sub-expressions are shared via symbol references
2. **Efficient Storage**: Complex expressions reuse common sub-trees
3. **Easy Traversal**: Graph structure enables pattern matching and transformation
4. **Notation Isolation**: Each `Notation` instance maintains its own symbol table

### Symbol References

Symbols can reference other symbols through their arguments:
```python
# If we have:
_n1: func (a, b)
_n2: + (_n1, c)

# Then _n2 references _n1, which references a, b, and c
# This forms a graph: _n2 → _n1 → {a, b}, _n2 → c
```

## Working with Notation

### Creating Expressions Programmatically

```python
from notation import Notation, Symbol

notation = Notation()

# Create symbols
x = Symbol('x')
y = Symbol('y')

# Create 2y as a product list
two_y = notation.setf(Notation.P_LIST, [2, y])

# Create x + 2y as a sum
sum_expr = notation.setf(Notation.S_LIST, [x, two_y])

print(notation)
```

### Accessing Graph Structure

```python
# Get function associated with a symbol
func = notation.get(sum_expr)
if func:
    print(f"Function: {func.sym}")
    print(f"Arguments: {func.args}")

# Find all instances of a specific function type
for sym, func in notation.select(Notation.PLUS):
    print(f"Found addition: {sym}")
```

### Notation Scope

Each `Notation` instance is independent:
- Symbols in one notation don't exist in another
- To move expressions between notations, use replicators (see `replicator.py`)
- This isolation enables safe transformation and manipulation

## The Walking Pattern: Graph Traversal and Transformation

All processing in ToyMath follows a **recursive walking pattern** that traverses the notation graph. Each transformation creates a new notation graph rather than modifying the original, following functional programming principles.

### How Walking Works

The base class `Replicator` (`engine/replicator.py`) implements a complete graph traversal using the **visitor pattern**:

```python
class Replicator(object):
    def __init__(self, notation, output_notation):
        self.notation = notation              # Source graph
        self.output_notation = output_notation  # Target graph
```

**Key characteristics:**
1. Walks the source notation graph recursively
2. Builds a new graph in output_notation as it walks
3. Maintains a call stack to track context during traversal
4. Follows the mathematical expression grammar hierarchy

### Traversal Hierarchy

The walker descends through expression types in this order:

```
enter_formula
  ↓
enter_or_expr_list        (logical OR: ∨)
  ↓
enter_and_expr_list       (logical AND: ∧)
  ↓
enter_not_expr            (negation: ¬)
  ↓
enter_subformula          (comparisons and comma-separated relation systems)
  ↓
enter_comma_list          (comma-separated relations, arguments, parameters)
  ↓
enter_additive_expr_list  (sum lists: a + b + c)
  ↓
enter_additive_expr       (individual +/- terms)
  ↓
enter_composite_expr      (product lists, division, multiplication)
  ↓
enter_expr                (operators, indices, limits)
  ↓
enter_scalar              (groups, functions, arrays)
  ↓
enter_term / enter_symbol (leaf nodes: variables, numbers)
```

Each level checks if the current symbol matches its expected type, then either:
- **Processes it**: Calls the corresponding handler method
- **Passes down**: Continues to the next level

### Example: Walking `x + 2y`

For the expression `x + 2y` with graph:
```
_n1: p-list [2, y]
_n2: + (_n1)
_n3: s-list [x, _n2]
```

The walk proceeds:
1. `enter_formula(_n3)` → No special operators, continue
2. `enter_or_expr_list(_n3)` → Not an OR list, continue
3. `enter_and_expr_list(_n3)` → Not an AND list, continue
4. `enter_not_expr(_n3)` → Not a negation, continue
5. `enter_subformula(_n3)` → Not a comparison, continue
6. `enter_comma_list(_n3)` → Not a comma list, continue
7. `enter_additive_expr_list(_n3)` → **Match!** It's an `s-list`
   - Walks first argument `x` → reaches `enter_symbol(x)`
   - Walks second argument `_n2`:
     - `enter_additive_expr(_n2)` → **Match!** It's a `+` operator
     - Walks its argument `_n1`:
       - `enter_composite_expr(_n1)` → **Match!** It's a `p-list`
       - Walks `2` → `enter_term(2)`
       - Walks `y` → `enter_symbol(y)`

### Transformation Pattern

Subclasses override specific `enter_*` methods to transform expressions:

```python
class MyTransformer(Replicator):
    def enter_additive_expr(self, sym, f):
        # Custom transformation for +/- expressions
        if f.sym == Notation.PLUS:
            # Transform plus to minus
            return self.output_notation.repf(
                self.mapsym(sym),
                Func(Notation.MINUS, (self.enter_composite_expr(f.args[0]),))
            )
        # Default behavior
        return super().enter_additive_expr(sym, f)
```

Common transformers in ToyMath:
- **Replicator** (`replicator.py`): Base walker, copies graph between notations
- **Replacer** (`replacer.py`): Applies variable substitutions during walk
- **LaTexWriter** (`LatexWriter.py`): Walks graph and outputs LaTeX strings
- **Calculator** (`processor.py`): Walks graph and applies simplification rules

### LaTexWriter: Walking to Output

`LaTexWriter` follows the same pattern but outputs strings instead of building a new graph:

```python
class LaTexWriter(object):
    def write_formula(self, sym):
        # Walk starts here

    def write_or_expr_list(self, sym):
        # Check for O_LIST, write '\lor' between items

    def write_additive_expr_list(self, sym):
        # Check for S_LIST, walk each additive term

    def write_composite_expr(self, sym):
        # Check for P_LIST (multiplication)

    def write_symbol(self, sym):
        # Output the symbol name
```

Direct `LaTexWriter` output is the faithful/raw graph spelling. User-visible
verified results and notebook-history values use `primitives.write_latex`,
whose `PrettyWriter` candidate is accepted only after reparsing to the same
normal form. This distinction matters for repeated `[[n]]` hops: reparsing an
index introduces transparent brace groups, so raw writing would grow
`x^{3}` into `x^{{3}}`, then `x^{{{3}}}`, while validated pretty output stays
idempotent.

### Context Tracking

The walker maintains a stack of `(symbol, func)` pairs during traversal:

```python
def context_sym(self):
    # Returns parent symbol

def parent_f(self):
    # Returns parent function

def context(self):
    # Returns parent function symbol
```

This enables context-aware transformations. For example, knowing that we're inside a fraction affects how we render expressions.

### Advantages of the Walking Pattern

1. **Immutability**: Original notation is never modified
2. **Composability**: Transformations can be chained
3. **Separation of Concerns**: Each walker handles one transformation
4. **Type Safety**: Grammar hierarchy ensures well-formed expressions
5. **Extensibility**: Override specific methods to customize behavior

## Expression Processing Philosophy

ToyMath separates **canonicalization** from **computation** to maintain predictable behavior:

### Automatic Canonicalization

When expressions with basic operators are entered, the system performs **canonicalization** with **light simplification** as a side effect:

- **Canonicalization**: Converts expressions into a standard form
  - Example: `2x + 3x` → canonical sum representation
  - Collects like terms
  - Simplifies basic arithmetic (`2 + 3` → `5`)
  - Normalizes structure for pattern matching

This happens automatically during parsing and initial processing through the `Calculator` class.

### Explicit Computation via Commands

**Heavy computation** (like bracket expansion, full multiplication, etc.) requires **explicit commands**:

```latex
\mul!{(a+b)(c+d)}      % Expands to: ac + ad + bc + bd
\add!{2x, 3x}          % Adds expressions: 5x
```

**Why this separation?**
1. **Predictability**: Users control when expensive operations occur
2. **Symbolic preservation**: Expressions stay in compact form until explicitly computed
3. **Performance**: Avoids automatic expansion that may create large expressions
4. **Flexibility**: Users can choose whether to keep `(a+b)(c+d)` symbolic or expand it

### Command Examples

Common computation commands (defined in `engine/cmd_*.py`):

- `\mul!{expr1, expr2}` - Multiply and expand expressions (bracket expansion)
- `\add!{expr1, expr2}` - Add expressions with full simplification
- `\simplify!{expr}` - Apply comprehensive simplification rules
- `\expand!{expr}` - Expand all products and powers

The `!` suffix indicates these are **action commands** that perform transformations, not just queries.

### Processing Pipeline

```
User Input (LaTeX)
    ↓
Parser → Notation Graph
    ↓
Automatic Canonicalization (light)
    ↓
Pattern Matching / Unification
    ↓
Explicit Commands (if requested)
    ↓
Heavy Computation (expansion, etc.)
    ↓
Writer → LaTeX Output
```

This design keeps the system responsive while allowing users to control computational complexity.

## Usage in ToyMath

The notation graph and walking pattern are central to how ToyMath processes expressions:

1. **Parser** (`LatexParser.py`) → Creates notation graph from LaTeX input
2. **Processing** → Walks source notation, builds transformed notation:
   - `Replicator` copies expressions between notations
   - `Replacer` applies variable substitutions
   - `Calculator` applies canonicalization and light simplification
3. **Commands** (`cmd_*.py`) → Perform explicit heavy computation when requested
4. **Unification** (`prolog.py`, `comparer.py`) → Walks graphs to match patterns
5. **Writer** (`LatexWriter.py`) → Walks notation graph to produce LaTeX output

Every operation follows the same pattern: **walk the source graph, transform as you go, build the result in a new notation**.

The graph representation combined with the walking pattern enables powerful symbolic manipulation while maintaining clean separation between syntax (LaTeX) and semantics (mathematical operations).
