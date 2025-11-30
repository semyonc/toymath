# Plan: Implementing Fraction Operations in add! and mul! Commands

## Overview

Implement support for symbolic fraction operations in the `add!` and `mul!` commands. Currently, these commands handle bracket expansion and simplification but don't have specific rules for fractions (`\frac`).

### Key Implementation Principles

1. **Fixed-Point Iteration Loop**: MathProcessor runs transformations until convergence (`s_equal()` returns true)
   - Each iteration: Calculator walks graph → applies rules → builds new notation
   - Fresh notation created each iteration (immutability)
   - **Critical**: Fraction operations must be monotonic (simplify/expand consistently, not oscillate)

2. **Recursive Command Chaining**: Fraction operations recursively invoke `mul!` and `add!` for numerator/denominator
   - Example: `\frac{a}{b} + \frac{c}{d}` → `\frac{\add!{\mul!{ad}+\mul!{cb}}}{\mul!{bd}}`
   - Nested commands evaluated in subsequent iterations
   - Use `chainexpr()` to wrap commands in groups: `\mul!{expr}` → `{\mul!{expr}}`

3. **Automatic Normalization**: Every fraction operation applies `normalize_frac()` (must be idempotent)
   - Numeric reduction: immediate GCD reduction via `FracValue` (inside helper, not deferred)
   - Style canonicalization: map `\\dfrac/\\tfrac/\\cfrac` → canonical `\\frac` (configurable)
   - Sign normalization: consolidate sign on numerator; avoid double negatives
   - Structural simplification: `\\frac{6}{2} → 3`, `\\frac{x}{x} → 1`, flatten nested numerator `\\frac{\\frac{a}{b}}{c} → \\frac{a}{bc}`

4. **Command Operators**:
   - **mul! vs mulex!**: Direct expansion vs deferred evaluation for multi-factor products
   - **add! vs addex!**: Differ by evaluation mode (active flag)
   - Both implemented in `engine/cmd_mul.py` and `engine/cmd_add.py`

5. **Notation Graph Integration**:
   - Fractions are represented as `Notation.FUNC` where the operator symbol name is one of `{\\frac, \\dfrac, \\cfrac, \\tfrac}` (FRAC_SYMBOLS)
   - Representation: `FUNC(Symbol('\\frac' | '\\dfrac' | '\\cfrac' | '\\tfrac'), group(numerator), group(denominator))`
   - Create shared `engine/fractions.py` module to avoid code duplication

## Current Implementation Analysis

### cmd_mul.py (Multiplication)
- **Key method**: `multiplay_plist()` - extracts sum lists and distributes products
- **mul!** (MUL): Direct bracket expansion `(a+b)(c+d) → ac+ad+bc+bd`
- **mulex!** (MULEX): Deferred evaluation for multi-factor products
- **No fraction handling**: Currently doesn't recognize `\frac` as special case
- Need to add `multiply_fractions()` before `multiplay_plist()`

### cmd_add.py (Addition)
- **Key method**: `add_slist()` - recursively builds flat sum list
- **add!** (ADD): Flattens nested sums, combines like terms
- **addex!** (ADDEX): Same as ADD with active flag = True
- **No fraction handling**: Currently doesn't recognize `\frac` as special case
- Need to add fraction collection and combination logic

### MathProcessor Fixed-Point Loop

Multi-iteration example: `\mul!{\frac{2}{3}\frac{3}{4}}`
- **Iteration 1**: `multiply_fractions()` creates `\frac{\mul!{2·3}}{\mul!{3·4}}`
- **Iteration 2**: Nested `mul!` commands expand to `\frac{6}{12}`
- **Iteration 3**: `reduce_frac()` via Preprocessor reduces to `\frac{1}{2}`
- **Iteration 4**: No changes → **STOP**

**Critical**: Reduction is mandatory for canonical form and unification.

### Existing Infrastructure
- **FRAC_SYMBOLS**: `{\\frac, \\dfrac, \\cfrac, \\tfrac}` operator-name set in helpers
- **FracValue**: Numeric fraction class in `engine/value.py` with GCD normalization
- **Preprocessor.enter_oper()**: Reduces numeric `\\frac{m}{n}` during the initial preprocessing pass only. Numeric reductions for fractions created later in the fixed-point loop must be handled inside the helper `normalize_frac()`.
- **chainexpr()**: Wraps commands in groups `{\\mul!{expr}}` (in `cmd_mul.py`)

## Required Fraction Rules

**Note**: All fraction operations include automatic reduction as the final step.

### Rule 1: Fraction Addition
```
\frac{x1}{y1} + \frac{x2}{y2} → \frac{\add!{\mul!{x1·y2} + \mul!{x2·y1}}}{\mul!{y1·y2}}
```

### Rule 2: Fraction Multiplication
```
\frac{x1}{y1} · \frac{x2}{y2} → \frac{\mul!{x1·x2}}{\mul!{y1·y2}}
```

### Rule 3: Scalar × Fraction
```
a · \frac{x}{y} → \frac{\mul!{a·x}}{y}
```

### Rule 4: Fraction × Sum
```
\frac{x}{y} · (a + b) → \frac{\mul!{x·(a+b)}}{y}
```

### Rule 5: Scalar + Fraction
```
a + \frac{x}{y} → \frac{\add!{\mul!{a·y} + x}}{y}
```

## Implementation Strategy

### Phase 1: Helper Module (`engine/fractions.py`)

Create shared helper functions to avoid code duplication:

```python
from notation import Notation, Symbol
from value import IntegerValue
from value import division
from cmd_mul import chainexpr, Mul

FRAC_SYMBOLS = {'\\frac', '\\dfrac', '\\cfrac', '\\tfrac'}

def is_frac(notation, sym):
    f = notation.getf(sym, Notation.FUNC)
    return f is not None and isinstance(f.sym, Symbol) and f.sym.name in FRAC_SYMBOLS

def extract_frac(notation, sym):
    """Extract numerator and denominator from FUNC(\\frac|\\dfrac|..., group(num), group(den)).
    Returns: (numerator_sym, denominator_sym) or None
    """
    f = notation.getf(sym, Notation.FUNC)
    if f is None or not (isinstance(f.sym, Symbol) and f.sym.name in FRAC_SYMBOLS):
        return None
    num_g = notation.getf(f.args[0], Notation.GROUP)
    den_g = notation.getf(f.args[1], Notation.GROUP)
    return (num_g.args[0], den_g.args[0])

def make_group(notation, expr):
    return notation.setf(Notation.GROUP, (expr,), br="{}")

def make_frac(notation, numerator, denominator, style='\\frac'):
    return notation.setf(Notation.FUNC, (Symbol(style), make_group(notation, numerator), make_group(notation, denominator)))

def normalize_frac(notation, frac_sym, canonical_style='\\frac'):
    """Normalize a fraction in one place (idempotent):
    - Style: map all FRAC_SYMBOLS to canonical_style (default: \\frac)
    - Sign: normalize sign to numerator; avoid double negatives
    - Flatten: \\frac{\\frac{a}{b}}{c} → \\frac{a}{bc}
    - Numeric reduction: immediate via FracValue (no preprocessor dependency)
    - Symbolic reductions: x/x → 1; 0/x → 0; keep a/0 and 0/0 symbolic
    Returns a Value or a normalized FUNC(\\frac, ...)
    """
    f = notation.getf(frac_sym, Notation.FUNC)
    if f is None or not (isinstance(f.sym, Symbol) and f.sym.name in FRAC_SYMBOLS):
        return frac_sym

    # Canonicalize operator style
    num_g = notation.getf(f.args[0], Notation.GROUP)
    den_g = notation.getf(f.args[1], Notation.GROUP)
    num = num_g.args[0]
    den = den_g.args[0]

    # Flatten nested numerator fraction: (a/b)/c → a/(b*c)
    nested = notation.getf(num, Notation.FUNC)
    if nested is not None and isinstance(nested.sym, Symbol) and nested.sym.name in FRAC_SYMBOLS:
        nx_g = notation.getf(nested.args[0], Notation.GROUP)
        ny_g = notation.getf(nested.args[1], Notation.GROUP)
        nx, ny = nx_g.args[0], ny_g.args[0]
        # new denominator: mul!(ny, den)
        den_pl = notation.setf(Notation.P_LIST, (ny, den))
        den = notation.getf(chainexpr(Mul.MUL, notation, den_pl, None), Notation.GROUP).args[0]
        num = nx

    # 0/x → 0 (unless x=0 too). a/0 and 0/0 remain symbolic
    if isinstance(num, IntegerValue) and num.val == 0:
        if isinstance(den, IntegerValue) and den.val == 0:
            # indeterminate stays symbolic
            pass
        else:
            return IntegerValue(0)
    if isinstance(den, IntegerValue) and den.val == 0:
        # keep symbolic
        return make_frac(notation, num, den, style=canonical_style)

    # Numeric reduction now
    from engine.processor import get_value
    xv = get_value(num, notation)
    yv = get_value(den, notation)
    if xv is not None and yv is not None:
        dv = division(xv.get_frac(), yv.get_frac())
        if dv is not None:
            return dv

    # Symbolic cancellation x/x → 1
    from comparer import s_equal
    if s_equal(num, notation, den, notation):
        return IntegerValue(1)

    return make_frac(notation, num, den, style=canonical_style)
```

**Design notes**:
- Single source of truth for fraction helpers
- `normalize_frac()` performs numeric GCD immediately and centralizes style/sign/flatten rules
- Handles `\frac`, `\dfrac`, `\cfrac`, `\tfrac` by operator-name check (FRAC_SYMBOLS)

### Phase 2: Update cmd_mul.py

#### Step 2.1: Add pairwise fraction handling inside `multiplay_plist()`

```python
# Inside multiplay_plist, before distribution:
from fractions import is_frac, extract_frac, make_frac, normalize_frac

# If both factors are present and one/both are fractions, build a single fraction:
if len(f.args) == 2:
    a, b = f.args
    a_is = is_frac(notation, a)
    b_is = is_frac(notation, b)
    if a_is or b_is:
        # frac*frac, frac*scalar or scalar*frac
        ax, ay = extract_frac(notation, a) if a_is else (a, IntegerValue(1))
        bx, by = extract_frac(notation, b) if b_is else (b, IntegerValue(1))
        num_pl = notation.setf(Notation.P_LIST, (ax, bx))
        den_pl = notation.setf(Notation.P_LIST, (ay, by))
        num = notation.getf(chainexpr(self.MUL, notation, num_pl, None), Notation.GROUP).args[0]
        den = notation.getf(chainexpr(self.MUL, notation, den_pl, None), Notation.GROUP).args[0]
        res = make_frac(notation, num, den)
        res = normalize_frac(notation, res)
        if negative:
            res = notation.setf(Notation.MINUS, (res,))
        return res
```

#### Step 2.2: Implement pairwise fraction combine before distribution

```python
from fractions import is_frac, extract_frac, make_frac, normalize_frac

def multiply_fractions(self, processor, notation, sym, negative):
    """Handle fraction multiplication cases:
    1. frac * frac
    2. scalar * frac
    3. frac * scalar
    4. frac * sum
    """
    # Extract product list
    f = notation.getf(sym, Notation.GROUP)
    if f is not None:
        sym = f.args[0]

    f = notation.getf(sym, Notation.P_LIST)
    if f is None:
        return None

    # Classify factors as fractions or scalars
    fractions = []
    scalars = []

    for arg in f.args:
        if is_frac(notation, arg):
            fractions.append(extract_frac(notation, arg))
        else:
            scalars.append(arg)

    if len(fractions) == 0:
        return None  # No fractions, let normal multiplication handle it

    # Multiply all numerators together (fractions + scalars)
    numerators = [frac[0] for frac in fractions] + scalars
    denominators = [frac[1] for frac in fractions]

    # Build mul! expressions for numerator and denominator
    if len(numerators) == 1:
        num_expr = numerators[0]
    else:
        num_plist = notation.setf(Notation.P_LIST, tuple(numerators))
        num_expr = chainexpr(self.MUL, notation, num_plist, None)
        # Unwrap the GROUP that chainexpr creates
        num_expr = notation.getf(num_expr, Notation.GROUP).args[0]

    if len(denominators) == 1:
        den_expr = denominators[0]
    else:
        den_plist = notation.setf(Notation.P_LIST, tuple(denominators))
        den_expr = chainexpr(self.MUL, notation, den_plist, None)
        # Unwrap the GROUP
        den_expr = notation.getf(den_expr, Notation.GROUP).args[0]

    # Create result fraction and normalize
    result = make_frac(notation, num_expr, den_expr)
    result = normalize_frac(notation, result)

    if negative:
        result = notation.setf(Notation.MINUS, (result,))

    return result
```

### Phase 3: Update cmd_add.py

#### Step 3.1: Modify `add_slist()` to collect fractions

```python
from fractions import is_frac, extract_frac, make_frac, normalize_frac
from cmd_mul import chainexpr, Mul

def add_slist(self, out, notation, sym):
    # ... existing code for GROUP and S_LIST handling ...

    # Collect fractions and non-fractions separately
    fractions = []
    non_fractions = []

    for arg in f.args:
        expr = arg
        negative = False

        # Extract sign
        f_sign = notation.vgetf(expr, [Notation.PLUS, Notation.MINUS])
        if f_sign is not None:
            if f_sign.sym == Notation.MINUS:
                negative = True
            expr = f_sign.args[0]

        # Classify
        if is_frac(notation, expr):
            fractions.append((expr, negative))
        else:
            non_fractions.append((expr, negative))

    # Process non-fractions normally (existing logic)
    for expr, negative in non_fractions:
        # ... existing logic ...
        pass

    # Combine fractions if present
    if len(fractions) > 0:
        combined = self.combine_fractions(notation, fractions)
        out.append(combined)

    return out
```

#### Step 3.2: Implement `combine_fractions()`

```python
def combine_fractions(self, notation, fractions):
    """Combine multiple fractions with common denominator
    Input: [(frac_sym, negative), ...]
    Output: combined fraction
    """
    if len(fractions) == 1:
        frac, negative = fractions[0]
        if negative:
            return notation.setf(Notation.MINUS, (frac,))
        return frac

    # For two fractions: \frac{x1}{y1} + \frac{x2}{y2} = \frac{x1*y2 + x2*y1}{y1*y2}
    if len(fractions) == 2:
        (x1, y1, neg1) = (extract_frac(notation, fractions[0][0])[0],
                          extract_frac(notation, fractions[0][0])[1],
                          fractions[0][1])
        (x2, y2, neg2) = (extract_frac(notation, fractions[1][0])[0],
                          extract_frac(notation, fractions[1][0])[1],
                          fractions[1][1])

        # Calculate x1 * y2
        mul1_plist = notation.setf(Notation.P_LIST, (x1, y2))
        mul1_expr = chainexpr(Mul.MUL, notation, mul1_plist, None)
        mul1_expr = notation.getf(mul1_expr, Notation.GROUP).args[0]

        # Apply sign to first term
        if neg1:
            mul1_expr = notation.setf(Notation.MINUS, (mul1_expr,))

        # Calculate x2 * y1
        mul2_plist = notation.setf(Notation.P_LIST, (x2, y1))
        mul2_expr = chainexpr(Mul.MUL, notation, mul2_plist, None)
        mul2_expr = notation.getf(mul2_expr, Notation.GROUP).args[0]

        # Apply sign to second term
        if neg2:
            mul2_expr = notation.setf(Notation.MINUS, (mul2_expr,))
        else:
            mul2_expr = notation.setf(Notation.PLUS, (mul2_expr,))

        # Create sum: x1*y2 + x2*y1
        add_slist = notation.setf(Notation.S_LIST, (mul1_expr, mul2_expr))
        numerator = chainexpr(Add.ADD, notation, add_slist, None)
        numerator = notation.getf(numerator, Notation.GROUP).args[0]

        # Calculate y1 * y2
        den_plist = notation.setf(Notation.P_LIST, (y1, y2))
        denominator = chainexpr(Mul.MUL, notation, den_plist, None)
        denominator = notation.getf(denominator, Notation.GROUP).args[0]

        # Create result fraction and normalize
        result = make_frac(notation, numerator, denominator)
        return normalize_frac(notation, result)

    # For more than 2 fractions, combine pairwise recursively
    # (Implement as needed)
```

#### Step 3.3: Handle scalar + fraction

```python
def add_scalar_and_fraction(self, notation, scalar, frac_sym, scalar_negative, frac_negative):
    """Add a scalar and a fraction: a + \frac{x}{y} = \frac{a*y + x}{y}"""
    x, y = extract_frac(notation, frac_sym)

    # Calculate a * y
    mul_plist = notation.setf(Notation.P_LIST, (scalar, y))
    mul_expr = chainexpr(Mul.MUL, notation, mul_plist, None)
    mul_expr = notation.getf(mul_expr, Notation.GROUP).args[0]

    # Apply signs
    if scalar_negative:
        mul_expr = notation.setf(Notation.MINUS, (mul_expr,))
    else:
        mul_expr = notation.setf(Notation.PLUS, (mul_expr,))

    if frac_negative:
        x_term = notation.setf(Notation.MINUS, (x,))
    else:
        x_term = notation.setf(Notation.PLUS, (x,))

    # Create sum: a*y + x
    add_slist = notation.setf(Notation.S_LIST, (mul_expr, x_term))
    numerator = chainexpr(Add.ADD, notation, add_slist, None)
    numerator = notation.getf(numerator, Notation.GROUP).args[0]

    # Create result fraction and normalize
    result = make_frac(notation, numerator, y)
    return normalize_frac(notation, result)
```

## Testing Strategy

### Phase 4: Create Unit Tests

Create/augment tests in `engine/unittests.py`:

```python
class TestFractionOperations(unittest.TestCase):
    """Test fraction operations in add! and mul! commands"""

    def test_reduce_simple(self):
        """Test: \frac{2}{4} reduces to \frac{1}{2} (inside loop)"""
        self.assertTrue(check(r'\mul!{\frac{2}{4}}', r'\frac{1}{2}'))

    def test_reduce_to_integer(self):
        """Test: \frac{6}{2} = 3"""
        self.assertTrue(check(r'\mul!{\frac{6}{2}}', r'3'))

    def test_symbolic_cancel(self):
        """Test: \frac{x}{x} = 1"""
        self.assertTrue(check(r'\mul!{\frac{x}{x}}', r'1'))

    def test_frac_mul_frac(self):
        """Test: \frac{a}{b} * \frac{c}{d} = \frac{\mul!{ac}}{\mul!{bd}}"""
        self.assertTrue(check(
            r'\mul!{\frac{a}{b}\frac{c}{d}}',
            r'\frac{\mul!{ac}}{\mul!{bd}}'
        ))

    def test_frac_mul_frac_numeric(self):
        """Test: \frac{2}{3} * \frac{3}{4} = \frac{1}{2}"""
        self.assertTrue(check(
            r'\mul!{\frac{2}{3}\frac{3}{4}}',
            r'\frac{1}{2}'
        ))

    def test_scalar_mul_frac(self):
        """Test: 2 * \frac{x}{y} = \frac{\mul!{2x}}{y}"""
        self.assertTrue(check(
            r'\mul!{2\frac{x}{y}}',
            r'\frac{\mul!{2x}}{y}'
        ))

    def test_frac_add_numeric(self):
        self.assertTrue(check(r'\add!{\frac{1}{2}+\frac{1}{3}}', r'\frac{5}{6}'))

    def test_frac_mul_three(self):
        self.assertTrue(check(r'\mul!{\frac{a}{b}\frac{c}{d}\frac{e}{f}}', r'\frac{\mul!{ace}}{\mul!{bdf}}'))

    def test_frac_add_frac_numeric(self):
        """Test: \frac{1}{4} + \frac{1}{4} = \frac{1}{2}"""
        self.assertTrue(check(
            r'\add!{\frac{1}{4}+\frac{1}{4}}',
            r'\frac{1}{2}'
        ))

    def test_scalar_add_frac(self):
        """Test: a + \frac{x}{y} = \frac{\add!{\mul!{ay}+x}}{y}"""
        self.assertTrue(check(
            r'\add!{a+\frac{x}{y}}',
            r'\frac{\add!{\mul!{ay}+x}}{y}'
        ))

    def test_mixed_operations(self):
        """Test: (\frac{a}{b} + c) * \frac{d}{e}"""
        self.assertTrue(check(
            r'\mul!{(\frac{a}{b}+c)\frac{d}{e}}',
            r'\frac{\mul!{(\add!{\frac{a}{b}+c})d}}{e}'
        ))

    def test_style_variants(self):
        self.assertTrue(check(r'\mul!{\dfrac{2}{4}}', r'\frac{1}{2}'))
        self.assertTrue(check(r'\mul!{\tfrac{2}{4}}', r'\frac{1}{2}'))
        self.assertTrue(check(r'\mul!{\cfrac{2}{4}}', r'\frac{1}{2}'))
```

### Test Execution Plan

1. **Phase 4.1**: Write tests first (TDD approach)
2. **Phase 4.2**: Run tests to confirm they fail
3. **Phase 4.3**: Implement `engine/fractions.py` helpers
4. **Phase 4.4**: Implement mul! fraction support
5. **Phase 4.5**: Run multiplication tests
6. **Phase 4.6**: Implement add! fraction support
7. **Phase 4.7**: Run addition tests
8. **Phase 4.8**: Fix edge cases and refine

## Edge Cases to Consider

| Case Description | Expected Behavior | Implementation Note | Test Snippet |
|------------------|-------------------|---------------------|--------------|
| Numeric reduction | \frac{2}{4} → \frac{1}{2} | Performed in normalize_frac() | self.assertTrue(check(r'\mul!{\frac{2}{4}}', r'\frac{1}{2}')) |
| GCD example | \frac{6}{9} → \frac{2}{3} | Immediate via FracValue in normalize_frac() | self.assertTrue(check(r'\mul!{\frac{6}{9}}', r'\frac{2}{3}')) |
| Negative in numerator | \frac{-4}{6} → \frac{-2}{3} | Normalize sign to numerator in normalize_frac() | self.assertTrue(check(r'\mul!{\frac{-4}{6}}', r'\frac{-2}{3}')) |
| Negative in denominator | \frac{4}{-6} → \frac{-2}{3} | Normalize sign to numerator in normalize_frac() | self.assertTrue(check(r'\mul!{\frac{4}{-6}}', r'\frac{-2}{3}')) |
| Symbolic cancellation | \frac{x}{x} → 1 | Check equality in normalize_frac() | self.assertTrue(check(r'\mul!{\frac{x}{x}}', r'1')) |
| Symbolic with common factors | \frac{2x}{4x} → \frac{1}{2} (requires factoring, not implemented yet) | Deferred to Future Enhancement; requires factoring out common terms | (Future: self.assertTrue(check(r'\mul!{\frac{2x}{4x}}', r'\frac{1}{2}'))) |
| Nested negatives | \frac{-a}{-b} → \frac{a}{b} | Count negatives in normalization | self.assertTrue(check(r'\mul!{\frac{-a}{-b}}', r'\frac{a}{b}')) |
| Zero numerator | \frac{0}{x} → 0 | Handled in normalize_frac() | self.assertTrue(check(r'\mul!{\frac{0}{x}}', r'0')) |
| Zero denominator | \frac{a}{0} → Keep symbolic | No evaluation; warn if possible | self.assertTrue(check(r'\mul!{\frac{a}{0}}', r'\frac{a}{0}')) |
| Indeterminate | \frac{0}{0} → Keep symbolic | No evaluation | self.assertTrue(check(r'\mul!{\frac{0}{0}}', r'\frac{0}{0}')) |
| Fraction in numerator | \frac{\frac{a}{b}}{c} → \frac{a}{bc} | Flatten in normalize_frac() | self.assertTrue(check(r'\mul!{\frac{\frac{a}{b}}{c}}', r'\frac{a}{bc}')) |
| Result is integer | \frac{6}{2} → 3 | If denominator == 1 after normalization, return numerator | self.assertTrue(check(r'\mul!{\frac{6}{2}}', r'3')) |
| Equals 1 | \frac{4}{4} → 1 | Same as above | self.assertTrue(check(r'\mul!{\frac{4}{4}}', r'1')) |
| Unary minus on fraction | -\frac{a}{b} + \frac{c}{d} | Handle in add_slist() sign extraction | self.assertTrue(check(r'\add!{-\frac{a}{b} + \frac{c}{d}}', r'\frac{\add!{-\mul!{a d} + \mul!{c b}}}{\mul!{b d}}')) |

## Implementation Phases

### Phase 1: Foundation
- [ ] Create `engine/fractions.py` with helpers (is_frac, extract_frac, make_frac, reduce_frac)
- [ ] Write basic unit tests for reduction and operations
- [ ] Set up test infrastructure

### Phase 2: Multiplication Support
- [ ] Wire `cmd_mul.Mul.main()` to call `multiply_fractions()` before `multiplay_plist()`
- [ ] Implement `multiply_fractions()` (frac × frac, scalar × frac)
- [ ] Test multiplication operations
- [ ] Handle edge cases

### Phase 3: Addition Support
- [ ] Extend `Add.add_slist()` to collect and combine fractions
- [ ] Implement `combine_fractions()` (same and different denominators)
- [ ] Implement `add_scalar_and_fraction()`
- [ ] Test addition operations

### Phase 4: Integration and Refinement
- [ ] Test complex mixed operations
- [ ] Handle nested fractions
- [ ] Add simplification rules
- [ ] Enhance `enter_slist()` in `processor.py` to ensure pure scalar terms (including scalar fractions) combine numerically without oscillation
- [ ] Documentation and code cleanup

### Phase 5: Documentation and Project Integration Requirements
- Update NOTATION.md with fraction graph examples (FUNC with FRAC_SYMBOLS and GROUP args; canonical style).
- Add to CLAUDE.md: Note new convergence rules for fraction operations (e.g., reductions must be idempotent to prevent loop oscillation).
- Inline code comments: Add comments in fractions.py, cmd_add.py, and cmd_mul.py explaining monotonicity (e.g., "# This reduction ensures term count decreases for convergence").
- Integration checks: Verify compatibility with Prolog unification in prolog.py (e.g., test fraction patterns in rules).
- Run full test suite and document any changes to existing behavior.

## Technical Considerations

### Numeric vs Symbolic Behavior
- **Numeric-only** `\frac{m}{n}` collapses in `Preprocessor.enter_oper()` only for initial input; numeric reductions for fractions introduced by rules happen inside `normalize_frac()`.
- **Symbolic** `\frac{x}{x}` simplifies to `1` via `normalize_frac()`.
- Tests should expect evaluated numeric form after the loop without requiring an extra preprocessor pass.

### Performance Guidelines
- Ensure all operations are idempotent (e.g., reduce_frac() returns the same output on repeat calls to avoid infinite loops in fixed-point iteration).
- Operations must be monotonic: Simplifications should strictly reduce complexity or term count to guarantee convergence.
- Benchmark iteration counts: Aim for <10 iterations in complex fractions; test with trace in MathProcessor.
- Optimizations: Cache common denominators in combine_fractions() for multi-fraction additions; avoid unnecessary chainexpr() calls.
- Monitor for cycles: Add debug assertions in Calculator to detect oscillation (e.g., if iteration >20, log warning).

## Success Criteria

The implementation is complete when:

1. ✅ All unit tests pass
2. ✅ Fraction reduction works correctly (numeric GCD, sign normalization, symbolic x/x → 1)
3. ✅ Fraction × fraction works with automatic reduction
4. ✅ Scalar × fraction works with automatic reduction
5. ✅ Fraction + fraction works (different denominators) with automatic reduction
6. ✅ Scalar + fraction works with automatic reduction
7. ✅ Nested operations work correctly
8. ✅ Edge cases handled gracefully
9. ✅ Code is well-documented
10. ✅ Performance is acceptable

## Future Enhancements

After basic implementation, prioritize these enhancements in sequence:

1. **Deep nested fraction flattening** [Medium priority]: Generalize flattening for multiple nested levels and denominator nesting (e.g., \frac{a}{\frac{b}{c}}), ensuring idempotence in `normalize_frac()`.
2. **Advanced symbolic reduction** [High priority; depends on Phase 4]: Factor and reduce \frac{2x}{4x} → \frac{1}{2}; integrate with LLM in llm_comparer.py; build on monomial enhancements in enter_slist() from processor.py.
3. **LCM optimization** [Medium priority; depends on Phase 3]: When adding multiple fractions, use LCM of denominators instead of product; add to combine_fractions().
4. **Fraction division** [Medium priority; depends on Phase 2]: Support \frac{a}{b} / \frac{c}{d} = \frac{a}{b} * \frac{d}{c}; extend multiply_fractions().
5. **Fraction powers** [Low priority; depends on Phase 2]: Handle (\frac{a}{b})^n = \frac{a^n}{b^n}; add to cmd_mul.py power handling.
6. **Mixed numbers** [Low priority]: Support mixed number notation (e.g., 1 \frac{2}{3}).
7. **Continued fractions** [Low priority]: Support expansion; may require new cmd_continued.py.
8. **Partial fraction decomposition** [Low priority; depends on advanced reduction]: For rational functions; integrate with Prolog rules.
