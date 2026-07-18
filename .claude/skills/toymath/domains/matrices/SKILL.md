---
name: toymath-matrices
key: matrices
description: Cell-wise arithmetic tactics for literal matrices - addition, scalar multiples, ordered products, transpose, and 2x2 determinants.
---

# Matrices and vectors

Use this skill when the task manipulates matrix literals: any of the
`matrix`/`pmatrix`/`bmatrix`/`Bmatrix`/`vmatrix`/`Vmatrix`/`smallmatrix`
environments (or the equivalent `\pmatrix{a & b \cr c & d}` command
spellings). Every tactic here works on literals only - matrices whose
cells are written out. A bare symbol standing for a matrix has no
declaration story yet and will be treated as a scalar; do not rely on it.

Cells may be symbolic (`\pmatrix{a & b \cr c & d}` is fine). Each result
cell is produced by the checked scalar `expand`, and the whole result is
compared against the input numerically, so cell placement is verified
independently - not just the per-cell arithmetic.

Composition order matters and mirrors the algebra:

- `mat_add` needs every top-level term to be a bare literal. Fold scalar
  coefficients first: `2A + B` is `mat_scale` on `2A`, then `mat_add` on
  the sum of the two literals.
- `mat_mul` multiplies exactly two literals and keeps their order (the
  checker will refute a commuted product - `AB \ne BA`). For `ABC`,
  multiply pairwise and feed the recorded result into the next call.
  Write a power `A^2` as an explicit product of the literal with itself.
- `transpose` accepts either the bare literal or the `M^T` spelling; in
  the `M^T` form the recorded step is a true equality between notation
  and value.
- `det_2x2` returns the determinant of a 2x2 literal as a checked
  scalar. A `\vmatrix{...}` input is matrix data like the other
  families; `det_2x2` is the named move that turns it into its
  determinant. Larger determinants have no tactic yet - refusals are
  honest, not divergence into cofactor strategy.

`expand` already merges like matrix *terms* (`2A + 3A` becomes `5A`, and
`A - A` becomes `0`) because matrix products are opaque ordered atoms.
Use these tactics when the arithmetic has to reach *inside* the cells.

Multiplying both sides of an equation by a matrix-valued expression
records that the factor is invertible - a nonzero matrix can still be
singular, so this is deliberately stronger than the scalar `\ne 0`
record. Dividing by a matrix-valued expression is refused; there is no
matrix inverse tactic yet.

If a matrix task needs a move that does not exist (inverse, row
operations, eigenvalues), say so plainly instead of improvising with
scalar tactics.
