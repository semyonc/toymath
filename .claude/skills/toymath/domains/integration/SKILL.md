---
name: toymath-integration
key: integration
description: Tactic-shaped indefinite integration, substitutions, by-parts, linearity, and provenance-aware assembly.
---

# Indefinite integration

There is no autonomous integrate tactic. Pick one narrow move at a time and
continue until no integral remains.

## Order of attack

1. Try `integrate_power_rule` for polynomial powers or `integrate_table` for
   basic functions, constant multiples, and `1/x`.
2. Use `integrate_substitute` when you can propose `u`, a new variable, and the
   rewritten integrand. ToyMath checks `f(u(x))u'(x)` against the original.
3. Use `integrate_by_parts` for one proposed `u·dv` split. Continue from its
   `remaining_integral` handle.
4. For algebraic massage or partial fractions, propose the new integrand with
   `integrate_rewrite`; core equality must confirm it.
5. Split a top-level sum with `integrate_linearity`, solve every returned
   integral independently, pin per-piece constants to zero with core
   `substitute`, then cite the ordered ledger step ids in
   `integrate_assemble`.

Never type an assembled sum into core `expand`: that checks only the expression
you typed, not whether it contains the recorded pieces. Assembly provenance is
part of the mechanical certificate.

Antiderivatives carry a fresh `+ C`. The independent leg differentiates the
candidate; `1/x -> ln x` records the positive-domain assumption used by the
current table rule.
