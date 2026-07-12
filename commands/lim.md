---
name: lim
description: Evaluate the limit move by move via the verified limit tactics
expr: true
---
Evaluate the limit $ARGUMENTS using the verified limit tactics only.

Always pass the whole `\lim` expression — the tactics parse the binder
themselves (one-sided `a^+`/`a^-` binders included). Order of attack:

1. limit_table — constants, the standard zero limits, finite rational
   leading-coefficient limits at infinity, and geometric decay at
   +infinity (`r^n` with numeric `0<r<1`, or division by `s^n` with
   numeric `s>1`, times constant factors — rewrite `e^{-cn}` into that
   form first).
2. limit_substitute — continuity at a finite approach point.
3. limit_rewrite — propose a mechanically equal body that IS in tactic
   shape (cancel, regroup, divide through by the dominant power). equal?
   gates every rewrite, so offer exact algebraic reshapes, not analytic
   leaps.
4. limit_lhopital — only on a quotient, only when the oracle confirms
   0/0 or infinity/infinity, one step at a time; keep the recorded
   differentiability and existence assumptions.
5. limit_linearity to split a top-level sum; finish every piece, then
   call limit_assemble with the linearity step id and the piece step ids
   in the returned order — never retype the assembled sum.

The table is intentionally narrow. If no tactic chain closes the limit,
report the last verified form and what move is missing, then stop —
never assert the remaining value in prose or with a plot.
