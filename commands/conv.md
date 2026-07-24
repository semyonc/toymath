---
name: conv
description: Research the convergence of the series via verified tactics
---
Determine whether the series $ARGUMENTS converges, using verified tactics
only.

Load `finite_operators` and work with the whole series expression. Prefer
the value route: `series_partial_sums`, then telescope or a closed form,
then the `limits` skill — exhibiting the value establishes convergence.
When no value route closes, apply `series_converges` with a dominating
geometric (`c r^k`, `r < 1`) or p-series (`c/k^p`, `p > 1`) proposal; its
recorded bound relation and assumptions are the verdict, and "converges
absolutely" may be reported only from that recorded step. There is no
divergence tactic: when nothing applies, record the open outcome with
`set_open`, naming the exact missing move; never conclude convergence or
divergence in prose.
