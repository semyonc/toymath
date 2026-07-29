---
name: simplify
description: Find a simpler equal form, every move mechanically checked
---
Find a simpler form of $ARGUMENTS. Every move must be a verified step; the
final form is the answer, and it is only worth having if the chain that
reached it is checked.

There is no simplicity oracle. Equality is mechanically checked, but *simpler*
is your judgement — so say what you were aiming for (fewer terms, a factored
shape, one trig function instead of three) and let the recorded steps show it.

Start with `expand`: inside the rational fragment it is already a canonical
form, so if it returns the input unchanged there is no rational simplification
to find. Use `collect` to group by a variable — or by a function application
like `\cos x` — and the two named factoring tactics when a factored shape is
the simpler one. `expand` is not always an improvement: on some multivariate
quotients it multiplies both sides out and reads worse than the input. Compare,
and keep the better form.

Outside the rational fragment there is no canonical form, so identities are
your move: propose the equal expression with `rewrite_as` and let `equal?`
check it. That is the route for trig, logs and radicals, and it works whether
or not a lemma is registered — `rewrite x lemma` is worth using when a
registered lemma names exactly the step, because it is an exact structural
match rather than a sampled one. Check `lemmas` to see what is registered.

Prefer several small proposals over one large leap. A single jump from the
input to the answer is checkable, but it teaches the reader nothing and hides
where the work happened.

Cancelling a factor records a `\ne 0` assumption automatically. Do not
suppress or restate those — they are part of the answer, and a "simplified"
form that quietly dropped a domain condition is wrong, not simpler.

When nothing genuinely improves the expression, say so with `set_open` and
name what you tried. Returning the input unchanged is a legitimate outcome;
inventing a rearrangement to look productive is not.
