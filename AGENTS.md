# AGENTS.md

Guidance for coding agents working in this repository. Focus on convergence-safe math transformations, the notation DAG, and Prolog-style matching.

## Project Overview

ToyMath is a symbolic math system packaged as a Jupyter kernel and console app. It parses LaTeX, stores expressions as a notation DAG, runs a fixed-point transformation loop using command plugins, and renders results back to LaTeX. Prolog-style unification and optional LLM matching support pattern-driven rewrites.

## Architecture Highlights

- **Notation graph**: `notation.py` defines `Symbol`/`Func` nodes; expressions live in a DAG keyed by `Notation.rel`. All transformers walk source notation and build results in a fresh notation object.
- **Parser/Writer**: `LatexParser.py` + `lexer.py` parse LaTeX → notation; `LatexWriter.py` converts notation → LaTeX. Parser artifacts (`parser.out`, `parsetab.py`) are cache files.
- **Processing loop**: `processor.py` drives a fixed-point iteration. Each round uses `Calculator` (extends `Replicator`) to walk the graph, execute commands (`engine/cmd_*.py`), and emit a new graph. Loop stops when `s_equal` finds no change. Commands must be monotonic/idempotent to avoid oscillation.
- **Commands**: Loaded via `register_actions()` scanning `engine/cmd_*.py`. Each command implements `exec(processor, sym, f)`, may return `Notation.NONE` to suppress output, and can nest other commands via helpers like `chainexpr`.
- **Logic layer**: `prolog.py` provides unification/pattern rules; `comparer.py` and `llm_comparer.py` supply structural and LLM-aware matching.

## Development Tips

- Preserve immutability: never mutate the input notation; always build into `processor.output_notation`.
- Favor single-iteration tests for new commands to isolate behavior; full processor loop runs until fixed point.
- Ensure transformations shrink/normalize expressions or expand in a stable, one-way fashion; avoid toggling forms.
- When adding commands, keep arity correct, register via `create_actions()`, and validate with pytest targets in `engine/unittests*.py`.
- LLM matching requires `OPENAI_API_KEY` in `.env`; otherwise skip those tests.

## Running and Testing

- Console mode: `python console.py`
- Kernel: `jupyter notebook` then pick the LaTeX kernel (registered by `toymathkernel.py`)
- Tests: `pytest engine/unittests.py` (core); `pytest engine/unittests_llm.py` (LLM)
- Activate the venv first (`source .venv/bin/activate`) or set `PYTHONPATH=.` when running tests to avoid import errors for `engine/*`.
- Package management: use `uv pip install -r requirements.txt` for dependencies.

## Reference Reading

- `NOTATION.md`: notation DAG and walking pattern details
- `FRACTION_OPERATIONS_PLAN.md`: roadmap for fraction support in `add!` and `mul!`
- `engine/cmd_*.py`: command implementations and examples
