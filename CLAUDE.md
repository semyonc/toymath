# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ToyMath is a symbolic mathematics system implemented as a Jupyter kernel that parses LaTeX mathematical expressions, performs symbolic manipulation and unification using Prolog-style logic, and outputs results as LaTeX. The system combines mathematical notation processing with logic programming capabilities and includes experimental LLM-based pattern matching.

## Core Architecture

### Execution Flow

1. **Input** → LaTeX expressions entered in Jupyter or console
2. **Parsing** → `LatexParser.py` (PLY-based yacc parser) + `lexer.py` → creates Symbol/Func representation
3. **Notation** → `notation.py` stores symbol metadata and function relationships
4. **Processing** → `processor.py` (Calculator/MathProcessor) applies rules and transformations
5. **Prolog Engine** → `prolog.py` provides unification, pattern matching, and rule-based inference
6. **Output** → `LatexWriter.py` converts symbols back to LaTeX for display

### Key Components

**Kernel Layer:**
- `toymathkernel.py`: IPython kernel implementation (MathKernel class)
- `console.py`: CLI interface for interactive use
- `engine/mathShell.py`: Core shell that orchestrates parsing, processing, and output

**Notation System:**
- `notation.py`: Symbol and Func classes represent mathematical terms
- Symbols have names and properties; Funcs link symbols to arguments
- Notation object manages the symbol table and relationships

**Parser/Writer:**
- `LatexParser.py`: Yacc-based parser converts LaTeX → internal representation
- `LatexWriter.py`: Converts internal representation → LaTeX output
- Both work with Notation to track symbol metadata

**Processing:**
- `processor.py`: MathProcessor applies transformations via Calculator and rule system
- `prolog.py`: PrologModel provides unification, pattern matching, and goal resolution
- `comparer.py`: UnifyComparer and pattern matching for symbolic expressions
- `replacer.py`: Replaces symbols according to substitutions
- `replicator.py`: Copies symbol structures across notations

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
```bash
pip install -r requirements.txt
```
Key dependencies: jupyter, ipykernel, ipywidgets, openai, ply, pytest, python-dotenv

For LLM features, create `.env` file with:
```
OPENAI_API_KEY=your_key_here
```

## Code Patterns

### Adding a New Command

1. Create `engine/cmd_yourname.py`:
```python
from notation import Notation

class YourCommand(object):
    arity = 1  # number of required arguments

    def exec(self, processor, sym, f):
        # f.args[0] = parameters (list or None)
        # f.args[1] = arguments tuple
        # Return Notation.NONE to suppress output
        # Return a symbol to display result
        pass

def create_actions():
    return {'yourname': YourCommand()}
```

2. The processor automatically loads commands from `engine/cmd_*.py` files.

3. Use in LaTeX as `\yourname{arg}` or `\yourname[params]{arg}`

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

## File Organization

- `engine/`: Core math processing engine
  - `cmd_*.py`: Command implementations
  - `*Parser.py`, `*Writer.py`: Input/output
  - `prolog.py`, `comparer.py`: Logic engine
  - `unittests*.py`: Test suites
- `examples/`: Jupyter notebooks demonstrating features
- `bin/`: Virtual environment (not in git)
- `obsolete/`: Old/deprecated code

## Important Notes

- Parser generates `parser.out` and `parsetab.py` from `LatexParser.py` (cached, safe to ignore)
- Symbols use name-based equality; Symbol('x') == Symbol('x')
- Notation objects are context-specific; symbols must be replicated across notations
- The `!` operator implements cut (like Prolog cut)
- LLM features are experimental and require API access
