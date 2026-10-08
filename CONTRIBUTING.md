# Contributing Guidelines

Thank you for contributing to `hrl-tl`! To maintain high code quality, consistency, and readability across the codebase, please follow the guidelines below.

---

## Code Conventions

All Python code in this repository must conform to the project conventions and the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

### 1. Tooling & Environment

- **Package & Environment Management**: Use `uv` for dependency management, virtual environments, and running scripts (`uv add`, `uv run`, `uv venv`).
- **Formatting & Linting**: Use `ruff` for formatting and linting:
  ```bash
  uv run ruff format .
  uv run ruff check .
  ```
- **Style Guide Baseline**: Follow best practices defined in the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). Line length is set to 80 characters.

### 2. Project & Module Structure

- **File Length**: Keep Python files under ~300 lines to prevent them from becoming "god modules".
- **Helper & Type Scoping**: Scope helper functions, types, and data models to the lowest module level where they are used. If a helper is only used within a specific module, place it directly in that module rather than in a shared `utils.py` or `types.py`.
- **Factoring Out Shared Code**: When functions or classes are genuinely shared across multiple sibling modules, extract them into a dedicated `utils.py`/`types.py` (or a `utils/`/`types/` package if the module grows).
- **File Naming**: Avoid repeating the parent directory name as a prefix for files within that directory:
  - ✅ `hrl_tl/ui/display_utils.py`
  - ❌ `hrl_tl/ui/ui_display_utils.py`
- **Single Source of Truth**: Avoid re-exporting symbols across modules (import directly from the defining module).
- **Simplicity**: Avoid reinventing wheels; use established standard library and third-party packages. Avoid unnecessary defensive branching—simpler code is better.

### 3. Type Hinting & Static Typing

- **Explicit Annotations**: Arguments, return values, and class attributes must have explicit type annotations.
- **Built-in Generics**: Use built-in collection types (`list`, `dict`, `tuple`, `set`) instead of typing aliases (`List`, `Dict`, `Tuple`, `Set`).
- **Abstract Arguments**: Prefer abstract types (from `collections.abc`) for function parameters instead of concrete container types:
  - Use `Mapping` instead of `dict`.
  - Use `Sequence` or `Iterable` instead of `list`.
- **Flag Typing**: Use `absl.flags` flag holder definitions (`flags.DEFINE_*`) to allow flag values to be statically type-checked via `.value` instead of accessing dynamic untyped attributes on `flags.FLAGS`.
- **Forward References**: Add `from __future__ import annotations` to enable forward referencing.

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence


def process_records(
    records: Sequence[str],
    config_by_name: Mapping[str, int],
) -> list[int]:
    return [config_by_name[name] for name in records if name in config_by_name]
```

### 4. Docstrings & Comments

- **Public APIs**: Follow Google-style docstrings (`Args:`, `Returns:`, `Raises:`) for all public functions, classes, and methods.
- **Class Docstrings**: Describe what the instance represents using a noun phrase without using a verb form (e.g., `"A configuration loader for model training pipelines."`).
- **Test Modules**: Module docstrings are not required for test modules and `TestCase` subclasses and should be omitted if they do not provide information beyond what the file name and structure already convey.
- **Nontrivial Functions**: Functions like `main()` or orchestration entry points must have a docstring explaining their purpose and semantics.
- **Attribute Visibility**: Undocumented attributes must be class-private (prefixed with `_`). If an attribute is public, it must be documented in the class docstring.
- **No Redundant Types in Docstrings**: Do not duplicate type annotations in docstrings (e.g., write `arg_name: Description.` rather than `arg_name (type): Description.`, and omit return types in `Returns:` sections).
- **Comments**:
  - Comments must end with trailing periods.
  - Make the code self-explanatory; use comments only when it is infeasible to express the rationale directly through code.
  - Avoid comments that merely restate what the code does.

### 5. Classes, Data Classes & Object-Oriented Design

- **Static Methods**: Never use `@staticmethod` unless required to integrate with an external third-party API. Use top-level module functions instead.
- **Class Methods**: Use `@classmethod` only for named constructors (factory methods) or routines modifying necessary process-wide state (such as a shared cache).
- **Data Classes**: Use Pydantic's `@dataclass` (`from pydantic.dataclasses import dataclass`) or Pydantic `BaseModel` instead of the standard library `@dataclass`.
- **Avoid Duck Typing**: Avoid dynamic introspection like `getattr()` and `hasattr()`; use well-defined types, base classes, or `typing.Protocol` interfaces instead.
- **OOP vs Modules**: Use OOP design where applicable, but avoid overcomplications and Java-style stateless utility classes—well-scoped module functions are preferred.

### 6. Imports & Formatting

- **Submodule Imports**: Only submodules should be imported with the `from ... import ...` syntax (exceptions: `typing` and `collections.abc`).
- **Flag Definition Scope**: Avoid defining flags in library modules to prevent global side-effects on import (like name collisions) and hidden dependencies. Define flags in entry points/binaries and pass configuration values explicitly into library functions.

### 7. Readability & Idioms

- **Explicit None Checks**: Use `is None` or `is not None` to check for the `None` singleton explicitly.
- **Direct Returns**: Use direct `return` / `yield` expressions when intermediate variable assignment is redundant.
- **No Hardcoded Paths**: Expose file paths via configuration flags or arguments with default values rather than hardcoding string paths in source code.
- **Specific Exceptions**: Avoid catching broad exceptions (e.g., bare `except:` or `except Exception:`) unless re-raising them or handling them at top-level thread/process boundaries. Catch specific expected exceptions instead.
- **Exception Logging**: Use `logging.exception()` inside an `except` block to automatically capture and log the full traceback.
- **No `assert` for Control Flow or Logic Switching**: Never use `assert` to switch logic, branch execution, or validate runtime inputs and business invariants. Assertions are stripped when Python executes with optimizations (`-O` or `PYTHONOPTIMIZE`). Reserve `assert` solely for internal consistency checks and unit tests.
- **Maximum Nesting Depth (Flatter is Better)**: Keep nesting levels to a maximum of 3 levels ("flat is better than nested"). If logic requires deeper nesting, factor out the inner loop algorithm or nested logic into a dedicated helper function, or simplify using early returns and guard clauses.
- **No Nested Functions**: Avoid defining functions inside other functions. Factor logic into private module-level helpers (prefixed with `_`), comprehensions, or classes instead (decorators excepted).

### 8. Strings, Regular Expressions & Datetime

- **Compiled Regex**: Regex patterns used in loops or frequently called functions should be pre-compiled at the module level to avoid repeated compilation and cache lookups.
- **Verbose Regex**: When compiling non-trivial regular expressions, use `re.VERBOSE` with inline comments explaining each component.
- **F-Strings**: Use a single cohesive f-string instead of combining a string literal with an f-string via implicit concatenation.
- **Date Arithmetic**: Account for leap years when adding or subtracting years to dates.

### 9. Variables & Data Structures

- **Single Assignment**: Prefer single-assignment form over assign-and-mutate for local variables.
- **Dictionary Naming**: Name dictionaries following the `Y_by_X` convention to enhance readability when indexing (e.g., `user_by_id`, `score_by_student_name`).
- **Global Constants**: Use immutable collections (such as `immutabledict`) for global constant mappings to prevent accidental mutation.

### 10. File & Stream I/O

- **Pathlib**: Prefer `pathlib.Path` for representing and manipulating filesystem paths instead of string-based `os.path` functions.
- **EAFP File Handling**: Avoid testing file existence with `path.exists()` before opening. Instead, handle potential `OSError` / `FileNotFoundError` when opening the file to prevent race conditions.
- **Streaming Reads**: Iterate over the file object directly (`for line in f:`) rather than reading the entire file into memory with `f.read()` and splitting it.

### 11. Testing & Mocking

- **Test Framework**: Use `absl.testing.absltest` and `absl.testing.parameterized` for unit tests. Tests can be executed using `pytest` or `absltest`.
- **Assertion Limits**: Test methods with more than 2–3 assertions should be refactored into separate test methods or parameterized test cases.
- **Strict Mocking**: Always prefer `mock.patch.object()` over string-path `mock.patch()` or bare `MagicMock()`, specifying `autospec=True, spec_set=True` to catch signature drift and invalid attribute access.

---

## GitHub Flow

We follow a GitHub / Gitflow-based development workflow. All development takes place on feature branches branched from and merged back into `dev`. The `dev` branch must always remain functional. Releases are prepared by merging `dev` into `main`.

### 1. Branching

Always ensure your local repository and `dev` branch are up to date:
```bash
git checkout dev
git pull origin dev
```

Create a new branch named descriptively according to the feature or issue:
```bash
git checkout -b feat/123-add-subpolicy-tl
git push -u origin feat/123-add-subpolicy-tl
```

Naming conventions:
- `feat/<issue-id>-<short-description>`
- `fix/<issue-id>-<short-description>`

### 2. Making Changes & Linting

Commit changes regularly. Always run formatter and linter checks before committing:
```bash
# Format code
uv run ruff format .

# Check linting
uv run ruff check .

# Run test suite
uv run pytest
```

### 3. Commit Messages

Commit messages should follow the Conventional Commits format:
```
<type>(<scope>): <body>
```

Where:
- `<type>` is the type of change:
  - `feat`: New features
  - `fix`: Bug fixes
  - `docs`: Documentation updates
  - `style`: Code style/formatting changes
  - `ref`: Code refactoring (e.g., renaming variables, decomposing functions)
  - `perf`: Performance improvements
  - `test`: Adding or updating tests
  - `chore`: Maintenance tasks, dependencies, or build scripts
- `<scope>`: The area of the codebase affected (optional, e.g., `(rewards)`, `(primitives)`)
- `<body>`: A concise explanation of the change in the imperative mood

Example:
```bash
git add hrl_tl/primitives/rep.py
git commit -m "feat(primitives): add representations caching with immutabledict"
```

### 4. Merging via Pull Request

1. Push your commits to your remote branch:
   ```bash
   git push origin feat/123-add-subpolicy-tl
   ```
2. Open a Pull Request targeting the `dev` branch.
3. Include a description of what was changed, the motivation, and verification steps (such as unit test output).
4. Request reviews from the repository maintainers.
