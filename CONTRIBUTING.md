# Contributing to Seismic Workflow Engine

Thanks for considering a contribution to SWE. This guide covers setup, conventions, and the test workflow.

## Development Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/<your-username>/seismic-workflow-engine.git
cd seismic-workflow-engine

# 2. Create the conda environment
conda env create -f environment.yml
conda activate swe

# 3. Install in editable mode with dev dependencies
pip install -e ".[dev,docs]" --no-deps

# 4. Install the git hooks (leak guard on files, commit messages and pushes).
#    Re-run after pulling a change to the hook types in .pre-commit-config.yaml.
pre-commit install
```

## Code Style

SWE enforces consistent style across all modules:

- **`from __future__ import annotations`** at the top of every module.
- **`logger = logging.getLogger(__name__)`** for logging; never `print()`.
- **snake_case** everywhere (functions, variables, modules).
- **Keyword-only arguments** after `*` in public function signatures.
- **NumPy-style docstrings** on all public functions.
- **`__all__`** defined in every `__init__.py` with re-exports of public names.
- No `plt.show()` in library code; return `Figure` objects instead.
- No hardcoded file paths or site-specific configuration.

Linting is enforced via [Ruff](https://docs.astral.sh/ruff/):

```bash
ruff check src/ tests/
ruff format --check src/ tests/
```

## Testing

All contributions must include tests. The project uses **pytest** with synthetic fixtures.

```bash
# Run the full test suite
pytest tests/ -v

# Run with coverage (must meet 85% threshold)
pytest tests/ --cov=swe --cov-report=term-missing --cov-fail-under=85

# Run tests for a specific module
pytest tests/test_noise/ -v
```

### Test conventions

- Test files mirror the source layout: `tests/test_<module>/test_<file>.py`.
- Fixtures use synthetic/in-memory ObsPy objects; no real data files required.
- External services (FDSN, SeisBench models) must be mocked.
- Use `conftest.py` for shared fixtures within a test directory.

## Pull Request Process

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style above.

3. **Add or update tests** to cover your changes.

4. **Run the full test suite** and ensure all tests pass:
   ```bash
   pytest tests/ -v --cov=swe --cov-fail-under=85
   ```

5. **Run linting**:
   ```bash
   ruff check src/ tests/
   ruff format --check src/ tests/
   ```

6. **Commit** with a clear, descriptive message.

7. **Open a pull request** against `main` with:
   - A summary of what changed and why
   - Any relevant issue references
   - Notes on testing performed

## Adding a New Module

If you're adding a new subpackage to `src/swe/`:

1. Create the directory with an `__init__.py` that defines `__all__`.
2. Add `from __future__ import annotations` and logger setup to every file.
3. Use keyword-only arguments in public function signatures.
4. Write NumPy-style docstrings for all public functions.
5. Create a corresponding `tests/test_<module>/` directory with tests.
6. Add an API doc page in `docs/api/<module>.rst`.
7. Update `docs/index.rst` to include the new API page.

## Reporting Issues

- **Bug reports**: Include a minimal reproducible example, Python/ObsPy versions, and the full traceback.
- **Feature requests**: Describe the use case and how it fits into existing SWE workflows.

## Questions?

Open a [GitHub Discussion](https://github.com/tjnewton/seismic-workflow-engine/discussions) for questions about usage or development.
