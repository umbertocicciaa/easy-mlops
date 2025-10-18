# Local Development

This guide covers environment setup, coding conventions, testing, and documentation workflows for Make MLOps Easy contributors.

## Environment Setup

Create a virtual environment and install the project with development extras:

```bash
make install-dev
```

This command installs the package in editable mode plus dependencies for testing, linting, type checks, documentation, and coverage utilities.

### Virtual Environment

The default environment lives under `.venv/`. Activate it explicitly when running commands outside the Makefile:

```bash
source .venv/bin/activate
```

## Coding Standards

- **Formatting**: Run `make format` (Black) before committing.
- **Linting**: Use `make lint` (Flake8) to keep code compliant.
- **Typing**: Optionally run `mypy easy_mlops/` to catch typing issues.

## Testing & Coverage

Unit and integration tests reside under `tests/`. Execute:

```bash
make test
```

Coverage uses Python’s built-in `trace` module because `pytest-cov` requires network access to fetch wheels in some environments. Generate a coverage summary with:

```bash
make coverage
```

Coverage artifacts are stored in `trace_summary/`. The GitHub Actions workflow uploads the same directory for CI review.

## Distributed runtime tips

- Start a development master/worker pair with `make-mlops-easy master start` and `make-mlops-easy worker start`. Use the `examples/distributed_runtime.sh` script when you prefer managed processes.
- Workers stream stdout back to the master; add `print` statements in pipeline code to debug tasks quickly.
- Override the state file path (`--state-path`) to keep reproducible fixtures for integration tests.

## Documentation

Docs are built with MkDocs and the Material theme. To preview locally:

```bash
make docs-serve
```

Build the static site with:

```bash
make docs-build
```

The `docs.yml` GitHub Actions workflow automatically deploys the site to GitHub Pages when changes land on `main`. For manual releases or forks, run:

```bash
make docs-deploy
```

## Makefile Shortcuts

The Makefile centralizes tasks such as linting, testing, documentation, and CLI invocations. Run `make help` to inspect available targets.

## Pre-Commit Checklist

Before pushing:

1. `make format lint test`
2. `make coverage` (optional but recommended)
3. `make docs-build` to ensure documentation builds cleanly
4. Review `git status` for unintended files (logs, artifacts should stay ignored).

See [CI/CD](cicd.md) for the automated pipelines that validate pushes and releases.
