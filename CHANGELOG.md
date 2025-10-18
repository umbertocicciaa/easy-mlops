# Changelog
All notable changes to this project are documented here.

## [0.4.0]
- **Added**: Distributed runtime with a FastAPI master service backed by a JSON state store for workflow orchestration.
- **Added**: Long-polling worker agent and CLI commands (`master start`, `worker start`) to execute workloads remotely.
- **Added**: HTTP submission and polling for `train`, `predict`, `status`, and `observe` commands with configurable master URLs and path validation.
- **Docs**: Added the distributed architecture plan and a unit test covering the `StateStore` lifecycle to capture the new runtime design.
- **Packaging**: Bumped the project to version `0.4.0` and added FastAPI, Uvicorn, and Requests to runtime dependencies across packaging metadata.
- **Fixed**: Extended `make clean` to purge Python caches and bytecode so local runs start from a clean slate.

## [0.3.2] - 2025-10-17
- **Fixed**: Added explicit GitHub Packages `contents` and `packages` permissions to the Docker publish workflow so pushes succeed from CI.
- **Chore**: Normalized whitespace around the login step in the publish workflow for consistency with the rest of the file.

## [0.3.1] - 2025-10-17
- **Fixed**: Inserted a repository checkout step before authenticating with the GitHub Package Registry to prevent missing metadata during image publication.

## [0.3.0] - 2025-10-17
- **Added**: Deployment and observability step libraries plus revamped documentation that explains end-to-end rollout and monitoring flows.
- **Added**: Data preprocessing module with reusable steps for imputing missing values, encoding categoricals, and scaling numeric features.
- **Added**: Training backend abstraction with richer error handling and expanded unit tests for trainer, preprocessor, deployment, and monitoring paths.
- **Docs**: Published comprehensive training and preprocessing guides, and refreshed deployment/observability content.
- **Examples**: Shipped pipeline scripts, configuration templates, and sample data illustrating multi-step workflows.
- **Packaging**: Updated project metadata to `0.3.0` and aligned CLI wiring with the new module layout.

## [0.2.0] - 2025-10-15
- **Added**: Integrated XGBoost into the training stack and covered it with additional unit tests.
- **CI/CD**: Updated GitHub workflows for GHCR publishing, macOS prerequisites, and streamlined release automation.
- **Docs**: Revised README and getting-started material, including the project rename and new licensing details.
- **Packaging**: Raised the minimum supported Python version to 3.10 across metadata and documentation.

## [0.1.1] - 2025-10-13
- **Changed**: Renamed the CLI entry point and documentation to `make-mlops-easy` for consistency with the published package.
- **Docs**: Synced manuals, examples, and publishing guides with the updated command names.
- **CI/CD**: Removed a redundant step from the GitHub Packages workflow to simplify releases.
- **Packaging**: Bumped version markers to `0.1.1` across the codebase.

## [0.1.0] - 2025-10-13
- **Added**: Initial Easy MLOps CLI with `train`, `predict`, `status`, `observe`, and `init` commands orchestrating the full pipeline.
- **Added**: Core modules for configuration, preprocessing, training, deployment, and observability integrated into a unified pipeline API.
- **Docs**: Published the full documentation suite, including getting-started, pipeline walkthroughs, and publishing instructions.
- **Examples**: Provided runnable workflow scripts, sample datasets, and monitoring report artifacts.
- **CI/CD**: Set up Docker packaging, Makefile automation, and GitHub workflows for CI, docs, and publishing.
- **Packaging**: Delivered the initial project metadata, setup tooling, and unit tests establishing baseline quality.
