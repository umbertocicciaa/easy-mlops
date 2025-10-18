# Welcome

Make MLOps Easy packages the operational lifecycle of a tabular machine learning project into a reusable framework. It combines a modular Python pipeline with a distributed runtime and a batteries-included CLI so you can train, deploy, and monitor models with minimal ceremony.

## What you get

- **Composable pipeline** – `MLOpsPipeline` chains configuration, preprocessing, training, deployment, and observability into a single call that you can drop into notebooks, scripts, or services.
- **Distributed runtime** – a FastAPI master coordinates long-running workflows across worker agents; the `make-mlops-easy` CLI drives the experience.
- **Artifact-first deployments** – every training run emits a deterministic directory that contains the model, fitted preprocessor, metadata, logs, and optionally an executable prediction helper.
- **Extensibility hooks** – registries let you register custom preprocessing steps, training backends, deployment steps, or observability sinks without modifying the core library.
- **Curated tooling** – MkDocs documentation, Makefile tasks, examples, and CI pipelines make teams productive from day one.

```
┌──────────────┐  submit/poll   ┌──────────────┐
│   CLI user   │ ─────────────▶ │   Master API │
└──────────────┘                └──────┬───────┘
                                        │ assign
                                        ▼
                                  ┌──────────────┐
                                  │  Worker pool │
                                  └──────┬───────┘
                                         │ runs
                                         ▼
                                 ┌──────────────┐
                                 │ MLOpsPipeline│
                                 └──────────────┘
```

## Documentation map

- **Getting Started** – install the project, launch the runtime, and walk through the CLI (`quick start`) or dive straight into the pipeline from Python (`CLI reference`).
- **Pipeline** – architecture overviews plus deep dives into preprocessing, training, deployment, and observability. Learn how to reconfigure or extend each stage.
- **Development** – set up a contribution environment, understand the automated checks, and explore CI/CD pipelines.

If you are new to the project, start with [Quick Start](getting-started.md) and then explore the [Pipeline Architecture](pipeline.md) to understand how the moving pieces fit together. The examples under `examples/` mirror the documentation, so you can follow along with runnable scripts.
