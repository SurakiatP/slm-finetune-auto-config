# SLM Finetune Auto Config

Task-agnostic SLM fine-tuning system for synthetic-data-assisted training, manual hyperparameter configuration, auto hyperparameter tuning, evaluation dashboards, and downloadable model artifacts.

Read `AGENT_RULES.md` first for project intent and `TASK_CHECKPOINT.md` for the latest implementation checkpoint.

## Current Structure

```text
configs/              Reusable model, training, tuning, and eval config templates.
data/                 Local raw, processed, synthetic, and sample data.
runs/                 Reproducible per-run outputs, configs, metrics, logs, and artifacts.
src/slm_auto_config/  Python package for the project pipeline.
tests/                Unit and integration tests.
```

## Core Design

Task differences live in `src/slm_auto_config/tasks/` through task adapters. Shared pipeline code should call the task registry instead of branching directly on task types.
