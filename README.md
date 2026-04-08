<h1 align="center">
LLMOps Course on Databricks
</h1>

## Practical information
- Weekly lectures on Wednesdays 16:00-18:00 CET.
- Weekly Q&A on Mondays 16:00-17:00 CET.
- Code for the lecture is shared before the lecture.
- Presentation and lecture materials are shared right after the lecture.
- Video of the lecture is uploaded within 24 hours after the lecture.

- Every week we set up a deliverable, and you implement it with your own dataset.
- To submit the deliverable, create a feature branch in that repository, and a PR to main branch. The code can be merged after we review & approve & CI pipeline runs successfully.
- The deliverables can be submitted with a delay (for example, lecture 1 & 2 together), but we expect you to finish all assignments for the course before the demo day.


## Set up your environment
In this course, we use serverless environment 4, which uses Python 3.12.
In our examples, we use UV. Check out the documentation on how to install it: https://docs.astral.sh/uv/getting-started/installation/

To create a new environment and create a lockfile, run:

```
uv sync --extra dev
```

## About This Project
Analytics and optimization utilities for LLM usage patterns.

## Overview

This package provides tools to:
- Collect and enrich LLM request logs
- Analyze cost patterns across models and categories
- Identify optimization opportunities
- Calculate model efficiency metrics
- Classify queries by category

## Progress

| Week | Date | Status | Deliverables | Notes |
| --- | --- | --- | --- | --- |
| 1 | 2026-03-18 | Done | Added logs to Unity Catalog table; Added cost comparison notebook | Table: `llmops_dev.logs.logs_20260201`, notebook: `notebooks/hw1_data_collection.py` |
| 2 | - | Planned | - | - |
| 3 | - | Planned | - | - |
| 4 | - | Planned | - | - |
