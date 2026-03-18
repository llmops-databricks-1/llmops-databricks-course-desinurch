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
This project provides insight about cost analytics for LLM usage and provides:
1. Cost validation across models
2. Cost analytics usage
3. Monitoring and anomaly detection
4. Find latest LLM usage trends and provide recommendation

## Progress

| Week | Date | Status | Deliverables | Notes |
| --- | --- | --- | --- | --- |
| 1 | 2026-03-18 | Done | Added logs to Unity Catalog table; Added cost comparison notebook with hardcoded actual cost | Table: `llmops_dev.logs.logs_20260201`, notebook: `notebooks/logs_data_ingestion.py` |
| 2 | - | Planned | - | - |
| 3 | - | Planned | - | - |
| 4 | - | Planned | - | - |
