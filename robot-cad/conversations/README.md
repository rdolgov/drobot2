# Engineering conversation records

This folder preserves reviewable records of important hardware, CAD, and
simulation decisions that were developed through a Codex conversation.
These records supplement the topic-owning files under `docs/`, `specs/`,
`hardware/`, and `simulation/`; they do not replace those sources of truth.

## Record format

For future records:

1. use `YYYY-MM-DD-short-topic.md`;
2. state the question, source inputs, measured results, and conclusion;
3. distinguish hardware observations, simulator measurements, and inference;
4. link the editable source and generated artifacts;
5. include exact reproduction and validation commands;
6. place durable review images under `reviews/` so Git LFS owns them;
7. embed those images with repository-relative Markdown paths;
8. list unresolved questions and the next decision gate;
9. add the record to the index below.

Do not commit transient JSON logs, checkpoints, caches, or screenshots from
ignored output directories. Copy only intentional, reviewed evidence into
`reviews/`.

## Index

| Date | Record | Outcome |
| --- | --- | --- |
| 2026-07-28 | [One-leg wall-mounted Isaac range revisit](2026-07-28-one-leg-wall-isaac-revisit.md) | Sagittal joints and drives passed isolated and full-asset probes; inward abduction was limited by the flush wall, while the stair gate remained blocked by loaded support dynamics. |
