# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Two-tower retrieval model: MLP towers, in-batch sampled softmax with the log-Q
  correction, deterministic training, recall@k against a popularity baseline, and
  `examples/train_synthetic.py` to reproduce the README table.
- Milestone 1: data. Interaction schema validation, `IdMap` for contiguous user
  and item indices, a per-user temporal split with a minimum training history, a
  MovieLens loader and a schema-compatible synthetic generator with latent-factor
  preferences and Zipf item popularity.
- Project foundation: package layout, CI (ruff, mypy in strict mode, pytest on Python
  3.11 to 3.13), roadmap, issue and pull request templates.
