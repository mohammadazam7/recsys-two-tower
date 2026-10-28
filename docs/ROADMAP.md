# Roadmap

Each milestone lands on its own branch with tests and is merged when complete. Checked items are done.

- [x] **M1: Data** (Oct 26 to Oct 31). Interaction schema and validation, MovieLens loader
      with a schema-compatible synthetic fallback, ID mapping, temporal per-user split.
- [ ] **M2: Two-tower retrieval** (Nov 1 to Nov 8). PyTorch user and item towers, in-batch
      sampled softmax with log-Q correction, training loop with checkpointing, recall@k on
      the validation split.
- [ ] **M3: ANN index** (Nov 9 to Nov 14). FAISS flat and IVF indexes over item embeddings,
      recall versus exact search, latency per query.
- [ ] **M4: Ranking** (Nov 15 to Nov 21). Ranking model over retrieved candidates, NDCG@k
      and MAP@k, comparison against retrieval-only ordering.
- [ ] **M5: Serving** (Nov 22 to Nov 26). FastAPI endpoint for top-k recommendations, model
      and index loading, request latency measurement.
- [ ] **M6: v1.0** (Nov 27 to Nov 30). Published results with hardware, design notes,
      CHANGELOG and release.
