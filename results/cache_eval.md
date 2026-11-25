# ModelMesh Response Cache Evaluation

- Provider: `mock`
- Candidate: `mock-fast@mock` (mock-fast)
- Generated: 2026-11-25T10:35:56+00:00
- Phases: A, B, C, D
- Golden prompts: 30
- Paraphrases: 30
- Near misses: 15
- Thresholds: 0.80, 0.90, 0.92, 0.95
- Cache: isolated temporary SQLite DB, clean for this run

## Phase A — cold pass

| Metric | Value |
|---|---|
| Turns | 30 |
| Hits | 0 (0.0%) |
| Avg latency | 0.0 ms |
| Total cost | $0.000095 |
| Tokens saved | 0 |

## Phase B — exact repeat pass

| Metric | Value |
|---|---|
| Turns | 30 |
| Hits | 30 (100.0%) |
| Exact hits | 30 |
| Avg latency | 0.1 ms |
| Total cost | $0.000000 |
| Tokens saved | 7,600 |

## Phase C — paraphrase pass

| Semantic | Threshold | Hits | Semantic hits | Semantic hit rate | Avg latency (ms) | Total cost | Tokens saved |
|---|---|---|---|---|---|---|---|
| disabled | — | 0 | 0 | 0.0% | 0.1 | $0.000096 | 0 |
| enabled | 0.80 | 3 | 3 | 10.0% | 3.1 | $0.000086 | 724 |
| enabled | 0.90 | 0 | 0 | 0.0% | 3.0 | $0.000096 | 0 |
| enabled | 0.92 | 0 | 0 | 0.0% | 3.3 | $0.000096 | 0 |
| enabled | 0.95 | 0 | 0 | 0.0% | 3.2 | $0.000096 | 0 |

## Phase D — near-miss pass

| Semantic | Threshold | Hits | False hits | False-hit rate | Avg latency (ms) | Total cost | Tokens saved |
|---|---|---|---|---|---|---|---|
| disabled | — | 0 | 0 | 0.0% | 0.5 | $0.000048 | 0 |
| enabled | 0.80 | 6 | 6 | 40.0% | 3.6 | $0.000026 | 1,714 |
| enabled | 0.90 | 5 | 5 | 33.3% | 5.9 | $0.000030 | 1,433 |
| enabled | 0.92 | 3 | 3 | 20.0% | 3.3 | $0.000036 | 927 |
| enabled | 0.95 | 1 | 1 | 6.7% | 3.3 | $0.000044 | 327 |

## Threshold sweep

| Threshold | Paraphrase semantic hit rate | Near-miss false-hit rate |
|---|---|---|
| 0.80 | 10.0% | 40.0% |
| 0.90 | 0.0% | 33.3% |
| 0.92 | 0.0% | 20.0% |
| 0.95 | 0.0% | 6.7% |

## Findings

- At the default threshold (0.92), semantic matching served 0/30 paraphrases and produced 3/15 false hits.
- Near misses that slipped through at 0.92: `near-11`, `near-13`, `near-14`.
- Paraphrases still missed at 0.92: 30/30.
- The hashing embedder is lexical: long prompts can exceed a high threshold after a single entity swap, while short natural paraphrases can score below 0.80. The numeric and negation gates stop changed numbers and explicit negations, but not entity swaps.
- Semantic matching is disabled by default in `config/cache.yaml`; these numbers are a reason to keep it opt-in until a neural embedder is configured.

## Reproduce

```bash
python scripts/cache_eval.py --provider mock
# live mode (requires GROQ_API_KEY in the environment):
python scripts/cache_eval.py --provider live --yes-i-know-this-spends-quota
```

## Notes

- Phase A uses `bypass_cache=True` so every turn is a guaranteed miss; semantic matching is enabled during this phase so each stored answer also gets an embedding.
- Phases C and D use a read-only cache wrapper: a miss cannot add an entry, so every threshold is measured against the same warmed entries.
- The dependency-free `HashingEmbedder` is used; the stopword list and feature weights are unchanged from Step 6.
- A near-miss hit is counted as a false hit because every near-miss fixture has `same_meaning: false`.
- Mock mode: the mock adapter embeds the last user prompt in its answer. Consistency checks failed 0 time(s).
