# llmscope benchmark (simulated model)

## Response cache, 600 requests, 28 intents (12 standalone + 16 hard-negative siblings)

| config | hit rate | exact | semantic | wrong hits | wrong % of hits | cost USD | mean latency ms | p95 latency ms | gateway overhead p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| no cache | 0.0% | 0 | 0 | 0 | 0.0% | 0.31142 | 1907 | 2831 | 0.01 |
| exact only | 77.3% | 464 | 0 | 0 | 0.0% | 0.07227 | 441 | 2506 | 0.01 |
| exact + semantic >= 0.8 (hashing) | 90.2% | 264 | 277 | 31 | 5.7% | 0.03168 | 193 | 1984 | 0.05 |
| exact + semantic >= 0.85 (hashing) | 87.3% | 302 | 222 | 2 | 0.4% | 0.04028 | 246 | 2095 | 0.06 |
| exact + semantic >= 0.9 (hashing) | 85.2% | 318 | 193 | 0 | 0.0% | 0.04854 | 296 | 2402 | 0.06 |
| exact + semantic >= 0.95 (hashing) | 81.8% | 371 | 120 | 0 | 0.0% | 0.06015 | 366 | 2497 | 0.06 |
| exact + semantic >= 0.85 (hashing), NO number guard | 87.8% | 299 | 228 | 8 | 1.5% | 0.0382 | 234 | 2001 | 0.05 |
| exact + semantic >= 0.8 (fastembed) | 96.0% | 151 | 425 | 65 | 11.3% | 0.0137 | 85 | 4 | 3.61 |
| exact + semantic >= 0.85 (fastembed) | 95.7% | 190 | 384 | 65 | 11.3% | 0.0145 | 90 | 4 | 3.57 |
| exact + semantic >= 0.9 (fastembed) | 93.7% | 230 | 332 | 36 | 6.4% | 0.02027 | 125 | 1286 | 3.56 |
| exact + semantic >= 0.95 (fastembed) | 92.2% | 257 | 296 | 0 | 0.0% | 0.02559 | 158 | 1728 | 3.49 |
| exact + semantic >= 0.85 (fastembed), NO number guard | 96.2% | 175 | 402 | 119 | 20.6% | 0.01316 | 82 | 4 | 3.52 |

## Context compaction, 20 conversations x 14 turns

| history budget (tokens) | input tokens | cost USD | TTFT p50 ms | TTFT p95 ms | calls compacted | tokens saved |
|---|---|---|---|---|---|---|
| none | 571779 (100%) | 0.3075 | 760 | 1243 | 0/280 | 0 |
| 1500 | 248728 (44%) | 0.2106 | 476 | 569 | 177/280 | 323051 |
| 800 | 177340 (31%) | 0.1892 | 399 | 434 | 223/280 | 394439 |

