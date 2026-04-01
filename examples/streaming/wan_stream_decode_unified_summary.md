# Wan Streaming Decode Unified Summary

All ratios below use warmup-corrected results when available for compile mode.

## A14B

| Mode | Full decode (s) | Stream decode (s) | Stream slower than full |
|---|---:|---:|---:|
| eager, no cache | 6.0999 | 6.5216 | 6.91% |
| eager, conv2 cache | 6.0140 | 6.5143 | 8.32% |
| compile + warmup, no cache | 4.9511 | 6.4453 | 30.18% |
| compile + warmup, conv2 cache | 4.9488 | 6.4743 | 30.83% |

## TI2V-5B

| Mode | Full decode (s) | Stream decode (s) | Stream slower than full |
|---|---:|---:|---:|
| eager, no cache (4-case mean) | 17.3541 | 18.3333 | 5.65% |
| eager, no cache (1 case) | 17.5976 | 18.3218 | 4.12% |
| eager, conv2 cache (1 case) | 17.5976 | 18.3218 | 4.12% |
| compile + warmup, no cache (1 case) | 15.7413 | 18.4519 | 17.22% |
| compile + warmup, conv2 cache (1 case) | 15.7700 | 18.5473 | 17.61% |

## Readout

- The large percentage jump in compile mode comes mostly from full decode becoming much faster after compile and warmup.
- Stream decode gets little benefit from compile because the heavy modules are compiled, but the orchestration remains a Python loop with many small calls and state/cache transitions.
- conv2 cache itself adds only a very small extra time cost; the main overhead is the streaming decode structure.
