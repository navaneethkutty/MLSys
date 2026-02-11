# Track A Go Starter

This repository now includes a minimal Track A starter binary in Go:

```bash
go run ./cmd/mlsys <path_to_input.json> <path_to_output.json>
```

The starter:
- Parses the contest input JSON schema.
- Emits a valid-looking schedule JSON with contiguous grouped subgraphs chosen by a small DP pass.
- Picks a granularity per op via a simple memory-fit heuristic.
- Uses immediate-next-use `tensors_to_retain` heuristics and `null` traversal orders.

## Build a contest binary

```bash
CGO_ENABLED=0 go build -o mlsys ./cmd/mlsys
```

## Next steps to improve quality

1. Add a stronger latency model matching your evaluator exactly.
2. Add global state-aware retention (beyond immediate next subgraph).
3. Add DAG-aware grouping beyond contiguous windows.
4. Add traversal-order search when tiled.
