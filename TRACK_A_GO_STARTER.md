# Track A Go Starter

This repository now includes a minimal Track A starter binary in Go:

```bash
go run ./cmd/mlsys <path_to_input.json> <path_to_output.json>
```

The starter:
- Parses the contest input JSON schema.
- Emits a valid-looking schedule JSON with one op per subgraph.
- Picks a granularity per op via a simple memory-fit heuristic.
- Uses empty `tensors_to_retain` and `null` traversal orders.

## Build a contest binary

```bash
CGO_ENABLED=0 go build -o mlsys ./cmd/mlsys
```

## Next steps to improve quality

1. Replace per-op scheduling with grouped subgraphs.
2. Add a better latency model matching your evaluator exactly.
3. Implement inter-subgraph retention heuristics.
4. Add traversal-order search when tiled.
