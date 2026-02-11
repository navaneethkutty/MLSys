package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
)

type InputProblem struct {
	Widths              []int64   `json:"widths"`
	Heights             []int64   `json:"heights"`
	Inputs              [][]int   `json:"inputs"`
	Outputs             [][]int   `json:"outputs"`
	BaseCosts           []float64 `json:"base_costs"`
	OpTypes             []string  `json:"op_types"`
	FastMemoryCapacity  float64   `json:"fast_memory_capacity"`
	SlowMemoryBandwidth float64   `json:"slow_memory_bandwidth"`
	NativeGranularity   [2]int64  `json:"native_granularity"`
}

type OutputSolution struct {
	Subgraphs         [][]int    `json:"subgraphs"`
	Granularities     [][3]int64 `json:"granularities"`
	TensorsToRetain   [][]int    `json:"tensors_to_retain"`
	TraversalOrders   []*[]int64 `json:"traversal_orders"`
	SubgraphLatencies []float64  `json:"subgraph_latencies"`
}

func main() {
	if len(os.Args) != 3 {
		fatal("usage: ./mlsys <path_to_input.json> <path_to_output.json>")
	}
	inPath := os.Args[1]
	outPath := os.Args[2]

	problem, err := readProblem(inPath)
	if err != nil {
		fatal(err.Error())
	}
	if err := validateProblem(problem); err != nil {
		fatal(err.Error())
	}

	solution := buildBaselineSolution(problem)
	if err := validateSolution(problem, solution); err != nil {
		fatal(err.Error())
	}
	if err := writeSolution(outPath, solution); err != nil {
		fatal(err.Error())
	}
}

func readProblem(path string) (InputProblem, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return InputProblem{}, fmt.Errorf("read input: %w", err)
	}
	var p InputProblem
	if err := json.Unmarshal(data, &p); err != nil {
		return InputProblem{}, fmt.Errorf("parse input JSON: %w", err)
	}
	return p, nil
}

func validateProblem(p InputProblem) error {
	nOps := len(p.OpTypes)
	if nOps == 0 {
		return errors.New("problem has no operations")
	}
	if len(p.Inputs) != nOps || len(p.Outputs) != nOps || len(p.BaseCosts) != nOps {
		return errors.New("inputs/outputs/base_costs/op_types length mismatch")
	}
	if len(p.Widths) != len(p.Heights) {
		return errors.New("widths/heights length mismatch")
	}
	if p.SlowMemoryBandwidth <= 0 {
		return errors.New("slow_memory_bandwidth must be > 0")
	}
	if p.FastMemoryCapacity <= 0 {
		return errors.New("fast_memory_capacity must be > 0")
	}
	if p.NativeGranularity[0] <= 0 || p.NativeGranularity[1] <= 0 {
		return errors.New("native_granularity entries must be > 0")
	}
	for op := 0; op < nOps; op++ {
		for _, t := range p.Inputs[op] {
			if t < 0 || t >= len(p.Widths) {
				return fmt.Errorf("op %d input tensor index out of range: %d", op, t)
			}
		}
		for _, t := range p.Outputs[op] {
			if t < 0 || t >= len(p.Widths) {
				return fmt.Errorf("op %d output tensor index out of range: %d", op, t)
			}
		}
	}
	return nil
}

func buildBaselineSolution(p InputProblem) OutputSolution {
	groups := chooseGroupsByDP(p, 4)
	s := OutputSolution{
		Subgraphs:         make([][]int, 0, len(groups)),
		Granularities:     make([][3]int64, 0, len(groups)),
		TensorsToRetain:   make([][]int, 0, len(groups)),
		TraversalOrders:   make([]*[]int64, 0, len(groups)),
		SubgraphLatencies: make([]float64, 0, len(groups)),
	}

	for i, group := range groups {
		g := chooseGranularityForGroup(p, group)
		lat := estimateSubgraphLatencyForGroup(p, group, g)
		retain := chooseRetainedTensors(p, groups, i)

		s.Subgraphs = append(s.Subgraphs, group)
		s.Granularities = append(s.Granularities, g)
		s.TensorsToRetain = append(s.TensorsToRetain, retain)
		s.TraversalOrders = append(s.TraversalOrders, nil)
		s.SubgraphLatencies = append(s.SubgraphLatencies, lat)
	}
	return s
}

func chooseRetainedTensors(p InputProblem, groups [][]int, idx int) []int {
	if idx < 0 || idx >= len(groups)-1 {
		return []int{}
	}
	current := groups[idx]
	next := groups[idx+1]
	if len(current) == 0 || len(next) == 0 {
		return []int{}
	}

	nextInputs := make(map[int]bool)
	for _, op := range next {
		for _, t := range p.Inputs[op] {
			nextInputs[t] = true
		}
	}

	retain := make([]int, 0)
	seen := make(map[int]bool)
	for _, op := range current {
		for _, t := range p.Outputs[op] {
			if nextInputs[t] && !seen[t] {
				retain = append(retain, t)
				seen[t] = true
			}
		}
	}
	return retain
}

func chooseGroupsByDP(p InputProblem, maxGroupSize int) [][]int {
	n := len(p.OpTypes)
	dp := make([]float64, n+1)
	prev := make([]int, n+1)
	for i := range dp {
		dp[i] = math.Inf(1)
		prev[i] = -1
	}
	dp[0] = 0

	for end := 1; end <= n; end++ {
		for size := 1; size <= maxGroupSize; size++ {
			start := end - size
			if start < 0 {
				break
			}
			group := makeContiguousOps(start, end)
			g := chooseGranularityForGroup(p, group)
			cost := estimateSubgraphLatencyForGroup(p, group, g)
			if dp[start]+cost < dp[end] {
				dp[end] = dp[start] + cost
				prev[end] = start
			}
		}
	}

	if prev[n] == -1 {
		groups := make([][]int, 0, n)
		for op := 0; op < n; op++ {
			groups = append(groups, []int{op})
		}
		return groups
	}

	reversed := make([][]int, 0)
	for at := n; at > 0; at = prev[at] {
		start := prev[at]
		if start < 0 {
			break
		}
		reversed = append(reversed, makeContiguousOps(start, at))
	}

	groups := make([][]int, 0, len(reversed))
	for i := len(reversed) - 1; i >= 0; i-- {
		groups = append(groups, reversed[i])
	}
	return groups
}

func makeContiguousOps(start, end int) []int {
	group := make([]int, 0, end-start)
	for op := start; op < end; op++ {
		group = append(group, op)
	}
	return group
}

func chooseGranularityForGroup(p InputProblem, ops []int) [3]int64 {
	if len(ops) == 0 {
		return [3]int64{1, 1, 1}
	}
	outTensor := p.Outputs[ops[0]][0]
	for _, op := range ops {
		t := p.Outputs[op][0]
		if p.Widths[t]*p.Heights[t] > p.Widths[outTensor]*p.Heights[outTensor] {
			outTensor = t
		}
	}
	maxW := minI64(p.NativeGranularity[0], p.Widths[outTensor])
	maxH := minI64(p.NativeGranularity[1], p.Heights[outTensor])
	if maxW < 1 {
		maxW = 1
	}
	if maxH < 1 {
		maxH = 1
	}

	candidatesW := descendingPowersOfTwo(maxW)
	candidatesH := descendingPowersOfTwo(maxH)
	best := [3]int64{1, 1, 1}
	bestArea := int64(1)

	for _, w := range candidatesW {
		for _, h := range candidatesH {
			k := int64(1)
			for _, op := range ops {
				if isMatMul(p.OpTypes[op]) && len(p.Inputs[op]) > 0 {
					lhs := p.Inputs[op][0]
					reduction := p.Widths[lhs]
					if reduction > 0 {
						k = maxI64(k, minI64(reduction, 16))
					}
				}
			}
			if fitsFastMemoryGroup(p, ops, w, h, k) {
				area := w * h
				if area > bestArea {
					bestArea = area
					best = [3]int64{w, h, k}
				}
			}
		}
	}
	return best
}

func estimateSubgraphLatencyForGroup(p InputProblem, ops []int, g [3]int64) float64 {
	if len(ops) == 0 {
		return 0
	}
	w, h, k := g[0], g[1], g[2]
	outTensor := p.Outputs[ops[len(ops)-1]][0]
	tilesW := ceilDiv(p.Widths[outTensor], maxI64(1, w))
	tilesH := ceilDiv(p.Heights[outTensor], maxI64(1, h))
	splitK := int64(1)
	for _, op := range ops {
		if isMatMul(p.OpTypes[op]) && len(p.Inputs[op]) > 0 {
			lhs := p.Inputs[op][0]
			reduction := p.Widths[lhs]
			splitK = maxI64(splitK, ceilDiv(reduction, maxI64(1, k)))
		}
	}
	nSteps := maxI64(1, tilesW*tilesH*splitK)

	computePerStep := 0.0
	for _, op := range ops {
		computePerStep += p.BaseCosts[op]
	}

	boundaryInputs, boundaryOutputs := boundaryTensorsForGroup(p, ops)
	boundaryElements := int64(0)
	for t := range boundaryInputs {
		boundaryElements += tileElementsForTensor(p, t, w, h)
	}
	for t := range boundaryOutputs {
		boundaryElements += tileElementsForTensor(p, t, w, h)
	}
	memPerStep := float64(maxI64(1, boundaryElements)) / p.SlowMemoryBandwidth
	if k > 1 {
		memPerStep *= 0.9
	}
	stepLatency := math.Max(computePerStep, memPerStep)
	return float64(nSteps) * stepLatency
}

func boundaryTensorsForGroup(p InputProblem, ops []int) (map[int]bool, map[int]bool) {
	internalOutputs := make(map[int]bool)
	for _, op := range ops {
		for _, t := range p.Outputs[op] {
			internalOutputs[t] = true
		}
	}

	boundaryInputs := make(map[int]bool)
	for _, op := range ops {
		for _, t := range p.Inputs[op] {
			if !internalOutputs[t] {
				boundaryInputs[t] = true
			}
		}
	}

	consumedInside := make(map[int]bool)
	for _, op := range ops {
		for _, t := range p.Inputs[op] {
			consumedInside[t] = true
		}
	}

	boundaryOutputs := make(map[int]bool)
	for _, op := range ops {
		for _, t := range p.Outputs[op] {
			if !consumedInside[t] {
				boundaryOutputs[t] = true
			}
		}
	}

	return boundaryInputs, boundaryOutputs
}

func tileElementsForTensor(p InputProblem, tensor int, w, h int64) int64 {
	tileW := minI64(maxI64(1, w), p.Widths[tensor])
	tileH := minI64(maxI64(1, h), p.Heights[tensor])
	return maxI64(1, tileW*tileH)
}

func validateSolution(p InputProblem, s OutputSolution) error {
	n := len(s.Subgraphs)
	if n == 0 {
		return errors.New("solution has no subgraphs")
	}
	if len(s.Granularities) != n || len(s.TensorsToRetain) != n || len(s.TraversalOrders) != n || len(s.SubgraphLatencies) != n {
		return errors.New("solution parallel list length mismatch")
	}

	covered := make([]int, len(p.OpTypes))
	for i := 0; i < n; i++ {
		if len(s.Subgraphs[i]) == 0 {
			return fmt.Errorf("subgraph %d has no ops", i)
		}
		g := s.Granularities[i]
		if g[0] <= 0 || g[1] <= 0 || g[2] <= 0 {
			return fmt.Errorf("subgraph %d has invalid granularity", i)
		}
		if s.SubgraphLatencies[i] < 0 {
			return fmt.Errorf("subgraph %d has negative latency", i)
		}

		for _, op := range s.Subgraphs[i] {
			if op < 0 || op >= len(p.OpTypes) {
				return fmt.Errorf("subgraph %d references invalid op index %d", i, op)
			}
			covered[op]++
		}
		if !fitsFastMemoryGroup(p, s.Subgraphs[i], g[0], g[1], g[2]) {
			return fmt.Errorf("subgraph %d violates fast memory capacity", i)
		}

		for _, t := range s.TensorsToRetain[i] {
			if t < 0 || t >= len(p.Widths) {
				return fmt.Errorf("subgraph %d retains invalid tensor %d", i, t)
			}
		}
	}

	for op, c := range covered {
		if c != 1 {
			return fmt.Errorf("operation %d must be scheduled exactly once (found %d)", op, c)
		}
	}
	return nil
}

func fitsFastMemoryGroup(p InputProblem, ops []int, w, h, k int64) bool {
	required := workingSetElementsForGroup(p, ops, w, h, k)
	return float64(required) <= p.FastMemoryCapacity
}

func workingSetElementsForGroup(p InputProblem, ops []int, w, h, k int64) int64 {
	if len(ops) == 0 {
		return 0
	}
	internalOutputs := make(map[int]bool)
	for _, op := range ops {
		for _, t := range p.Outputs[op] {
			internalOutputs[t] = true
		}
	}

	var boundaryIn int64
	for _, op := range ops {
		if isMatMul(p.OpTypes[op]) {
			if len(p.Inputs[op]) > 0 {
				if !internalOutputs[p.Inputs[op][0]] {
					boundaryIn += h * maxI64(1, k)
				}
			}
			if len(p.Inputs[op]) > 1 {
				if !internalOutputs[p.Inputs[op][1]] {
					boundaryIn += w * maxI64(1, k)
				}
			}
			continue
		}
		for _, t := range p.Inputs[op] {
			if !internalOutputs[t] {
				boundaryIn += w * h
			}
		}
	}

	lastOp := ops[len(ops)-1]
	boundaryOut := w * h * maxI64(1, int64(len(p.Outputs[lastOp])))
	return boundaryIn + boundaryOut
}

func fitsFastMemory(p InputProblem, op int, w, h, k int64) bool {
	required := workingSetElementsForOp(p, op, w, h, k)
	return float64(required) <= p.FastMemoryCapacity
}

func workingSetElementsForOp(p InputProblem, op int, w, h, k int64) int64 {
	if isMatMul(p.OpTypes[op]) {
		lhs := h * maxI64(1, k)
		rhs := w * maxI64(1, k)
		out := w * h * maxI64(1, int64(len(p.Outputs[op])))
		return lhs + rhs + out
	}
	in := w * h * maxI64(1, int64(len(p.Inputs[op])))
	out := w * h * maxI64(1, int64(len(p.Outputs[op])))
	return in + out
}

func writeSolution(path string, s OutputSolution) error {
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal solution: %w", err)
	}
	if err := os.WriteFile(path, append(data, '\n'), 0o644); err != nil {
		return fmt.Errorf("write solution: %w", err)
	}
	return nil
}

func descendingPowersOfTwo(max int64) []int64 {
	vals := make([]int64, 0)
	v := int64(1)
	for v*2 <= max {
		v *= 2
	}
	for v >= 1 {
		vals = append(vals, v)
		v /= 2
	}
	if len(vals) == 0 {
		return []int64{1}
	}
	return vals
}

func isMatMul(opType string) bool {
	return opType == "MatMul" || opType == "matmul"
}

func ceilDiv(a, b int64) int64 {
	if b <= 0 {
		return 0
	}
	return (a + b - 1) / b
}

func minI64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

func maxI64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func fatal(msg string) {
	fmt.Fprintln(os.Stderr, "error:", msg)
	os.Exit(1)
}
