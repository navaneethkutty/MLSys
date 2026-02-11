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
	logSolutionLatency(solution)
	if err := writeSolution(outPath, solution); err != nil {
		fatal(err.Error())
	}
}

func logSolutionLatency(s OutputSolution) {
	total := 0.0
	for i, lat := range s.SubgraphLatencies {
		total += lat
		fmt.Fprintf(os.Stderr, "latency: subgraph=%d estimated_latency=%.4f\n", i, lat)
	}
	fmt.Fprintf(os.Stderr, "latency: total_estimated_latency=%.4f subgraphs=%d\n", total, len(s.SubgraphLatencies))
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
	nOps := len(p.OpTypes)
	s := OutputSolution{
		Subgraphs:         make([][]int, 0, nOps),
		Granularities:     make([][3]int64, 0, nOps),
		TensorsToRetain:   make([][]int, 0, nOps),
		TraversalOrders:   make([]*[]int64, 0, nOps),
		SubgraphLatencies: make([]float64, 0, nOps),
	}

	for op := 0; op < nOps; op++ {
		g := chooseGranularityForOp(p, op)
		lat := estimateSubgraphLatencySingleOp(p, op, g)

		s.Subgraphs = append(s.Subgraphs, []int{op})
		s.Granularities = append(s.Granularities, g)
		s.TensorsToRetain = append(s.TensorsToRetain, []int{})
		s.TraversalOrders = append(s.TraversalOrders, nil)
		s.SubgraphLatencies = append(s.SubgraphLatencies, lat)
	}
	return s
}

func chooseGranularityForOp(p InputProblem, op int) [3]int64 {
	outTensor := p.Outputs[op][0]
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
			if isMatMul(p.OpTypes[op]) && len(p.Inputs[op]) > 0 {
				lhs := p.Inputs[op][0]
				reduction := p.Widths[lhs]
				if reduction > 0 {
					k = minI64(reduction, 16)
				}
			}
			if fitsFastMemory(p, op, w, h, k) {
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

func estimateSubgraphLatencySingleOp(p InputProblem, op int, g [3]int64) float64 {
	w, h, k := g[0], g[1], g[2]
	outTensor := p.Outputs[op][0]
	outW := p.Widths[outTensor]
	outH := p.Heights[outTensor]
	tilesW := ceilDiv(outW, w)
	tilesH := ceilDiv(outH, h)
	splitK := int64(1)
	if isMatMul(p.OpTypes[op]) && len(p.Inputs[op]) > 0 {
		lhs := p.Inputs[op][0]
		reduction := p.Widths[lhs]
		splitK = ceilDiv(reduction, maxI64(1, k))
	}
	nSteps := maxI64(1, tilesW*tilesH*splitK)

	computePerStep := p.BaseCosts[op]
	memPerStep := float64(workingSetElementsForOp(p, op, w, h, k)) / p.SlowMemoryBandwidth
	stepLatency := math.Max(computePerStep, memPerStep)
	return float64(nSteps) * stepLatency
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
