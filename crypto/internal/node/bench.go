package node

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/4uhn/Meta-AI-Hackathon/crypto/internal/he"
	"github.com/4uhn/Meta-AI-Hackathon/crypto/internal/protocol"
)

// Bench runs one full encrypted round through the file-based flow at payload size n and
// records times, sizes and precision; it does not pass or fail. The active set is the
// first party plus the last t−1 (2-of-3: op_a, op_c).
func Bench(root string, n int, parties protocol.Parties) (Result, error) {
	t0 := time.Now()
	cer, err := Ceremony(root, he.DefaultConfig, parties)
	if err != nil {
		return nil, err
	}
	ceremonyMs := since(t0)
	agg, err := Load(root, protocol.Aggregator)
	if err != nil {
		return nil, err
	}
	want := make([]float64, n)
	var enc []Result
	for i, op := range agg.Parties.Names() {
		c, err := Load(root, op)
		if err != nil {
			return nil, err
		}
		in := filepath.Join(root, "parties", op, "work", "update.f64")
		v, err := randomUpdate(in, n, 0.15+0.02*float64(i))
		if err != nil {
			return nil, err
		}
		for j := range v {
			want[j] += v[j]
		}
		r, err := Encrypt(c, 1, in)
		if err != nil {
			return nil, err
		}
		os.Remove(in)
		enc = append(enc, r)
	}
	ag, err := Aggregate(agg, 1)
	if err != nil {
		return nil, err
	}
	names := agg.Parties.Names()
	active := append([]string{names[0]}, names[len(names)-agg.Parties.Threshold+1:]...)
	if _, err := RequestDecrypt(agg, 1, active); err != nil {
		return nil, err
	}
	var shares []Result
	for _, op := range active {
		c, _ := Load(root, op)
		r, err := DecShare(c, 1)
		if err != nil {
			return nil, err
		}
		shares = append(shares, r)
	}
	dc, err := DecCombine(agg, 1, "")
	if err != nil {
		return nil, err
	}
	got, err := ReadF64(filepath.Join(root, "bus", "round_1", "decrypt", "agg.f64"))
	if err != nil {
		return nil, err
	}
	maxErr := 0.0
	for j := range want {
		maxErr = math.Max(maxErr, math.Abs(got[j]-want[j]))
	}
	slots := he.Slots(agg.Params)
	return Result{
		"spec":                 "T6",
		"n_values":             n,
		"n_ciphertexts":        (n + slots - 1) / slots,
		"bytes_per_submission": enc[0]["bytes"],
		"expansion_vs_fp16":    enc[0]["expansion_vs_fp16"],
		"ceremony_ms":          ceremonyMs,
		"encrypt":              enc,
		"aggregate":            ag,
		"dec_shares":           shares,
		"combine":              dc,
		"max_abs_err":          maxErr,
		"params":               cer["init"],
		"machine": Result{"goos": runtime.GOOS, "goarch": runtime.GOARCH, "num_cpu": runtime.NumCPU(),
			"workers": he.Workers, "go": runtime.Version()},
		"note": fmt.Sprintf("one round, %d submissions, decrypt with %v; timings include file I/O and verification", len(names), active),
	}, nil
}
