package node

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"

	"github.com/4uhn/Meta-AI-Hackathon/crypto/internal/he"
	"github.com/4uhn/Meta-AI-Hackathon/crypto/internal/protocol"
)

// ScalingConfigs are the consortium shapes (N, t) the scaling table measures.
var ScalingConfigs = [][2]int{{3, 2}, {4, 3}, {5, 3}, {5, 4}}

// Scaling runs the selftest and the benchmark for each (N, t) at payload size n, each in
// its own run directory under tmp (deleted afterwards), and collects one row per shape.
func Scaling(tmp string, n int, configs [][2]int) (Result, error) {
	names := []string{"op_a", "op_b", "op_c", "op_d", "op_e", "op_f", "op_g"}
	var rows []Result
	for _, cfg := range configs {
		if cfg[0] > len(names) {
			return nil, fmt.Errorf("%d parties: at most %d supported", cfg[0], len(names))
		}
		parties, err := protocol.NewParties(names[:cfg[0]], cfg[1])
		if err != nil {
			return nil, err
		}
		row, err := scalingRow(tmp, n, parties)
		if err != nil {
			return nil, fmt.Errorf("%d-of-%d: %w", cfg[1], cfg[0], err)
		}
		rows = append(rows, row)
	}
	return Result{
		"n_values": n,
		"rows":     rows,
		"machine": Result{"goos": runtime.GOOS, "goarch": runtime.GOARCH, "num_cpu": runtime.NumCPU(),
			"workers": he.Workers, "go": runtime.Version()},
		"note": "per shape: selftest (one round per t-subset, all attack checks) and bench (one round); " +
			"crypto-only times are medians over the selftest rounds; max_abs_err is the max over those rounds",
	}, nil
}

func scalingRow(tmp string, n int, parties protocol.Parties) (Result, error) {
	N, t := parties.N(), parties.Threshold
	dir := filepath.Join(tmp, fmt.Sprintf("selftest_%dof%d", t, N))
	st, err := Selftest(filepath.Join(dir, "run"), n, parties)
	os.RemoveAll(dir)
	if err != nil {
		return nil, err
	}
	dir = filepath.Join(tmp, fmt.Sprintf("bench_%dof%d", t, N))
	bn, err := Bench(filepath.Join(dir, "run"), n, parties)
	os.RemoveAll(dir)
	if err != nil {
		return nil, err
	}

	w := st["ceremony"].(Result)["init"].(Result)["flood_log_width"].(int)
	var enc, agg, share, comb, maxErr, partial []float64
	for _, r := range st["timings"].([]Result) {
		enc = append(enc, r["encrypt_ms"].([]float64)...)
		agg = append(agg, r["aggregate_ms"].(float64))
		share = append(share, r["share_ms"].([]float64)...)
		comb = append(comb, r["combine_ms"].(float64))
		maxErr = append(maxErr, r["max_abs_err"].(float64))
		partial = append(partial, r["partial_rel_err"].(float64))
	}
	checks := st["checks"].([]Check)
	passed := 0
	for _, c := range checks {
		if c.OK {
			passed++
		}
	}
	shares := bn["dec_shares"].([]Result)
	return Result{
		"n_parties":             N,
		"threshold":             t,
		"min_contributors":      parties.MinContributors(),
		"tolerated_offline":     N - t,
		"flood_log_width":       w,
		"max_abs_value":         he.MaxAbs(N),
		"max_abs_err":           slices.Max(maxErr),
		"max_abs_err_tolerance": ErrTolerance(w, t),
		"partial_rel_err_min":   slices.Min(partial),
		"selftest_ok_all":       st["ok_all"],
		"selftest_passed":       passed,
		"selftest_checks":       checks,
		"selftest_rounds":       len(maxErr),
		"encrypt_ms_median":     median(enc),
		"aggregate_ms_median":   median(agg),
		"dec_share_ms_median":   median(share),
		"combine_ms_median":     median(comb),
		"verify_ms_bench":       shares[0]["verify_ms"],
		"bytes_per_submission":  bn["bytes_per_submission"],
		"bytes_aggregate":       bn["aggregate"].(Result)["bytes"],
		"bytes_dec_share":       shares[0]["bytes"],
		"bench_max_abs_err":     bn["max_abs_err"],
		"n_ciphertexts":         bn["n_ciphertexts"],
	}, nil
}

func median(xs []float64) float64 {
	s := slices.Sorted(slices.Values(xs))
	if len(s) == 0 {
		return math.NaN()
	}
	if len(s)%2 == 1 {
		return s[len(s)/2]
	}
	return (s[len(s)/2-1] + s[len(s)/2]) / 2
}

// ScalingTable renders the rows for humans.
func ScalingTable(res Result) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Threshold CKKS scaling at %d values per submission (crypto-only medians, %v cores)\n\n",
		res["n_values"], res["machine"].(Result)["num_cpu"])
	fmt.Fprintf(&b, "%-6s %-5s %-7s %-6s %-11s %-11s %-11s %-7s %-8s %-9s %-8s %-9s %-10s\n",
		"t-of-N", "minC", "offline", "flood", "max err", "tolerance", "(t-1) rel", "checks", "enc ms", "agg ms", "share ms", "comb ms", "MB/sub")
	for _, r := range res["rows"].([]Result) {
		fmt.Fprintf(&b, "%-6s %-5d %-7d 2^%-4d %-11.2e %-11.2e %-11.3g %-7s %-8.0f %-9.1f %-8.0f %-9.0f %-10.2f\n",
			fmt.Sprintf("%d-of-%d", r["threshold"], r["n_parties"]), r["min_contributors"], r["tolerated_offline"],
			r["flood_log_width"], r["max_abs_err"], r["max_abs_err_tolerance"], r["partial_rel_err_min"],
			fmt.Sprintf("%d/%d", r["selftest_passed"], len(r["selftest_checks"].([]Check))),
			r["encrypt_ms_median"], r["aggregate_ms_median"], r["dec_share_ms_median"], r["combine_ms_median"],
			float64(r["bytes_per_submission"].(int))/1e6)
	}
	return b.String()
}
