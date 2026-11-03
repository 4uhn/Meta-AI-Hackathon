package node

import (
	"fmt"
	"math"
	mrand "math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"

	"github.com/4uhn/Meta-AI-Hackathon/crypto/internal/he"
	"github.com/4uhn/Meta-AI-Hackathon/crypto/internal/protocol"
)

// Check is one named pass/fail line of the selftest.
type Check struct {
	Name   string `json:"name"`
	OK     bool   `json:"ok"`
	Detail string `json:"detail"`
}

type tester struct {
	root   string
	checks []Check
	parts  []string
}

func (t *tester) check(name string, ok bool, format string, a ...any) {
	t.checks = append(t.checks, Check{name, ok, fmt.Sprintf(format, a...)})
}

func (t *tester) as(actor string) *Ctx {
	c, err := Load(t.root, actor)
	if err != nil {
		panic(fmt.Errorf("load %s: %w", actor, err))
	}
	return c
}

// must panics on error. The selftest recovers and reports the failure.
func must(r Result, err error) Result {
	if err != nil {
		panic(err)
	}
	return r
}

// Ceremony runs the dealer-free key setup for all parties in root.
func Ceremony(root string, cfg he.Config, parties protocol.Parties) (Result, error) {
	init, err := Init(root, cfg, parties)
	if err != nil {
		return nil, err
	}
	agg, err := Load(root, protocol.Aggregator)
	if err != nil {
		return nil, err
	}
	steps := []func(*Ctx) (Result, error){Keygen, PKShare, ShamirDeal}
	for _, step := range steps {
		for _, p := range agg.Parties.Names() {
			c, err := Load(root, p)
			if err != nil {
				return nil, err
			}
			if _, err := step(c); err != nil {
				return nil, fmt.Errorf("%s: %w", p, err)
			}
		}
	}
	for _, p := range agg.Parties.Names() {
		c, _ := Load(root, p)
		if _, err := ShamirAgg(c); err != nil {
			return nil, fmt.Errorf("%s: %w", p, err)
		}
	}
	pk, err := PKCombine(agg)
	if err != nil {
		return nil, err
	}
	return Result{"init": init, "pk": pk}, nil
}

// randomUpdate writes a synthetic update: n−1 values ~ N(0, 0.02²)·w plus the weight w.
func randomUpdate(path string, n int, w float64) ([]float64, error) {
	v := make([]float64, n)
	for i := 0; i < n-1; i++ {
		v[i] = w * 0.02 * mrand.NormFloat64()
	}
	v[n-1] = w
	return v, WriteF64(path, v, 0o600)
}

// roundSpec describes one selftest round.
type roundSpec struct {
	submit []string
	active []string
}

// selftestWeights are the synthetic submission weights, by party index.
var selftestWeights = []float64{0.18, 0.17, 0.19, 0.16, 0.20, 0.15, 0.21}

// combinations returns every k-subset of xs, in lexicographic order of indices.
func combinations(xs []string, k int) [][]string {
	var out [][]string
	var rec func(start int, cur []string)
	rec = func(start int, cur []string) {
		if len(cur) == k {
			out = append(out, append([]string(nil), cur...))
			return
		}
		for i := start; i < len(xs); i++ {
			rec(i+1, append(cur, xs[i]))
		}
	}
	rec(0, nil)
	return out
}

// ErrTolerance is the selftest bound on the max abs decryption error: 1e-6 for 2-of-3,
// scaled with the standard deviation of the sum of t floods of width 2^w (∝ 2^w·√t).
func ErrTolerance(w, t int) float64 {
	return 1e-6 * math.Exp2(float64(w-85)) * math.Sqrt(float64(t)/2)
}

// Selftest runs the full protocol on synthetic data in a fresh run directory and checks
// its security and correctness properties. n is the payload length, weight slot included.
// Every t-subset of the parties decrypts one round; fewer than t shares must give noise.
func Selftest(root string, n int, parties protocol.Parties) (res Result, err error) {
	t := &tester{root: root}
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("selftest aborted: %v", r)
		}
	}()

	cer, err := Ceremony(root, he.DefaultConfig, parties)
	if err != nil {
		return nil, err
	}
	aggC := t.as(protocol.Aggregator)
	t.parts = aggC.Parties.Names()
	thr := aggC.Parties.Threshold
	w, err := he.FloodLogWidth(aggC.Params, aggC.Parties.N())
	if err != nil {
		return nil, err
	}
	tol := ErrTolerance(w, thr)

	// Isolation: the aggregator cannot open any party's private files, nor one party another's.
	_, e1 := aggC.Run.Path("parties", "op_a", "secret", "tsk.bin")
	_, e2 := t.as("op_b").Run.Path("parties", "op_a", "secret", "tsk.bin")
	_, e3 := t.as("op_b").Run.Path("bus", "private", "op_a", "from_op_c.bin")
	t.check("isolation", e1 != nil && e2 != nil && e3 != nil, "aggregator→op_a/secret: %v; op_b→op_a/secret: %v; op_b→op_a inbox: %v", e1 != nil, e2 != nil, e3 != nil)
	_, skErr := os.Stat(filepath.Join(root, "parties", "op_a", "secret", "sk.bin"))
	t.check("sk erased after setup", os.IsNotExist(skErr), "only tsk.bin remains in each secret/ dir")

	weights := map[string]float64{}
	for i, op := range t.parts {
		weights[op] = selftestWeights[i]
	}
	// One round per t-subset, so every way of having N−t parties offline at decryption is
	// exercised. 2-of-3: {a,b}, {a,c}, {b,c} (op_a offline in round 3).
	var rounds []roundSpec
	for _, set := range combinations(t.parts, thr) {
		rounds = append(rounds, roundSpec{t.parts, set})
	}
	var timings []Result
	for i, rs := range rounds {
		r := i + 1
		want := make([]float64, n)
		var encMs []float64
		for _, op := range rs.submit {
			c := t.as(op)
			in := filepath.Join(root, "parties", op, "work", fmt.Sprintf("update_r%d.f64", r))
			v, err := randomUpdate(in, n, weights[op])
			if err != nil {
				return nil, err
			}
			for j := range v {
				want[j] += v[j]
			}
			er := must(Encrypt(c, r, in))
			encMs = append(encMs, er["encrypt_ms"].(float64))
			os.Remove(in) // the plaintext update never outlives the operator's encrypt step
		}
		ag := must(Aggregate(aggC, r))
		must(RequestDecrypt(aggC, r, rs.active))
		var shareMs []float64
		for _, op := range rs.active {
			sr := must(DecShare(t.as(op), r))
			shareMs = append(shareMs, sr["share_ms"].(float64))
		}
		dc := must(DecCombine(aggC, r, ""))
		got, err := ReadF64(filepath.Join(root, "bus", fmt.Sprintf("round_%d", r), "decrypt", "agg.f64"))
		if err != nil {
			return nil, err
		}
		maxErr := 0.0
		for j := range want {
			maxErr = math.Max(maxErr, math.Abs(got[j]-want[j]))
		}
		t.check(fmt.Sprintf("round %d: submit %v, decrypt with %v", r, rs.submit, rs.active), maxErr <= tol && maxErr <= 1e-5, "max abs err %.2e", maxErr)
		solo := must(DecCombine(aggC, r, rs.active[0]))
		rel := solo["rel_err_vs_true"].(float64)
		t.check(fmt.Sprintf("round %d: %s's share alone is noise", r, rs.active[0]), rel >= 0.5, "relative error %.3g", rel)
		partial := rel
		if thr > 2 {
			use := rs.active[:thr-1]
			pr := must(DecCombine(aggC, r, strings.Join(use, ",")))
			partial = pr["rel_err_vs_true"].(float64)
			t.check(fmt.Sprintf("round %d: %d of %d shares %v are noise", r, thr-1, thr, use), partial >= 0.5, "relative error %.3g", partial)
		}
		timings = append(timings, Result{"round": r, "encrypt_ms": encMs, "aggregate_ms": ag["sum_ms"], "share_ms": shareMs,
			"combine_ms": dc["combine_ms"], "max_abs_err": maxErr, "partial_rel_err": partial})
	}

	// Attacks the protocol must refuse.

	// One share per round: a second decryption of round 1, even with a new active set.
	_, err = DecShare(t.as("op_a"), 1)
	t.check("one-shot: same request again", err != nil, "%v", err)
	rf := filepath.Join(root, "bus", "round_1", "decrypt", "request.json")
	var q1 protocol.DecryptRequest
	_ = protocol.ReadJSON(rf, &q1)
	// The last t parties; since 2t > N its first member also shared in round 1.
	q1.Active = t.parts[len(t.parts)-thr:]
	_ = protocol.WriteJSON(rf, q1, 0o644)
	_, err = DecShare(t.as(q1.Active[0]), 1)
	t.check("one-shot: new active set, same round", err != nil, "%v", err)

	// Too few submit (2-of-3: op_c skips). A sum of k < t+1 would let t−1 colluding
	// contributors subtract their own updates and read an individual one (2-of-3: op_a
	// and op_b each read the other's), so the round must not be decrypted at all.
	minC := aggC.Parties.MinContributors()
	submitters, skipped := t.parts[:minC-1], t.parts[minC-1:]
	r := len(rounds) + 1
	for _, op := range submitters {
		in := filepath.Join(root, "parties", op, "work", "u.f64")
		if _, err := randomUpdate(in, n, weights[op]); err != nil {
			return nil, err
		}
		must(Encrypt(t.as(op), r, in))
		os.Remove(in)
	}
	_, err = Aggregate(aggC, r)
	t.check(fmt.Sprintf("round %d: %s skipped → aggregation refused (%d contributors)", r, strings.Join(skipped, ", "), len(submitters)), err != nil, "%v", err)
	// A deviating aggregator sums the genuine submissions itself and asks anyway. The
	// recompute would match, so the operators' contributor rule must refuse.
	_, pkHash, err := aggC.readPK()
	if err != nil {
		return nil, err
	}
	var few [][]*rlwe.Ciphertext
	for _, op := range submitters {
		cts, _, err := aggC.verifySubmission(r, op, pkHash)
		if err != nil {
			return nil, err
		}
		few = append(few, cts)
	}
	sum2, err := he.Aggregate(aggC.Params, few)
	if err != nil {
		return nil, err
	}
	b2, err := he.MarshalList(he.MagicCiphertexts, sum2)
	if err != nil {
		return nil, err
	}
	active2 := append(slices.Clone(submitters[:thr-1]), skipped[0]) // 2-of-3: op_a, op_c
	req2 := protocol.DecryptRequest{RunID: aggC.RunID, Round: r, Active: active2,
		Contributors: submitters, AggSHA256: he.SHA256Hex(b2)}
	reqDir2 := filepath.Join(root, "bus", fmt.Sprintf("round_%d", r), "decrypt")
	_ = os.MkdirAll(reqDir2, 0o755)
	_ = protocol.WriteJSON(filepath.Join(reqDir2, "request.json"), req2, 0o644)
	allRefused := true
	var details []string
	for _, op := range active2 {
		_, e := DecShare(t.as(op), r)
		allRefused = allRefused && e != nil
		details = append(details, fmt.Sprintf("%s: %v", op, e))
	}
	name := "refuse: genuine 2-contributor sum (each would learn the other's update)"
	if k := len(submitters); k != 2 {
		name = fmt.Sprintf("refuse: genuine %d-contributor sum (%d colluders would learn an individual update)", k, thr-1)
	}
	t.check(name, allRefused, "%s", strings.Join(details, "; "))

	// The aggregator asks to decrypt op_a's individual submission as "the aggregate".
	// Every party is listed as a contributor, so only the recompute check can catch it.
	r++
	for _, op := range t.parts {
		in := filepath.Join(root, "parties", op, "work", "u.f64")
		if _, err := randomUpdate(in, n, weights[op]); err != nil {
			return nil, err
		}
		must(Encrypt(t.as(op), r, in))
		os.Remove(in)
	}
	ctA, _ := os.ReadFile(filepath.Join(root, "bus", fmt.Sprintf("round_%d", r), "submissions", "op_a", "ct.bin"))
	forged := protocol.DecryptRequest{RunID: aggC.RunID, Round: r, Active: t.parts[1 : 1+thr],
		Contributors: t.parts, AggSHA256: he.SHA256Hex(ctA)}
	reqDir := filepath.Join(root, "bus", fmt.Sprintf("round_%d", r), "decrypt")
	_ = os.MkdirAll(reqDir, 0o755)
	_ = protocol.WriteJSON(filepath.Join(reqDir, "request.json"), forged, 0o644)
	_, err = DecShare(t.as("op_b"), r)
	t.check("refuse: individual ciphertext labelled as aggregate", err != nil, "%v", err)
	forged.Contributors = []string{"op_a"}
	_ = protocol.WriteJSON(filepath.Join(reqDir, "request.json"), forged, 0o644)
	_, err = DecShare(t.as("op_b"), r)
	t.check("refuse: aggregate with a single contributor", err != nil, "%v", err)

	// Tampered ciphertext and replayed submission are rejected at aggregation.
	r++
	for _, op := range t.parts {
		in := filepath.Join(root, "parties", op, "work", "u.f64")
		if _, err := randomUpdate(in, n, weights[op]); err != nil {
			return nil, err
		}
		must(Encrypt(t.as(op), r, in))
		os.Remove(in)
	}
	cf := filepath.Join(root, "bus", fmt.Sprintf("round_%d", r), "submissions", "op_c", "ct.bin")
	b, _ := os.ReadFile(cf)
	b[len(b)/2] ^= 1
	_ = os.WriteFile(cf, b, 0o644)
	src := filepath.Join(root, "bus", "round_1", "submissions", "op_b")
	dst := filepath.Join(root, "bus", fmt.Sprintf("round_%d", r), "submissions", "op_b")
	for _, f := range []string{"ct.bin", "statement.json"} {
		bb, _ := os.ReadFile(filepath.Join(src, f))
		_ = os.WriteFile(filepath.Join(dst, f), bb, 0o644)
	}
	ar, err := Aggregate(aggC, r)
	rej := ""
	rejected := err != nil
	if err != nil {
		rej = err.Error()
	} else { // enough honest submissions remain: both bad ones must still be rejected
		m, _ := ar["rejected"].(map[string]string)
		_, b := m["op_b"]
		_, c := m["op_c"]
		rejected = b && c
		rej = fmt.Sprintf("aggregated %v, rejected %v", ar["contributors"], m)
	}
	t.check("reject: tampered ct (op_c) and replayed round-1 submission (op_b)", rejected, "%s", rej)

	allOK := true
	for _, ch := range t.checks {
		allOK = allOK && ch.OK
	}
	slots := he.Slots(aggC.Params)
	return Result{"ok_all": allOK, "n_values": n, "n_ciphertexts": (n + slots - 1) / slots,
		"checks": t.checks, "timings": timings, "ceremony": cer}, nil
}

// PrettyChecks renders checks for humans (stderr).
func PrettyChecks(res Result) string {
	cs, _ := res["checks"].([]Check)
	out := ""
	for _, c := range cs {
		mark := "PASS"
		if !c.OK {
			mark = "FAIL"
		}
		out += fmt.Sprintf("  [%s] %-62s %s\n", mark, c.Name, c.Detail)
	}
	return out
}
