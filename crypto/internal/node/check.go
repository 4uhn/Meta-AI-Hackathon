package node

import (
	"fmt"
	"math"
	"strings"

	"github.com/4uhn/Meta-AI-Hackathon/crypto/internal/he"
	"github.com/4uhn/Meta-AI-Hackathon/crypto/internal/protocol"
)

// VerifiedMaxParties is the largest consortium measured end to end at the Muse Glimmer
// payload (2-of-3, 3-of-4, 3-of-5, 4-of-5). Larger ones, up to he.MaxParties, need
// --unverified.
const VerifiedMaxParties = 5

// logQP128 is the largest log2(QP) with 128-bit security at ring degree 2^14 for a uniform
// ternary secret (Homomorphic Encryption Security Standard, Albrecht et al. 2018, Table 1).
const logQP128 = 438

// PartyNames returns op_a, op_b, … for a consortium of n.
func PartyNames(n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = fmt.Sprintf("op_%c", 'a'+i)
	}
	return out
}

type rule struct {
	OK   bool   `json:"ok"`
	Name string `json:"name"`
	Why  string `json:"why"`
}

// CheckShape validates a t-of-N consortium and reports the parameters it would run with.
// It returns the report, the human-readable text and whether the shape may run.
func CheckShape(n, t int, unverified bool) (Result, string, bool) {
	rules := []rule{
		{n >= he.MinParties && n <= he.MaxParties, fmt.Sprintf("%d ≤ N ≤ %d", he.MinParties, he.MaxParties),
			"this build's parameters are derived and tested for these sizes; N > 7 needs new ones (crypto/README.md)"},
		{t >= 2, "t ≥ 2", "with t = 1 any single party could decrypt every individual update"},
		{2*t > n, fmt.Sprintf("2t > N (%d > %d)", 2*t, n),
			"two decrypting groups always share a member, so one round cannot be decrypted twice and differenced"},
		{t <= n-1, fmt.Sprintf("t ≤ N − 1 (%d ≤ %d)", t, n-1),
			fmt.Sprintf("a sum is decrypted only with ≥ t + 1 = %d contributors (so t − 1 colluders never isolate one update); at t = N that is more than exist", t+1)},
	}
	measured := n <= VerifiedMaxParties
	why := "protocol, precision and cost measured at this size"
	if !measured && unverified {
		why = "not measured by the maintainers: --unverified, so the selftest below is the evidence"
	} else if !measured {
		why = "not measured: add --unverified to run the full selftest at this size yourself (crypto/README.md)"
	}
	rules = append(rules, rule{measured || unverified, fmt.Sprintf("N ≤ %d measured", VerifiedMaxParties), why})

	ok := true
	var b strings.Builder
	fmt.Fprintf(&b, "Consortium %d-of-%d: %s\n", t, n, strings.Join(PartyNames(min(max(n, 0), 26)), ", "))
	for _, r := range rules {
		ok = ok && r.OK
		fmt.Fprintf(&b, "  [%s] %-24s %s\n", map[bool]string{true: "ok", false: "NO"}[r.OK], r.Name, r.Why)
	}
	res := Result{"parties": n, "threshold": t, "rules": rules, "measured": measured, "allowed": ok}
	if !(n >= he.MinParties && n <= he.MaxParties && t >= 2) {
		return res, b.String(), false
	}

	params, err := he.NewParameters(he.DefaultConfig)
	if err != nil {
		return res, b.String() + err.Error() + "\n", false
	}
	w, err := he.FloodLogWidth(params, n)
	if err != nil {
		return res, b.String() + err.Error() + "\n", false
	}
	bct, _ := he.CiphertextNoiseBound(params, n, n)
	marginBits := float64(w-1) - math.Log2(bct)
	headroom := math.Exp2(float64(sumLogQ(he.DefaultConfig.LogQ) - 1 - he.DefaultConfig.LogScale))
	maxAbs := he.MaxAbs(n)
	need := 2 * float64(n) * maxAbs
	tol := ErrTolerance(w, t)
	minC := max(protocol.MinContributors, t+1)
	res["params"] = Result{
		"log_n": params.LogN(), "log_qp": params.LogQP(), "security_bits_min": 128,
		"flood_log_width": w, "noise_bound_log2": math.Log2(bct), "flood_margin_bits": marginBits,
		"stat_security_bits": he.StatSecurity, "headroom": headroom, "max_abs_value": maxAbs,
		"headroom_needed": need, "err_tolerance": tol, "min_contributors": minC,
		"offline_at_decryption": n - t, "may_skip_a_round": max(n-minC, 0),
	}
	fmt.Fprintf(&b, "Parameters\n")
	fmt.Fprintf(&b, "  security        ≥ 128-bit: ring 2^%d, log2 QP = %.0f ≤ %d\n", params.LogN(), params.LogQP(), logQP128)
	fmt.Fprintf(&b, "  noise flooding  2^%d wide, 2^%.1f × the proven noise bound 2^%.1f: statistical distance ≤ 2^-%d\n",
		w, marginBits, math.Log2(bct), he.StatSecurity)
	fmt.Fprintf(&b, "  headroom        each value in [-%g, %g]; a sum of %d needs %g < Q/2Δ = %g\n", maxAbs, maxAbs, n, need, headroom)
	fmt.Fprintf(&b, "  expected error  ≤ %.1e per value (the selftest's bound)\n", tol)
	fmt.Fprintf(&b, "  availability    every round needs %d submissions (%d may skip); %d may be offline at decryption\n",
		minC, max(n-minC, 0), n-t)
	return res, b.String(), ok
}

// CheckConsortium validates a t-of-N consortium and, if it may run, runs the full
// selftest on synthetic updates of n values in a fresh run directory under root.
func CheckConsortium(root string, n, parties, t int, unverified bool) (Result, string, error) {
	res, text, ok := CheckShape(parties, t, unverified)
	if !ok {
		return res, text, fmt.Errorf("%d-of-%d is not allowed: see the rules marked NO", t, parties)
	}
	ps, err := protocol.NewParties(PartyNames(parties), t)
	if err != nil {
		return res, text, err
	}
	st, err := Selftest(root, n, ps)
	if err != nil {
		return res, text, err
	}
	text += fmt.Sprintf("Selftest (%d values per update, every %d-subset decrypts)\n", n, t) + PrettyChecks(st)
	res["selftest"] = Result{"ok_all": st["ok_all"], "n_values": st["n_values"], "checks": st["checks"]}
	if st["ok_all"] != true {
		return res, text, fmt.Errorf("selftest: some checks failed")
	}
	text += fmt.Sprintf("Result: %d-of-%d is safe to run", t, parties)
	if parties > VerifiedMaxParties {
		text += " as far as this selftest shows (not measured by the maintainers; see crypto/README.md)"
	}
	return res, text + "\n", nil
}

func sumLogQ(xs []int) (s int) {
	for _, x := range xs {
		s += x
	}
	return
}
