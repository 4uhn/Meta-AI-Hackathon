package he

import (
	"fmt"
	"math"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/multiparty"
	"github.com/tuneinsight/lattigo/v6/ring"
)

// tofnConfigs are the (N, t) shapes tested end to end.
var tofnConfigs = [][2]int{{3, 2}, {4, 3}, {5, 3}, {5, 4}}

// subsets returns every k-subset of {0..n-1} in lexicographic order.
func subsets(n, k int) [][]int {
	var out [][]int
	var rec func(start int, cur []int)
	rec = func(start int, cur []int) {
		if len(cur) == k {
			out = append(out, append([]int(nil), cur...))
			return
		}
		for i := start; i < n; i++ {
			rec(i+1, append(cur, i))
		}
	}
	rec(0, nil)
	return out
}

// errTolerance scales the 2-of-3 tolerance (1e-6) with the standard deviation of the sum
// of t uniform floods of width 2^w, which is ∝ 2^w·√t.
func errTolerance(w, t int) float64 {
	return 1e-6 * math.Exp2(float64(w-85)) * math.Sqrt(float64(t)/2)
}

// sharesFrom computes decryption shares for the active set (indices into c.ops) and returns
// those of the parties in use only.
func sharesFrom(t *testing.T, c *ceremony, active, use []int, cts []*rlwe.Ciphertext) [][]multiparty.KeySwitchShare {
	t.Helper()
	var pts []Point
	for _, i := range active {
		pts = append(pts, c.ops[i].point)
	}
	var out [][]multiparty.KeySwitchShare
	for _, i := range use {
		s, err := DecryptionShare(c.params, c.ops[i].tsk, c.ops[i].point, pts, c.thr, c.n(), cts)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, s)
	}
	return out
}

// TestThresholdEverySubset: for every (N, t) we claim, every t-subset decrypts the
// aggregate of N submissions within tolerance, and every (t−1)-subset gets noise, both by
// withholding one share of a valid set and by interpolating over only t−1 points.
func TestThresholdEverySubset(t *testing.T) {
	for _, cfg := range tofnConfigs {
		n, thr := cfg[0], cfg[1]
		t.Run(fmt.Sprintf("%dof%d", thr, n), func(t *testing.T) {
			c := runCeremonyN(t, n, thr)
			w, err := FloodLogWidth(c.params, n)
			if err != nil {
				t.Fatal(err)
			}
			nv := Slots(c.params) + 5 // two ciphertexts per submission
			var subs [][]*rlwe.Ciphertext
			want := make([]float64, nv)
			for range c.ops {
				v := gaussianVec(nv, 0.05)
				for i := range v {
					want[i] += v[i]
				}
				ct, err := Encrypt(c.params, c.pk, n, v)
				if err != nil {
					t.Fatal(err)
				}
				subs = append(subs, ct)
			}
			agg, err := Aggregate(c.params, subs)
			if err != nil {
				t.Fatal(err)
			}

			tol := errTolerance(w, thr)
			worst := 0.0
			for _, set := range subsets(n, thr) {
				got, err := CombineDecryption(c.params, agg, sharesFrom(t, c, set, set, agg), nv)
				if err != nil {
					t.Fatal(err)
				}
				e := maxAbsErr(got, want)
				worst = math.Max(worst, e)
				if e > tol || e > 1e-5 {
					t.Errorf("set %v: max abs err %.3e > tol %.3e", set, e, tol)
				}
			}

			minRel := math.Inf(1)
			for _, part := range subsets(n, thr-1) {
				// (i) a valid active set with one member's share withheld.
				active := append([]int(nil), part...)
				for i := 0; len(active) < thr; i++ {
					if !containsInt(active, i) {
						active = append(active, i)
					}
				}
				got, err := CombineDecryption(c.params, agg, sharesFrom(t, c, active, part, agg), nv)
				if err != nil {
					t.Fatal(err)
				}
				minRel = math.Min(minRel, relErrVec(got, want))
				// (ii) the t−1 parties interpolate among themselves.
				got = interpolateOver(t, c, part, agg, nv)
				minRel = math.Min(minRel, relErrVec(got, want))
			}
			t.Logf("%d-of-%d: flood 2^%d; %d sets decrypt, worst err %.2e (tol %.2e); %d (t−1)-sets ×2 attacks, min rel err %.3g",
				thr, n, w, len(subsets(n, thr)), worst, tol, len(subsets(n, thr-1)), minRel)
			if minRel < 0.5 {
				t.Errorf("a (t−1)-subset recovered the aggregate (rel err %.3g)", minRel)
			}
		})
	}
}

// interpolateOver: the parties in part (fewer than t) build additive shares by Lagrange
// interpolation over themselves only and key-switch to the zero key without flooding.
// Their Shamir points lie on a degree-(t−1) polynomial, so this does not reach s.
func interpolateOver(t *testing.T, c *ceremony, part []int, cts []*rlwe.Ciphertext, nv int) []float64 {
	t.Helper()
	var pts []Point
	for _, i := range part {
		pts = append(pts, c.ops[i].point)
	}
	var shares [][]multiparty.KeySwitchShare
	for _, i := range part {
		op := c.ops[i]
		sk := rlwe.NewSecretKey(c.params)
		if err := multiparty.NewCombiner(c.params, op.point, pts, len(pts)).GenAdditiveShare(pts, op.point, op.tsk, sk); err != nil {
			t.Fatal(err)
		}
		cks, err := multiparty.NewKeySwitchProtocol(c.params, ring.DiscreteGaussian{Sigma: 0, Bound: 0})
		if err != nil {
			t.Fatal(err)
		}
		var s []multiparty.KeySwitchShare
		for _, ct := range cts {
			sh := cks.AllocateShare(ct.Level())
			cks.GenShare(sk, rlwe.NewSecretKey(c.params), ct, &sh)
			s = append(s, sh)
		}
		shares = append(shares, s)
	}
	got, err := CombineDecryption(c.params, cts, shares, nv)
	if err != nil {
		t.Fatal(err)
	}
	return got
}

func containsInt(xs []int, x int) bool {
	for _, y := range xs {
		if y == x {
			return true
		}
	}
	return false
}

func relErrVec(got, want []float64) float64 {
	var num, den float64
	for i := range want {
		num += (got[i] - want[i]) * (got[i] - want[i])
		den += want[i] * want[i]
	}
	return math.Sqrt(num / den)
}

// TestMaxAbsByN pins the encrypt guard per consortium size and the headroom check.
func TestMaxAbsByN(t *testing.T) {
	want := map[int]float64{3: 4, 4: 2, 5: 2, 6: 2, 7: 2}
	for n, v := range want {
		if got := MaxAbs(n); got != v {
			t.Errorf("MaxAbs(%d) = %v, want %v", n, got, v)
		}
		if 2*float64(n)*MaxAbs(n) >= 32 {
			t.Errorf("n = %d: 2·n·MaxAbs = %v does not fit under headroom 32", n, 2*float64(n)*MaxAbs(n))
		}
	}
	for _, ls := range []int{115, 116} {
		if _, err := NewParameters(Config{LogN: 14, LogQ: []int{60, 60}, LogScale: ls}); err == nil {
			t.Errorf("LogScale %d accepted: headroom too small for %d parties", ls, MaxParties)
		}
	}
}

// TestFloodWidthByN: w(3) is the pinned 85; w never decreases with n; the budget holds
// up to MaxParties.
func TestFloodWidthByN(t *testing.T) {
	params, err := NewParameters(DefaultConfig)
	if err != nil {
		t.Fatal(err)
	}
	prev := 0
	for n := MinParties; n <= MaxParties; n++ {
		w, err := FloodLogWidth(params, n)
		if err != nil {
			t.Fatalf("n = %d: %v", n, err)
		}
		bct, _ := CiphertextNoiseBound(params, n, n)
		t.Logf("n = %d: B_ct = 2^%.2f, flood width 2^%d, MaxAbs %v", n, math.Log2(bct), w, MaxAbs(n))
		if n == 3 && w != 85 {
			t.Fatalf("w(3) = %d, the 2-of-3 runs pin 85", w)
		}
		if w < prev {
			t.Fatalf("w(%d) = %d < w(%d) = %d", n, w, n-1, prev)
		}
		prev = w
	}
}

// TestCiphertextNoiseUnderBoundN measures the real noise at larger consortia, where the
// bound's n-dependence matters.
func TestCiphertextNoiseUnderBoundN(t *testing.T) {
	for _, n := range []int{5, 7} {
		t.Run(fmt.Sprint(n), func(t *testing.T) { checkNoiseUnderBound(t, runCeremonyN(t, n, n/2+1)) })
	}
}

// TestMaxAbsSevenParties: seven submissions at ±MaxAbs(7) (the tightest headroom) decrypt.
func TestMaxAbsSevenParties(t *testing.T) {
	c := runCeremonyN(t, 7, 4)
	w, _ := FloodLogWidth(c.params, 7)
	n := Slots(c.params)
	m := MaxAbs(7)
	for name, sign := range map[string]func(int) float64{
		"constant": func(int) float64 { return 1 },
		"signs":    func(i int) float64 { return math.Copysign(1, math.Sin(float64(i)*1.7)) },
	} {
		var subs [][]*rlwe.Ciphertext
		want := make([]float64, n)
		for range c.ops {
			v := make([]float64, n)
			for i := range v {
				v[i] = m * sign(i)
				want[i] += v[i]
			}
			ct, err := Encrypt(c.params, c.pk, 7, v)
			if err != nil {
				t.Fatal(err)
			}
			subs = append(subs, ct)
		}
		agg, err := Aggregate(c.params, subs)
		if err != nil {
			t.Fatal(err)
		}
		set := []int{0, 2, 4, 6}
		got, err := CombineDecryption(c.params, agg, sharesFrom(t, c, set, set, agg), n)
		if err != nil {
			t.Fatal(err)
		}
		if e := maxAbsErr(got, want); e > errTolerance(w, 4) {
			t.Fatalf("%s at ±%v ×7: err %.3e", name, m, e)
		}
	}
	if _, err := Encrypt(c.params, c.pk, 7, []float64{MaxAbs(7) * 1.01}); err == nil {
		t.Fatal("value over MaxAbs(7) accepted")
	}
}

// TestDecryptionShareRejectsBadSets: the active set must have exactly t distinct members
// including the caller.
func TestDecryptionShareRejectsBadSets(t *testing.T) {
	c := runCeremonyN(t, 4, 3)
	ct, err := Encrypt(c.params, c.pk, 4, []float64{1})
	if err != nil {
		t.Fatal(err)
	}
	a := c.ops[0]
	for name, pts := range map[string][]Point{
		"too few":    {1, 2},
		"too many":   {1, 2, 3, 4},
		"repeat":     {1, 1, 2},
		"not in set": {2, 3, 4},
	} {
		if _, err := DecryptionShare(c.params, a.tsk, a.point, pts, 3, 4, ct); err == nil {
			t.Errorf("%s: %v accepted", name, pts)
		}
	}
}
