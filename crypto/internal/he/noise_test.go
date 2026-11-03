package he

import (
	"crypto/rand"
	"encoding/binary"
	"math"
	"math/big"
	mrand "math/rand/v2"
	"os"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// idealKey is a TEST-ONLY oracle: s = Σ sk_i, which no real party ever computes. It lets
// us look inside a ciphertext and measure its noise directly.
func idealKey(c *ceremony) *rlwe.SecretKey {
	s := rlwe.NewSecretKey(c.params)
	for _, op := range c.ops {
		c.params.RingQ().Add(s.Value.Q, op.sk.Value.Q, s.Value.Q)
	}
	return s
}

// gaussianVec returns n values ~ N(0, sd²) (test data; math/rand is fine here).
func gaussianVec(n int, sd float64) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = sd * mrand.NormFloat64()
	}
	return v
}

// TestCiphertextNoiseUnderBound measures the real decryption noise of a 3-submission
// aggregate (using the test-only ideal key) and checks it against the proved bound
// B_ct that sizes the flooding.
func TestCiphertextNoiseUnderBound(t *testing.T) { checkNoiseUnderBound(t, runCeremony(t)) }

// checkNoiseUnderBound measures an aggregate of one submission per party of c against
// B_ct(n, n), the bound that sizes the flooding for that consortium.
func checkNoiseUnderBound(t *testing.T, c *ceremony) {
	t.Helper()
	params := c.params
	bct, err := CiphertextNoiseBound(params, c.n(), c.n())
	if err != nil {
		t.Fatal(err)
	}
	w, err := FloodLogWidth(params, c.n())
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("B_ct bound (%d submissions, fail ≤ 2^%d/coeff) = %.0f = 2^%.2f; flood width = 2^%d",
		c.n(), NoiseFailLog2, bct, math.Log2(bct), w)

	const chunks = 8
	slots := Slots(params)
	ecd := newEncoder(params)
	ringQ := params.RingQ().AtLevel(params.MaxLevel())
	dec := rlwe.NewDecryptor(params, idealKey(c))

	var subs [][]*rlwe.Ciphertext
	expected := make([]ring.Poly, chunks) // Σ of the exact encoded plaintexts
	for k := range expected {
		expected[k] = ringQ.NewPoly()
	}
	for range c.ops {
		vals := gaussianVec(chunks*slots, 0.05)
		ct, err := Encrypt(params, c.pk, c.n(), vals)
		if err != nil {
			t.Fatal(err)
		}
		subs = append(subs, ct)
		for k := 0; k < chunks; k++ {
			pt := ckks.NewPlaintext(params, params.MaxLevel())
			if err := ecd.Encode(vals[k*slots:(k+1)*slots], pt); err != nil {
				t.Fatal(err)
			}
			ringQ.Add(expected[k], pt.Value, expected[k])
		}
	}
	agg, err := Aggregate(params, subs)
	if err != nil {
		t.Fatal(err)
	}

	q0 := ringQ.ModuliChain()[0]
	var maxAbs, sumSq float64
	count := 0
	for k, ct := range agg {
		pt := dec.DecryptNew(ct)
		diff := ringQ.NewPoly()
		ringQ.Sub(pt.Value, expected[k], diff)
		ringQ.INTT(diff, diff)
		for _, x := range diff.Coeffs[0] {
			v := float64(x)
			if x > q0/2 {
				v = -float64(q0 - x)
			}
			maxAbs = math.Max(maxAbs, math.Abs(v))
			sumSq += v * v
			count++
		}
	}
	std := math.Sqrt(sumSq / float64(count))
	t.Logf("measured noise over %d coefficients: std = %.1f (2^%.2f), max |e| = %.0f (2^%.2f); bound/max = 2^%.1f",
		count, std, math.Log2(std), maxAbs, math.Log2(maxAbs), math.Log2(bct/maxAbs))
	if maxAbs > bct {
		t.Fatalf("measured noise %.0f exceeds bound %.0f: flooding is undersized", maxAbs, bct)
	}
}

// TestFloodCentreResidue checks the 128-bit sign/magnitude and RNS arithmetic in flood.go
// against math/big for random and edge-case inputs.
func TestFloodCentreResidue(t *testing.T) {
	const w = 85
	q := uint64(0xfffffffffffc001) // any 60-bit odd modulus works for this check
	qBig := new(big.Int).SetUint64(q)
	half := new(big.Int).Lsh(big.NewInt(1), w-1)
	hiMask := uint64(1)<<(w-64) - 1

	cases := [][2]uint64{{0, 0}, {0, 1}, {hiMask, ^uint64(0)}, {1 << (w - 65), 0}, {1<<(w-65) - 1, ^uint64(0)}}
	buf := make([]byte, 16)
	for i := 0; i < 20000; i++ {
		rand.Read(buf)
		cases = append(cases, [2]uint64{binary.LittleEndian.Uint64(buf[8:]) & hiMask, binary.LittleEndian.Uint64(buf)})
	}
	for _, cse := range cases {
		hi, lo := cse[0], cse[1]
		u := new(big.Int).Lsh(new(big.Int).SetUint64(hi), 64)
		u.Add(u, new(big.Int).SetUint64(lo))
		want := new(big.Int).Sub(u, half) // v = u − 2^(w−1)

		neg, mHi, mLo := centre(hi, lo, w)
		got := new(big.Int).Lsh(new(big.Int).SetUint64(mHi), 64)
		got.Add(got, new(big.Int).SetUint64(mLo))
		if neg {
			got.Neg(got)
		}
		if got.Cmp(want) != 0 {
			t.Fatalf("centre(%x,%x) = %v, want %v", hi, lo, got, want)
		}
		wantRes := new(big.Int).Mod(want, qBig).Uint64()
		if r := residue(neg, mHi, mLo, q); r != wantRes {
			t.Fatalf("residue(%v) = %d, want %d", want, r, wantRes)
		}
	}
}

// TestFloodIsUniform samples a flood polynomial through the real ring, reconstructs each
// coefficient with CRT, and checks the range and the bucket uniformity of the top bits.
func TestFloodIsUniform(t *testing.T) {
	params, err := NewParameters(DefaultConfig)
	if err != nil {
		t.Fatal(err)
	}
	w, err := FloodLogWidth(params, 3)
	if err != nil {
		t.Fatal(err)
	}
	ringQ := params.RingQ().AtLevel(params.MaxLevel())
	const polys = 8
	const buckets = 64
	counts := make([]int, buckets)
	lim := new(big.Int).Lsh(big.NewInt(1), uint(w-1))
	total := 0
	sum := new(big.Int)
	for p := 0; p < polys; p++ {
		f := ringQ.NewPoly()
		if err := addFlood(ringQ, w, f); err != nil {
			t.Fatal(err)
		}
		vals := make([]*big.Int, ringQ.N())
		for i := range vals {
			vals[i] = new(big.Int)
		}
		ringQ.PolyToBigintCentered(f, 1, vals)
		for _, v := range vals {
			if v.Cmp(lim) >= 0 || v.Cmp(new(big.Int).Neg(lim)) < 0 {
				t.Fatalf("flood value %v outside [−2^%d, 2^%d)", v, w-1, w-1)
			}
			sum.Add(sum, v)
			b := new(big.Int).Add(v, lim) // [0, 2^w)
			b.Rsh(b, uint(w-6))           // top 6 bits → bucket
			counts[b.Int64()]++
			total++
		}
	}
	exp := float64(total) / buckets
	chi := 0.0
	for _, c := range counts {
		chi += (float64(c) - exp) * (float64(c) - exp) / exp
	}
	meanBits := new(big.Float).Quo(new(big.Float).SetInt(sum), big.NewFloat(float64(total)))
	mf, _ := meanBits.Float64()
	t.Logf("flood width 2^%d: %d samples, chi² (63 dof) = %.1f, mean/2^(w−1) = %.4f", w, total, chi, mf/math.Ldexp(1, w-1))
	// 63 dof: p(chi² > 130) < 1e-6.
	if chi > 130 {
		t.Fatalf("top-bit buckets not uniform: chi² = %.1f", chi)
	}
}

// TestScaleSweep reproduces the choice of Δ in DefaultConfig: max decryption error under
// real flooding vs headroom, for several scales. Opt-in (HORIZON_SWEEP=1; make crypto-report).
func TestScaleSweep(t *testing.T) {
	if os.Getenv("HORIZON_SWEEP") == "" {
		t.Skip("set HORIZON_SWEEP=1")
	}
	saved := DefaultConfig
	defer func() { DefaultConfig = saved }()
	for _, ls := range []int{108, 110, 112, 113, 114, 115, 116} {
		DefaultConfig = Config{LogN: saved.LogN, LogQ: saved.LogQ, LogScale: ls}
		if _, err := NewParameters(DefaultConfig); err != nil {
			t.Logf("sweep: log Δ = %d rejected: %v", ls, err)
			continue
		}
		c := runCeremony(t)
		n := 20 * Slots(c.params)
		subs, want := encryptThree(t, c, n, func(n int) []float64 { return gaussianVec(n, 0.05) })
		agg, err := Aggregate(c.params, subs)
		if err != nil {
			t.Fatal(err)
		}
		got := decryptWith(t, c, []int{0, 1}, agg, n)
		t.Logf("sweep: log Δ = %d, headroom Q/2Δ = %.0f, max abs err %.2e over %d values",
			ls, math.Exp2(c.params.LogQ()-1-float64(ls)), maxAbsErr(got, want), n)
	}
}
