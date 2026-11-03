package he

import (
	"bytes"
	"encoding"
	"math"
	"math/big"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/multiparty"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

var pairs = [][2]int{{0, 1}, {0, 2}, {1, 2}}

// decryptWith runs threshold decryption of cts with the operators at the given indices.
func decryptWith(t *testing.T, c *ceremony, idx []int, cts []*rlwe.Ciphertext, n int) []float64 {
	t.Helper()
	active := make([]Point, 0, len(idx))
	for _, i := range idx {
		active = append(active, c.ops[i].point)
	}
	if len(active) == 1 { // solo attempt: pair the operator with any other so the share is well-formed
		active = append(active, c.ops[(idx[0]+1)%3].point)
	}
	var shares [][]multiparty.KeySwitchShare
	for _, i := range idx {
		s, err := DecryptionShare(c.params, c.ops[i].tsk, c.ops[i].point, active, c.thr, c.n(), cts)
		if err != nil {
			t.Fatal(err)
		}
		shares = append(shares, s)
	}
	got, err := CombineDecryption(c.params, cts, shares, n)
	if err != nil {
		t.Fatal(err)
	}
	return got
}

// encryptThree encrypts three independent random vectors and returns them with their sum.
func encryptThree(t *testing.T, c *ceremony, n int, gen func(int) []float64) ([][]*rlwe.Ciphertext, []float64) {
	t.Helper()
	sum := make([]float64, n)
	var subs [][]*rlwe.Ciphertext
	for range c.ops {
		v := gen(n)
		for i := range v {
			sum[i] += v[i]
		}
		ct, err := Encrypt(c.params, c.pk, c.n(), v)
		if err != nil {
			t.Fatal(err)
		}
		subs = append(subs, ct)
	}
	return subs, sum
}

// TestT1SinglePartyRoundtrip checks encoding with a single-party key: no threshold, no flood.
func TestT1SinglePartyRoundtrip(t *testing.T) {
	params, err := NewParameters(DefaultConfig)
	if err != nil {
		t.Fatal(err)
	}
	sk, pk := rlwe.NewKeyGenerator(params).GenKeyPairNew()
	x := gaussianVec(3*Slots(params)+17, 0.05)
	cts, err := Encrypt(params, pk, 3, x)
	if err != nil {
		t.Fatal(err)
	}
	ecd, dec := newEncoder(params), rlwe.NewDecryptor(params, sk)
	got := make([]float64, 0, len(x))
	buf := make([]float64, Slots(params))
	for _, ct := range cts {
		if err := ecd.Decode(dec.DecryptNew(ct), buf); err != nil {
			t.Fatal(err)
		}
		got = append(got, buf...)
	}
	maxX := 0.0
	for _, v := range x {
		maxX = math.Max(maxX, math.Abs(v))
	}
	e := maxAbsErr(got[:len(x)], x)
	t.Logf("T1: %d values, max abs err %.2e (bound %.2e)", len(x), e, 1e-6*maxX)
	if e > 1e-6*maxX {
		t.Fatalf("T1 failed: %.3e", e)
	}
}

// TestT2ThreeParty100k: every 2-subset decrypts the sum of three 100k-value updates
// ~ N(0, 0.05²) within 1e-6.
func TestT2ThreeParty100k(t *testing.T) {
	c := runCeremony(t)
	const n = 100_000
	subs, want := encryptThree(t, c, n, func(n int) []float64 { return gaussianVec(n, 0.05) })
	agg, err := Aggregate(c.params, subs)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range pairs {
		got := decryptWith(t, c, p[:], agg, n)
		e := maxAbsErr(got, want)
		t.Logf("T2: %s+%s over %d values (%d ciphertexts): max abs err %.2e", c.ops[p[0]].name, c.ops[p[1]].name, n, len(agg), e)
		if e > 1e-6 {
			t.Errorf("T2 %v: max abs err %.3e > 1e-6", p, e)
		}
	}
}

// TestT3SoloShareGarbage: any single party's share yields garbage (relative error ≥ 0.5).
func TestT3SoloShareGarbage(t *testing.T) {
	c := runCeremony(t)
	const n = 20_000
	subs, want := encryptThree(t, c, n, func(n int) []float64 { return gaussianVec(n, 0.05) })
	agg, err := Aggregate(c.params, subs)
	if err != nil {
		t.Fatal(err)
	}
	for i := range c.ops {
		got := decryptWith(t, c, []int{i}, agg, n)
		rel := l2(sub(got, want)) / l2(want)
		t.Logf("T3: %s alone: relative error %.3g", c.ops[i].name, rel)
		if rel < 0.5 {
			t.Errorf("T3: %s alone has relative error %.3g < 0.5", c.ops[i].name, rel)
		}
	}
}

// TestT4TenMillion decrypts weighted updates w·B (w ≤ 1) of 10M values. Skipped with -short.
func TestT4TenMillion(t *testing.T) {
	if testing.Short() {
		t.Skip("T4 is slow; run without -short")
	}
	c := runCeremony(t)
	const n = 10_000_000
	ws := []float64{0.18, 0.31, 1.0}
	k := 0
	subs, want := encryptThree(t, c, n, func(n int) []float64 {
		v := gaussianVec(n, 0.05)
		for i := range v {
			v[i] *= ws[k]
		}
		k++
		return v
	})
	agg, err := Aggregate(c.params, subs)
	if err != nil {
		t.Fatal(err)
	}
	got := decryptWith(t, c, []int{0, 2}, agg, n)
	e := maxAbsErr(got, want)
	t.Logf("T4: %d values (%d ciphertexts), max abs err %.2e = 2^%.1f (%.1f bits after the binary point)", n, len(agg), e, math.Log2(e), -math.Log2(e))
	if e > 1e-6 {
		t.Fatalf("T4: max abs err %.3e > 1e-6", e)
	}
}

// roundtrip checks marshal → unmarshal → marshal gives identical bytes.
func roundtrip(t *testing.T, name string, obj encoding.BinaryMarshaler, fresh interface {
	encoding.BinaryMarshaler
	encoding.BinaryUnmarshaler
}) {
	t.Helper()
	b1, err := obj.MarshalBinary()
	if err != nil {
		t.Fatalf("%s marshal: %v", name, err)
	}
	if err := fresh.UnmarshalBinary(b1); err != nil {
		t.Fatalf("%s unmarshal: %v", name, err)
	}
	b2, err := fresh.MarshalBinary()
	if err != nil {
		t.Fatalf("%s re-marshal: %v", name, err)
	}
	if !bytes.Equal(b1, b2) {
		t.Fatalf("%s: re-marshalled bytes differ", name)
	}
	t.Logf("T5: %-10s %9d bytes, byte-identical", name, len(b1))
}

// TestT5Serialisation: every artefact survives serialisation byte-identically, and
// decryption from deserialised artefacts alone is still correct.
func TestT5Serialisation(t *testing.T) {
	c := runCeremony(t)
	p := c.params
	a, b := c.ops[0], c.ops[2]

	roundtrip(t, "sk", a.sk, new(rlwe.SecretKey))
	tsk := new(multiparty.ShamirSecretShare)
	roundtrip(t, "tsk", a.tsk, tsk)
	seed, _ := NewCRSSeed()
	pks, err := PublicKeyShare(p, a.sk, seed)
	if err != nil {
		t.Fatal(err)
	}
	roundtrip(t, "pk-share", pks, new(multiparty.PublicKeyGenShare))
	pk := new(rlwe.PublicKey)
	roundtrip(t, "pk", c.pk, pk)

	x := gaussianVec(2*Slots(p)+5, 0.05)
	cts, err := Encrypt(p, pk, 3, x) // encrypt under the deserialised pk
	if err != nil {
		t.Fatal(err)
	}
	ctBytes, err := MarshalList(MagicCiphertexts, cts)
	if err != nil {
		t.Fatal(err)
	}
	cts2, err := ReadCiphertexts(bytes.NewReader(ctBytes), p)
	if err != nil {
		t.Fatal(err)
	}
	if again, _ := MarshalList(MagicCiphertexts, cts2); !bytes.Equal(again, ctBytes) {
		t.Fatal("ciphertext list re-marshal differs")
	}
	t.Logf("T5: %-10s %9d bytes, byte-identical (%d ciphertexts)", "ct list", len(ctBytes), len(cts))

	tskB := new(multiparty.ShamirSecretShare)
	roundtrip(t, "tsk(b)", b.tsk, tskB)
	active := []Point{a.point, b.point}
	var shares [][]multiparty.KeySwitchShare
	for _, op := range []struct {
		pt  Point
		tsk multiparty.ShamirSecretShare
	}{{a.point, *tsk}, {b.point, *tskB}} {
		s, err := DecryptionShare(p, op.tsk, op.pt, active, 2, 3, cts2)
		if err != nil {
			t.Fatal(err)
		}
		sb, err := MarshalList(MagicShares, s)
		if err != nil {
			t.Fatal(err)
		}
		s2, err := ReadShares(bytes.NewReader(sb), p)
		if err != nil {
			t.Fatal(err)
		}
		if again, _ := MarshalList(MagicShares, s2); !bytes.Equal(again, sb) {
			t.Fatal("share list re-marshal differs")
		}
		shares = append(shares, s2)
	}
	got, err := CombineDecryption(p, cts2, shares, len(x))
	if err != nil {
		t.Fatal(err)
	}
	if e := maxAbsErr(got, x); e > 1e-6 {
		t.Fatalf("T5: decryption from deserialised artefacts err %.3e", e)
	} else {
		t.Logf("T5: decrypt from deserialised artefacts OK (max abs err %.2e)", e)
	}

	// Malformed inputs are rejected, not misparsed.
	if _, err := ReadShares(bytes.NewReader(ctBytes), p); err == nil {
		t.Fatal("ciphertext file accepted as a share file")
	}
	if _, err := ReadCiphertexts(bytes.NewReader(ctBytes[:len(ctBytes)-1]), p); err == nil {
		t.Fatal("truncated ciphertext file accepted")
	}
	if _, err := ReadCiphertexts(bytes.NewReader(append(append([]byte{}, ctBytes...), 0)), p); err == nil {
		t.Fatal("ciphertext file with trailing bytes accepted")
	}
}

// TestAggregateCanonical: the aggregate's bytes do not depend on submission order, so
// an operator can recompute it and compare a sha256.
func TestAggregateCanonical(t *testing.T) {
	c := runCeremony(t)
	subs, _ := encryptThree(t, c, 3*Slots(c.params), func(n int) []float64 { return gaussianVec(n, 0.05) })
	h := func(order []int) string {
		var s [][]*rlwe.Ciphertext
		for _, i := range order {
			s = append(s, subs[i])
		}
		agg, err := Aggregate(c.params, s)
		if err != nil {
			t.Fatal(err)
		}
		b, err := MarshalList(MagicCiphertexts, agg)
		if err != nil {
			t.Fatal(err)
		}
		return SHA256Hex(b)
	}
	ref := h([]int{0, 1, 2})
	for _, o := range [][]int{{2, 1, 0}, {1, 0, 2}, {2, 0, 1}} {
		if got := h(o); got != ref {
			t.Fatalf("aggregate hash depends on order %v", o)
		}
	}
	t.Logf("aggregate sha256 identical for every order: %s…", ref[:16])
}

// TestMaxAbsValuesDecrypt: three submissions at the encrypt guard ±MaxAbs(3) still
// decrypt, both for a constant vector and for random signs. Values over the guard are
// rejected.
func TestMaxAbsValuesDecrypt(t *testing.T) {
	c := runCeremony(t)
	MaxAbsValue := MaxAbs(3)
	n := Slots(c.params)
	gens := map[string]func(int) []float64{
		"constant": func(n int) []float64 {
			v := make([]float64, n)
			for i := range v {
				v[i] = MaxAbsValue
			}
			return v
		},
		"signs": func(n int) []float64 {
			v := gaussianVec(n, 1)
			for i := range v {
				v[i] = math.Copysign(MaxAbsValue, v[i])
			}
			return v
		},
	}
	for name, g := range gens {
		subs, want := encryptThree(t, c, n, g)
		agg, err := Aggregate(c.params, subs)
		if err != nil {
			t.Fatal(err)
		}
		got := decryptWith(t, c, []int{0, 1}, agg, n)
		if e := maxAbsErr(got, want); e > 1e-6 {
			t.Fatalf("%s at ±%v: err %.3e", name, MaxAbsValue, e)
		}
	}
	if _, err := Encrypt(c.params, c.pk, c.n(), []float64{0, MaxAbsValue * 1.01}); err == nil {
		t.Fatal("value over MaxAbsValue accepted")
	}
	if _, err := Encrypt(c.params, c.pk, c.n(), []float64{math.NaN()}); err == nil {
		t.Fatal("NaN accepted")
	}
}

// TestCIEmbeddingRowNorm backs the headroom argument in params.go: every polynomial
// coefficient of an encoded vector is at most 2·max|slot|, i.e. every row of the inverse
// embedding M has L1 norm ≤ 2. Unit slot vectors probe the entries of M.
func TestCIEmbeddingRowNorm(t *testing.T) {
	params, err := NewParameters(DefaultConfig)
	if err != nil {
		t.Fatal(err)
	}
	ecd := newEncoder(params)
	ringQ := params.RingQ().AtLevel(params.MaxLevel())
	n := Slots(params)
	delta := params.DefaultScale().Float64()
	// Scanning all N unit vectors is too slow, so sample 128 and bound each row's L1 norm
	// by max |M[j][k]|·N.
	maxEntry := 0.0
	unit := make([]float64, n)
	coeffs := make([]*big.Int, n)
	for i := range coeffs {
		coeffs[i] = new(big.Int)
	}
	deltaBig := new(big.Float).SetFloat64(delta)
	for k := 0; k < n; k += n / 128 {
		unit[k] = 1
		pt := ckks.NewPlaintext(params, params.MaxLevel())
		if err := ecd.Encode(unit, pt); err != nil {
			t.Fatal(err)
		}
		unit[k] = 0
		if pt.IsNTT {
			ringQ.INTT(pt.Value, pt.Value)
		}
		ringQ.PolyToBigintCentered(pt.Value, 1, coeffs) // Δ·M[j][k] exceeds one 60-bit limb
		for _, x := range coeffs {
			v, _ := new(big.Float).Quo(new(big.Float).SetInt(x), deltaBig).Float64()
			maxEntry = math.Max(maxEntry, math.Abs(v))
		}
	}
	rowL1 := maxEntry * float64(n) // ≥ any row's L1 norm
	t.Logf("max |M[j][k]| = %.4g/N → row L1 ≤ %.3f", maxEntry*float64(n), rowL1)
	if rowL1 > 2.0001 {
		t.Fatalf("CI row L1 bound %.4f exceeds 2: headroom argument in params.go is wrong", rowL1)
	}
}

func l2(v []float64) float64 {
	s := 0.0
	for _, x := range v {
		s += x * x
	}
	return math.Sqrt(s)
}

func sub(a, b []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] - b[i]
	}
	return out
}
