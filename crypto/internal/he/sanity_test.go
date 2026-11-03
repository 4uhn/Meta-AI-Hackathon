package he

import (
	"bytes"
	"math"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/multiparty"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// operator holds only what one organisation holds. Tests run all of them in one process,
// but nothing secret is read across operators: Shamir shares move via the mailbox.
type operator struct {
	name  string
	point Point
	sk    *rlwe.SecretKey              // original additive key (setup only)
	tsk   multiparty.ShamirSecretShare // threshold share, the long-lived secret
}

type ceremony struct {
	params ckks.Parameters
	ops    []*operator
	pk     *rlwe.PublicKey
	thr    int // t in t-of-N
}

// n is the number of parties in the ceremony.
func (c *ceremony) n() int { return len(c.ops) }

// runCeremony is the 2-of-3 setup.
func runCeremony(t *testing.T) *ceremony { return runCeremonyN(t, 3, 2) }

// runCeremonyN performs the dealer-free t-of-n setup: local keygen, Shamir re-sharing over
// private channels (a mailbox keyed by recipient), and collective public key generation.
// Operators are op_a, op_b, … at points 1..n.
func runCeremonyN(t *testing.T, n, thr int) *ceremony {
	t.Helper()
	params, err := NewParameters(DefaultConfig)
	if err != nil {
		t.Fatal(err)
	}
	var ops []*operator
	var points []Point
	for i := range n {
		ops = append(ops, &operator{name: "op_" + string(rune('a'+i)), point: Point(i + 1)})
		points = append(points, Point(i+1))
	}

	// 1. Each operator generates its own secret key locally.
	for _, op := range ops {
		op.sk = GenSecretKey(params)
	}

	// 2. Each operator deals Shamir shares of its key; mailbox[recipient] collects them.
	mailbox := map[Point][]multiparty.ShamirSecretShare{}
	for _, op := range ops {
		shares, err := DealShamirShares(params, op.sk, thr, points)
		if err != nil {
			t.Fatal(err)
		}
		for p, s := range shares {
			mailbox[p] = append(mailbox[p], s)
		}
	}
	// Each operator aggregates only its own mailbox into tsk.
	for _, op := range ops {
		if op.tsk, err = AggregateShamirShares(params, mailbox[op.point]); err != nil {
			t.Fatal(err)
		}
	}

	// 3. Collective public key from public shares; the aggregator combines them.
	seed, err := NewCRSSeed()
	if err != nil {
		t.Fatal(err)
	}
	var pkShares []multiparty.PublicKeyGenShare
	for _, op := range ops {
		s, err := PublicKeyShare(params, op.sk, seed)
		if err != nil {
			t.Fatal(err)
		}
		pkShares = append(pkShares, s)
	}
	pk, err := CombinePublicKey(params, seed, pkShares)
	if err != nil {
		t.Fatal(err)
	}
	return &ceremony{params: params, ops: ops, pk: pk, thr: thr}
}

func maxAbsErr(got, want []float64) float64 {
	m := 0.0
	for i := range want {
		m = math.Max(m, math.Abs(got[i]-want[i]))
	}
	return m
}

// TestSanityTwoOfThree is a smoke test: three operators encrypt small known vectors,
// the aggregator sums them blind, and every 2-of-3 subset decrypts the sum.
func TestSanityTwoOfThree(t *testing.T) {
	c := runCeremony(t)
	inputs := [][]float64{
		{0.01, 0.02, 0.03, 0.04}, // op_a
		{0.1, 0.2, 0.3, 0.4},     // op_b
		{1, 2, 3, 4},             // op_c
	}
	want := []float64{1.11, 2.22, 3.33, 4.44}

	var subs [][]*rlwe.Ciphertext
	for i, op := range c.ops {
		ct, err := Encrypt(c.params, c.pk, c.n(), inputs[i])
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%s encrypts %v -> %d ciphertext(s)", op.name, inputs[i], len(ct))
		subs = append(subs, ct)
	}
	agg, err := Aggregate(c.params, subs)
	if err != nil {
		t.Fatal(err)
	}

	for _, pair := range [][2]int{{0, 1}, {0, 2}, {1, 2}} {
		a, b := c.ops[pair[0]], c.ops[pair[1]]
		active := []Point{a.point, b.point}
		var shares [][]multiparty.KeySwitchShare
		for _, op := range []*operator{a, b} {
			s, err := DecryptionShare(c.params, op.tsk, op.point, active, c.thr, c.n(), agg)
			if err != nil {
				t.Fatal(err)
			}
			shares = append(shares, s)
		}
		got, err := CombineDecryption(c.params, agg, shares, len(want))
		if err != nil {
			t.Fatal(err)
		}
		e := maxAbsErr(got, want)
		t.Logf("%s + %s decrypt -> %.9f  (max abs err %.2e)", a.name, b.name, got, e)
		if e > 1e-6 {
			t.Errorf("%s+%s: max abs err %.3e > 1e-6", a.name, b.name, e)
		}
	}
}

// TestSoloShareIsGarbage: one operator's share alone leaves c1·s_other in the result.
func TestSoloShareIsGarbage(t *testing.T) {
	c := runCeremony(t)
	ct, err := Encrypt(c.params, c.pk, c.n(), []float64{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{1, 2, 3, 4}
	for _, op := range c.ops {
		other := c.ops[(int(op.point))%3] // any partner, to form a valid active set
		active := []Point{op.point, other.point}
		s, err := DecryptionShare(c.params, op.tsk, op.point, active, c.thr, c.n(), ct)
		if err != nil {
			t.Fatal(err)
		}
		got, err := CombineDecryption(c.params, ct, [][]multiparty.KeySwitchShare{s}, 4)
		if err != nil {
			t.Fatal(err)
		}
		e := maxAbsErr(got, want)
		t.Logf("%s alone -> %.3g ... (max abs err %.2e)", op.name, got[:2], e)
		if e < 1e3 {
			t.Errorf("%s alone recovered something close to the plaintext (err %.3e)", op.name, e)
		}
	}
}

// TestWrongActiveSetIsGarbage: additive shares are specific to the active set, so
// mixing a share computed for {A,B} with one computed for {A,C} must not decrypt.
func TestWrongActiveSetIsGarbage(t *testing.T) {
	c := runCeremony(t)
	ct, err := Encrypt(c.params, c.pk, c.n(), []float64{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	a, b, cc := c.ops[0], c.ops[1], c.ops[2]
	sa, err := DecryptionShare(c.params, a.tsk, a.point, []Point{a.point, b.point}, c.thr, c.n(), ct)
	if err != nil {
		t.Fatal(err)
	}
	sc, err := DecryptionShare(c.params, cc.tsk, cc.point, []Point{a.point, cc.point}, c.thr, c.n(), ct)
	if err != nil {
		t.Fatal(err)
	}
	got, err := CombineDecryption(c.params, ct, [][]multiparty.KeySwitchShare{sa, sc}, 4)
	if err != nil {
		t.Fatal(err)
	}
	if e := maxAbsErr(got, []float64{1, 2, 3, 4}); e < 1e3 {
		t.Errorf("mismatched active sets decrypted (err %.3e)", e)
	} else {
		t.Logf("A(for {A,B}) + C(for {A,C}) -> garbage (max abs err %.2e)", e)
	}
}

// TestSecretKeysAreFresh: identical calls never yield the same secret key.
func TestSecretKeysAreFresh(t *testing.T) {
	params, err := NewParameters(DefaultConfig)
	if err != nil {
		t.Fatal(err)
	}
	k1, err := GenSecretKey(params).MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	k2, err := GenSecretKey(params).MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(k1, k2) {
		t.Fatal("two secret keys are identical: randomness is not fresh")
	}
}
