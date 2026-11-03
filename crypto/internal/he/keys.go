package he

import (
	"crypto/rand"
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/multiparty"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Point is a party's public Shamir evaluation point (by default 1..N in party order).
// Points must be distinct and non-zero, since S(0) is the ideal secret key.
type Point = multiparty.ShamirPublicPoint

// GenSecretKey runs on one operator. The key is sampled from crypto/rand through
// Lattigo's sampling.NewPRNG (a direct rand.Read wrapper), so it is never reproducible.
func GenSecretKey(params ckks.Parameters) *rlwe.SecretKey {
	return rlwe.NewKeyGenerator(params).GenSecretKeyNew()
}

// NewCRSSeed draws the 32-byte common reference string seed for a run. This is the
// only deterministic randomness in the protocol, and it is public by design: the CRP it
// expands to is the public half `a` of the collective public key (-a·s+e, a).
func NewCRSSeed() ([]byte, error) {
	seed := make([]byte, 32)
	if _, err := rand.Read(seed); err != nil {
		return nil, fmt.Errorf("crs seed: %w", err)
	}
	return seed, nil
}

// DealShamirShares runs on one operator (the dealer of its own key). It hides sk_i as
// the constant term of a random degree-(t-1) polynomial S_i and evaluates S_i at every
// recipient's point, including its own. Each output share must travel only to its
// recipient over a private channel (simulated by per-recipient private directories).
func DealShamirShares(params ckks.Parameters, sk *rlwe.SecretKey, threshold int, recipients []Point) (map[Point]multiparty.ShamirSecretShare, error) {
	thr := multiparty.NewThresholdizer(params)
	poly, err := thr.GenShamirPolynomial(threshold, sk)
	if err != nil {
		return nil, fmt.Errorf("shamir polynomial: %w", err)
	}
	out := make(map[Point]multiparty.ShamirSecretShare, len(recipients))
	for _, p := range recipients {
		share := thr.AllocateThresholdSecretShare()
		thr.GenShamirSecretShare(p, poly, &share)
		out[p] = share
	}
	return out, nil
}

// AggregateShamirShares runs on one operator j. It sums the shares S_i(α_j) received
// from every dealer i (its own included) into its threshold secret share
// tsk_j = S(α_j), where S = Σ_i S_i and therefore S(0) = Σ_i sk_i.
func AggregateShamirShares(params ckks.Parameters, received []multiparty.ShamirSecretShare) (multiparty.ShamirSecretShare, error) {
	thr := multiparty.NewThresholdizer(params)
	acc := thr.AllocateThresholdSecretShare()
	for i, s := range received {
		if err := thr.AggregateShares(acc, s, &acc); err != nil {
			return acc, fmt.Errorf("aggregate share %d: %w", i, err)
		}
	}
	return acc, nil
}
