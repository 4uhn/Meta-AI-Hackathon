// Package he implements Horizon's t-of-N threshold CKKS layer on top of Lattigo v6.
//
// Every exported function takes only the material a single party would hold (its own
// secret, public artefacts or ciphertexts), so party boundaries hold even when all
// parties run in one process.
package he

import (
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Config holds the tunable CKKS parameters. Everything else is fixed.
type Config struct {
	LogN     int   // ring degree N = 2^LogN
	LogQ     []int // bit sizes of the ciphertext modulus primes
	LogScale int   // CKKS scale Δ = 2^LogScale
}

// DefaultConfig uses the conjugate-invariant ring (N = 2^14 real slots per ciphertext)
// with Q ≈ 2^120: two 60-bit primes, Lattigo's per-prime maximum, well under the 438-bit
// limit for 128-bit security at N = 2^14. The circuit is additions only, so one level
// suffices and there is no P modulus and no evaluation key.
//
// Δ = 2^114 trades precision against headroom (TestScaleSweep). Under λ = 64 flooding
// (width 2^85) the max decryption error is ≈ 7e-7, halving per extra bit of Δ. Headroom
// is Q/2Δ = 32, so with |value| ≤ MaxAbs(n) per submission an aggregate of n never wraps
// (n = 3: ≤ 12 per slot, ≤ 24 per polynomial coefficient).
var DefaultConfig = Config{LogN: 14, LogQ: []int{60, 60}, LogScale: 114}

// Supported consortium sizes. The upper bound comes from the headroom:
// 2·7·MaxAbs(7) = 28 < Q/2Δ = 32 (see NewParameters for the factor 2).
const (
	MinParties = 3
	MaxParties = 7
)

// designHeadroom is the Q/2Δ that MaxAbs is sized for (DefaultConfig).
const designHeadroom = 32.0

// MaxAbs is the largest |value| Encrypt accepts in a consortium of n parties: the
// largest power of two p with 2·n·p < 32, so that a sum of n submissions never wraps.
// n = 3 → 4; n = 4..7 → 2. LoRA B entries and weights are ≪ 1.
func MaxAbs(n int) float64 {
	p := 1.0
	for 2*float64(n)*(2*p) < designHeadroom {
		p *= 2
	}
	return p
}

// NewParameters builds the CKKS parameters every party must share. Their hash is pinned
// in params.json and in every signed submission.
func NewParameters(cfg Config) (ckks.Parameters, error) {
	// Decoding needs |coefficient| < Q/2Δ. The CI inverse embedding has entries ≤ 1/N
	// (TestCIEmbeddingRowNorm), so row L1 norm ≤ 1; budgeting 2 for margin gives the
	// bound 2·n·MaxAbs(n), which must hold for every supported n.
	need := 0.0
	for n := MinParties; n <= MaxParties; n++ {
		need = math.Max(need, 2*float64(n)*MaxAbs(n))
	}
	if headroom := sumInts(cfg.LogQ) - 1 - cfg.LogScale; math.Exp2(float64(headroom)) <= need {
		return ckks.Parameters{}, fmt.Errorf("LogScale %d leaves headroom 2^%d under LogQ %v, need > %v",
			cfg.LogScale, headroom, cfg.LogQ, need)
	}
	return ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            cfg.LogN,
		LogQ:            cfg.LogQ,
		LogP:            nil,
		Xs:              rlwe.DefaultXs, // uniform ternary secret
		Xe:              rlwe.DefaultXe, // σ = 3.2 encryption error
		RingType:        ring.ConjugateInvariant,
		LogDefaultScale: cfg.LogScale,
	})
}

// Slots is the number of real values packed into one ciphertext.
func Slots(params ckks.Parameters) int { return params.MaxSlots() }

// newEncoder pins the encoder to float64 (53-bit) arithmetic. Otherwise Lattigo switches
// to big.Float FFTs once the scale exceeds 2^53, which is far slower and gains nothing:
// inputs are float64, and scale·value ≥ 2^64 is still converted exactly through big.Int
// (ckks.SingleFloat64ToFixedPointCRT).
func newEncoder(params ckks.Parameters) *ckks.Encoder { return ckks.NewEncoder(params, 53) }

func sumInts(xs []int) (s int) {
	for _, x := range xs {
		s += x
	}
	return
}
