package he

import (
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Noise flooding parameters.
//
// An honest decryption share is c1·s'_i + f_i. Anyone who knows the other shares and the
// output can compute f_i − e_ct, and the ciphertext noise e_ct depends on the secret key
// (it contains e1·s). f_i is therefore uniform over an interval 2^(λ+1)·B_ct wide: by the
// smudging lemma, f_i and f_i − e_ct are within statistical distance 2^-(λ+1).
const (
	// StatSecurity is λ, the statistical security parameter of the flooding.
	StatSecurity = 64
	// NoiseFailLog2 is log2 of the per-coefficient probability that the ciphertext noise
	// exceeds the bound returned by CiphertextNoiseBound.
	NoiseFailLog2 = -80
)

// CiphertextNoiseBound returns B_ct such that every coefficient of the decryption noise
// of a sum of `summands` fresh public-key encryptions has |e| ≤ B_ct, except with
// probability ≤ 2·summands·2^NoiseFailLog2 per coefficient.
//
// Derivation (no special modulus P). With pk = (−a·s + e_pk, a) and
// ct = (u·pk0 + e0 + Δm, u·pk1 + e1), decryption gives Δm + u·e_pk + e0 + e1·s, where
//   - s = Σ sk_i: a sum of `parties` ternary keys, so |s_j| ≤ parties
//   - e_pk = Σ e_i: |e_pk_j| ≤ parties·β, β = the Gaussian sampler's hard bound
//   - u ternary, e0 and e1 bounded by β.
//
// In the conjugate-invariant ring Z[X+X^-1]/(X^2N+1), each product coefficient is
// Σ_j x_j·c_j, where c_j combines at most two coefficients of the other factor, so
// |c_j| ≤ 2·max. Conditioned on that factor the terms are independent and bounded, and
// Hoeffding gives Pr[|Σ| > t] ≤ 2·exp(−t² / (2·Σ c_j²)) with Σ c_j² ≤ N·(2·parties·β)².
// The same bound covers u·e_pk and e1·s (|u|, |s/parties| ≤ 1 against |e| ≤ β). It is
// deliberately loose; TestCiphertextNoiseUnderBound checks the measured noise against it.
func CiphertextNoiseBound(params ckks.Parameters, parties, summands int) (float64, error) {
	beta, err := gaussianHardBound(params)
	if err != nil {
		return 0, err
	}
	n := float64(params.N())
	lnTerm := math.Log(2) * float64(1-NoiseFailLog2) // ln(2/δ), δ = 2^NoiseFailLog2
	tail := 2 * float64(parties) * beta * math.Sqrt(2*n*lnTerm)
	perCt := 2*tail + beta // u·e_pk + e1·s + e0
	return float64(summands) * perCt, nil
}

// FloodLogWidth is the bit width w of the flooding interval [−2^(w−1), 2^(w−1)):
// w = λ + 1 + ⌈log2 B_ct⌉ for an aggregate of up to n submissions under a key shared by
// n parties. B_ct grows roughly as n² (n keys inside each of n summands), so w grows by
// about one bit from n = 3 to 4 and two bits by n = 7.
func FloodLogWidth(params ckks.Parameters, n int) (int, error) {
	if n < 1 {
		return 0, fmt.Errorf("flood width for %d parties", n)
	}
	bct, err := CiphertextNoiseBound(params, n, n)
	if err != nil {
		return 0, err
	}
	w := StatSecurity + 1 + int(math.Ceil(math.Log2(bct)))
	if w <= 64 || w > 127 {
		return 0, fmt.Errorf("flood width 2^%d outside the supported (2^64, 2^127] range", w)
	}
	// The sum of up to n floods (each < 2^(w−1)) must stay far below Q/2, or decryption
	// wraps. A decryption uses t ≤ n shares, so n is the worst case.
	if float64(w)+math.Ceil(math.Log2(float64(n))) >= params.LogQ()-2 {
		return 0, fmt.Errorf("flood width 2^%d too large for LogQ %.1f", w, params.LogQ())
	}
	return w, nil
}

// gaussianHardBound is the largest |e| Lattigo's error sampler can output: the
// sampler rejects |v| > Bound and then rounds v to the nearest integer.
func gaussianHardBound(params ckks.Parameters) (float64, error) {
	if _, ok := params.Xe().(ring.DiscreteGaussian); !ok {
		return 0, fmt.Errorf("error distribution %T is not a bounded discrete Gaussian", params.Xe())
	}
	if _, ok := params.Xs().(ring.Ternary); !ok {
		return 0, fmt.Errorf("secret distribution %T is not ternary", params.Xs())
	}
	return math.Floor(params.NoiseBound() + 0.5), nil
}
