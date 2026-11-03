package he

import (
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"math/bits"

	"github.com/tuneinsight/lattigo/v6/ring"
)

// addFlood adds exactly uniform integer noise from [−2^(w−1), 2^(w−1)), sampled from
// crypto/rand, to every coefficient of p (coefficient domain).
//
// Each sample is w raw random bits read as an unsigned integer and shifted down by
// 2^(w−1). There is no floating point and no rejection, so the distribution is exactly
// uniform over 2^w integers, as the smudging lemma assumes.
func addFlood(r *ring.Ring, w int, p ring.Poly) error {
	if w <= 64 || w > 127 {
		return fmt.Errorf("flood width %d not in (64, 127]", w)
	}
	n := r.N()
	buf := make([]byte, 16*n)
	if _, err := rand.Read(buf); err != nil {
		return fmt.Errorf("flood randomness: %w", err)
	}
	moduli := r.ModuliChain()[:r.Level()+1]
	hiMask := uint64(1)<<(w-64) - 1
	for i := 0; i < n; i++ {
		lo := binary.LittleEndian.Uint64(buf[16*i:])
		hi := binary.LittleEndian.Uint64(buf[16*i+8:]) & hiMask
		neg, mHi, mLo := centre(hi, lo, w)
		for j, q := range moduli {
			res := residue(neg, mHi, mLo, q)
			p.Coeffs[j][i] = ring.CRed(p.Coeffs[j][i]+res, q)
		}
	}
	return nil
}

// centre maps u = hi·2^64 + lo ∈ [0, 2^w) to v = u − 2^(w−1) ∈ [−2^(w−1), 2^(w−1)),
// returned as a sign plus a 128-bit magnitude (mHi, mLo).
func centre(hi, lo uint64, w int) (neg bool, mHi, mLo uint64) {
	half := uint64(1) << (w - 65) // 2^(w−1) expressed in units of 2^64
	if hi >= half {
		return false, hi - half, lo
	}
	// |v| = 2^(w−1) − u = (half − hi)·2^64 − lo
	if lo == 0 {
		return true, half - hi, 0
	}
	return true, half - hi - 1, -lo
}

// residue returns v mod q in [0, q) for v = ±(mHi·2^64 + mLo).
func residue(neg bool, mHi, mLo, q uint64) uint64 {
	r := bits.Rem64(mHi%q, mLo, q)
	if neg && r != 0 {
		return q - r
	}
	return r
}
