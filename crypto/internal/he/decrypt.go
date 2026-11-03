package he

import (
	"fmt"
	"slices"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/multiparty"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// DecryptionShare runs on one operator in the active set of exactly t parties, in a
// consortium of n. Threshold decryption is a collective key switch from s to the
// all-zero key:
//
//  1. Combiner turns the Shamir share tsk = S(α_own) into an additive share for this
//     exact active set, sk' = S(α_own)·λ_own (λ = Lagrange coefficient), so the active
//     parties' sk' sum to s.
//  2. KeySwitchProtocol.GenShare outputs c1·sk' + e per ciphertext, with Lattigo's own
//     noise e at its minimum (σ = 3.2: Lattigo adds the fresh-key noise to the σ = 0
//     requested here). An exactly uniform flood f of width 2^FloodLogWidth(n) is then
//     added. Adding all t shares to c0 gives Δ·m + e_ct + Σ(e + f).
//
// Callers must verify the aggregate and ensure at most one share per round: this
// function keeps no record of what it has shared.
func DecryptionShare(params ckks.Parameters, tsk multiparty.ShamirSecretShare, own Point, active []Point, t, n int, cts []*rlwe.Ciphertext) ([]multiparty.KeySwitchShare, error) {
	// The combiner interpolates over the first t points only, so the set must be exact.
	if t < 2 || len(active) != t {
		return nil, fmt.Errorf("active set has %d parties, want exactly t = %d ≥ 2", len(active), t)
	}
	if !slices.Contains(active, own) {
		return nil, fmt.Errorf("party %d is not in the active set %v", own, active)
	}
	if len(slices.Compact(slices.Sorted(slices.Values(active)))) != len(active) {
		return nil, fmt.Errorf("active set %v repeats a party", active)
	}
	w, err := FloodLogWidth(params, n)
	if err != nil {
		return nil, err
	}

	skAdditive := rlwe.NewSecretKey(params)
	cmb := multiparty.NewCombiner(params, own, active, t)
	if err := cmb.GenAdditiveShare(active, own, tsk, skAdditive); err != nil {
		return nil, fmt.Errorf("additive share: %w", err)
	}

	zero := rlwe.NewSecretKey(params) // target key s' = 0 means "decrypt"
	out := make([]multiparty.KeySwitchShare, len(cts))
	err = parallelFor(len(cts), func() (multiparty.KeySwitchProtocol, error) {
		return multiparty.NewKeySwitchProtocol(params, ring.DiscreteGaussian{Sigma: 0, Bound: 0})
	}, func(cks multiparty.KeySwitchProtocol, k int) error {
		out[k] = cks.AllocateShare(cts[k].Level())
		cks.GenShare(skAdditive, zero, cts[k], &out[k])
		if err := floodShare(params, cts[k], w, &out[k]); err != nil {
			return fmt.Errorf("flood chunk %d: %w", k, err)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

// floodShare adds the flood to a share. The share is in the NTT domain whenever the
// ciphertext is (CKKS default), so the flood is sampled in coefficient form and
// transformed first.
func floodShare(params ckks.Parameters, ct *rlwe.Ciphertext, w int, share *multiparty.KeySwitchShare) error {
	ringQ := params.RingQ().AtLevel(share.Level())
	f := ringQ.NewPoly()
	if err := addFlood(ringQ, w, f); err != nil {
		return err
	}
	if ct.IsNTT {
		ringQ.NTT(f, f)
	}
	ringQ.Add(share.Value, f, share.Value)
	return nil
}

// CombineDecryption runs on the aggregator. It sums the parties' shares per ciphertext,
// applies them to c0, and decodes with the zero key. With fewer than t correct shares
// the c1·s term does not cancel and the output is uniformly random garbage.
func CombineDecryption(params ckks.Parameters, cts []*rlwe.Ciphertext, shares [][]multiparty.KeySwitchShare, nValues int) ([]float64, error) {
	cks, err := multiparty.NewKeySwitchProtocol(params, rlwe.DefaultXe)
	if err != nil {
		return nil, err
	}
	for p, s := range shares {
		if len(s) != len(cts) {
			return nil, fmt.Errorf("party %d sent %d shares for %d ciphertexts", p, len(s), len(cts))
		}
	}
	slots := Slots(params)
	if nValues > len(cts)*slots {
		return nil, fmt.Errorf("asked for %d values but only %d slots", nValues, len(cts)*slots)
	}

	out := make([]float64, len(cts)*slots)
	type worker struct {
		ecd *ckks.Encoder
		dec *rlwe.Decryptor
	}
	err = parallelFor(len(cts), func() (worker, error) {
		return worker{newEncoder(params), rlwe.NewDecryptor(params, rlwe.NewSecretKey(params))}, nil // the zero key
	}, func(w worker, k int) error {
		ct := cts[k]
		agg := cks.AllocateShare(ct.Level())
		for _, s := range shares {
			if err := cks.AggregateShares(agg, s[k], &agg); err != nil {
				return fmt.Errorf("aggregate chunk %d: %w", k, err)
			}
		}
		switched := ckks.NewCiphertext(params, 1, ct.Level())
		cks.KeySwitch(ct, agg, switched)
		if err := w.ecd.Decode(w.dec.DecryptNew(switched), out[k*slots:(k+1)*slots]); err != nil {
			return fmt.Errorf("decode chunk %d: %w", k, err)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return out[:nValues], nil
}
