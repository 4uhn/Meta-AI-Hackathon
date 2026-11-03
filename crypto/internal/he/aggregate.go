package he

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Aggregate sums the submissions chunk by chunk. It runs on the aggregator and, to verify
// the aggregate, on each operator. Addition is the only homomorphic operation, so there
// are no evaluation keys and the multiplicative depth is zero.
//
// The sum is taken directly on the ring (c0 and c1 mod Q), skipping ckks.NewEvaluator's
// unused precomputation, after the metadata checks the evaluator would make.
func Aggregate(params ckks.Parameters, submissions [][]*rlwe.Ciphertext) ([]*rlwe.Ciphertext, error) {
	if len(submissions) == 0 {
		return nil, fmt.Errorf("no submissions")
	}
	n := len(submissions[0])
	for i, s := range submissions {
		if len(s) != n {
			return nil, fmt.Errorf("submission %d has %d ciphertexts, want %d", i, len(s), n)
		}
	}
	ringQ := params.RingQ().AtLevel(params.MaxLevel())
	ref := submissions[0][0]
	out := make([]*rlwe.Ciphertext, n)
	err := parallelFor(n, func() (struct{}, error) { return struct{}{}, nil }, func(_ struct{}, k int) error {
		for i, s := range submissions {
			ct := s[k]
			if ct.Degree() != 1 || ct.Level() != params.MaxLevel() || ct.IsNTT != ref.IsNTT ||
				ct.Scale.Cmp(ref.Scale) != 0 || ct.LogDimensions != ref.LogDimensions {
				return fmt.Errorf("submission %d chunk %d: ciphertext metadata does not match", i, k)
			}
		}
		out[k] = submissions[0][k].CopyNew()
		for _, s := range submissions[1:] {
			ringQ.Add(out[k].Value[0], s[k].Value[0], out[k].Value[0])
			ringQ.Add(out[k].Value[1], s[k].Value[1], out[k].Value[1])
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}
