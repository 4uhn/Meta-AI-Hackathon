package he

import (
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Encrypt runs on one operator of a consortium of `parties` operators. It splits the vector into
// chunks of Slots(params) real values and encrypts each chunk under the collective public
// key. Nobody holds the matching secret key s = Σ sk_i, so not even the operator can
// decrypt its own ciphertexts.
func Encrypt(params ckks.Parameters, pk *rlwe.PublicKey, parties int, values []float64) ([]*rlwe.Ciphertext, error) {
	maxAbs := MaxAbs(parties)
	for i, v := range values {
		if math.IsNaN(v) || math.Abs(v) > maxAbs {
			return nil, fmt.Errorf("value[%d] = %v outside [−%v, %v]", i, v, maxAbs, maxAbs)
		}
	}
	slots := Slots(params)
	n := (len(values) + slots - 1) / slots
	out := make([]*rlwe.Ciphertext, n)
	type worker struct {
		ecd *ckks.Encoder
		enc *rlwe.Encryptor
	}
	err := parallelFor(n, func() (worker, error) {
		return worker{newEncoder(params), rlwe.NewEncryptor(params, pk)}, nil
	}, func(w worker, k int) error {
		start, end := k*slots, min((k+1)*slots, len(values))
		pt := ckks.NewPlaintext(params, params.MaxLevel())
		if err := w.ecd.Encode(values[start:end], pt); err != nil {
			return fmt.Errorf("encode chunk %d: %w", k, err)
		}
		ct, err := w.enc.EncryptNew(pt)
		if err != nil {
			return fmt.Errorf("encrypt chunk %d: %w", k, err)
		}
		out[k] = ct
		return nil
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}
