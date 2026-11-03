package he

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/multiparty"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

// sampleCRP expands the public CRS seed into the common random polynomial `a`. A fresh
// KeyedPRNG per call guarantees every party derives the identical `a`.
func sampleCRP(params ckks.Parameters, crsSeed []byte) (multiparty.PublicKeyGenProtocol, multiparty.PublicKeyGenCRP, error) {
	ckg := multiparty.NewPublicKeyGenProtocol(params)
	crs, err := sampling.NewKeyedPRNG(crsSeed)
	if err != nil {
		return ckg, multiparty.PublicKeyGenCRP{}, fmt.Errorf("crs prng: %w", err)
	}
	return ckg, ckg.SampleCRP(crs), nil
}

// PublicKeyShare runs on one operator: -a·sk_i + e_i. It uses the original additive key
// sk_i (every operator is online for the setup ceremony), so the aggregate is a public
// key for s = Σ sk_i = S(0), the same secret the Shamir shares reconstruct.
func PublicKeyShare(params ckks.Parameters, sk *rlwe.SecretKey, crsSeed []byte) (multiparty.PublicKeyGenShare, error) {
	ckg, crp, err := sampleCRP(params, crsSeed)
	if err != nil {
		return multiparty.PublicKeyGenShare{}, err
	}
	share := ckg.AllocateShare()
	ckg.GenShare(sk, crp, &share)
	return share, nil
}

// CombinePublicKey runs on the aggregator: it only ever sees public shares.
func CombinePublicKey(params ckks.Parameters, crsSeed []byte, shares []multiparty.PublicKeyGenShare) (*rlwe.PublicKey, error) {
	if len(shares) == 0 {
		return nil, fmt.Errorf("no public key shares")
	}
	ckg, crp, err := sampleCRP(params, crsSeed)
	if err != nil {
		return nil, err
	}
	agg := ckg.AllocateShare()
	for _, s := range shares {
		ckg.AggregateShares(agg, s, &agg)
	}
	pk := rlwe.NewPublicKey(params)
	ckg.GenPublicKey(agg, crp, pk)
	return pk, nil
}
