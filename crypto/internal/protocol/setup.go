package protocol

import (
	"fmt"
	"slices"

	"github.com/4uhn/Meta-AI-Hackathon/crypto/internal/he"
)

// ParamsFile is public/params.json. ParamsSHA256 pins the exact Lattigo parameters, so
// a party whose library or config differs fails loudly instead of silently mis-decrypting.
type ParamsFile struct {
	LogN         int    `json:"log_n"`
	LogQ         []int  `json:"log_q"`
	LogScale     int    `json:"log_scale"`
	RingType     string `json:"ring_type"`
	FloodLogW    int    `json:"flood_log_width"`
	StatSecurity int    `json:"stat_security"`
	ParamsSHA256 string `json:"params_sha256"`
}

// Parties is public/parties.json: names and Shamir points, plus the threshold.
type Parties struct {
	Threshold int     `json:"threshold"`
	Parties   []Party `json:"parties"`
}

type Party struct {
	Name  string `json:"name"`
	Point uint64 `json:"point"`
}

// DefaultParties is op_a=1, op_b=2, op_c=3 with t = 2.
func DefaultParties() Parties {
	return Parties{Threshold: 2, Parties: []Party{{"op_a", 1}, {"op_b", 2}, {"op_c", 3}}}
}

// NewParties assigns Shamir points 1..N to names in order, with threshold t.
func NewParties(names []string, t int) (Parties, error) {
	p := Parties{Threshold: t}
	for i, n := range names {
		p.Parties = append(p.Parties, Party{n, uint64(i + 1)})
	}
	return p, p.Validate()
}

// N is the number of parties.
func (p Parties) N() int { return len(p.Parties) }

// MinContributors is the fewest distinct signed submissions an aggregate needs before
// any party helps decrypt it: max(3, t+1). The decrypted sum is revealed to every member,
// so any t−1 colluding contributors can subtract their own updates from it. With at
// least t+1 contributors they are left with a sum of at least two others, never an
// individual update.
func (p Parties) MinContributors() int { return max(MinContributors, p.Threshold+1) }

// Point returns the Shamir point of a party.
func (p Parties) Point(name string) (he.Point, error) {
	for _, x := range p.Parties {
		if x.Name == name {
			return he.Point(x.Point), nil
		}
	}
	return 0, fmt.Errorf("unknown party %q", name)
}

// Names returns the party names in order.
func (p Parties) Names() []string {
	out := make([]string, len(p.Parties))
	for i, x := range p.Parties {
		out[i] = x.Name
	}
	return out
}

// Points returns all Shamir points.
func (p Parties) Points() []he.Point {
	out := make([]he.Point, len(p.Parties))
	for i, x := range p.Parties {
		out[i] = he.Point(x.Point)
	}
	return out
}

// Validate checks names, distinct non-zero points and the access structure:
// 2 ≤ t ≤ N − 1, MinParties ≤ N ≤ MaxParties, and 2t > N. 2t > N makes any two
// t-subsets overlap: each party releases one decryption share per round, so an
// aggregator cannot have two different aggregates of the same round decrypted by
// disjoint subsets and difference them. t ≤ N − 1 because an aggregate needs t + 1
// contributors (MinContributors), which t = N could never reach.
func (p Parties) Validate() error {
	n, t := len(p.Parties), p.Threshold
	if n < he.MinParties || n > he.MaxParties {
		return fmt.Errorf("parties.json: %d parties, this build supports %d to %d", n, he.MinParties, he.MaxParties)
	}
	if t < 2 || t > n-1 || 2*t <= n {
		return fmt.Errorf("parties.json: threshold %d-of-%d, need 2 ≤ t ≤ N − 1 and 2t > N", t, n)
	}
	var pts []uint64
	for _, x := range p.Parties {
		if !ValidParty(x.Name) {
			return fmt.Errorf("invalid party name %q", x.Name)
		}
		if x.Point == 0 || slices.Contains(pts, x.Point) {
			return fmt.Errorf("party %s: Shamir point %d must be non-zero and distinct", x.Name, x.Point)
		}
		pts = append(pts, x.Point)
	}
	return nil
}
