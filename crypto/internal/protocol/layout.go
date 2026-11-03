// Package protocol holds everything around the cryptography: the run-directory layout,
// file access that enforces party isolation, signed submission statements and the
// one-shot decryption log.
package protocol

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
)

// Aggregator is the actor name of the Horizon server.
const Aggregator = "aggregator"

var partyName = regexp.MustCompile(`^op_[a-z0-9]+$`)

// ValidParty reports whether name is a well-formed operator name.
func ValidParty(name string) bool { return partyName.MatchString(name) }

// Run is a run directory seen through the eyes of one actor. Every file the CLI touches
// goes through Path, which refuses paths in another party's private area, so "the
// aggregator never opens a secret/ directory" is a check rather than a convention.
type Run struct {
	Root  string
	Actor string // "aggregator" or an operator name
}

// NewRun validates the actor and returns the run as seen by that actor.
func NewRun(root, actor string) (*Run, error) {
	if actor != Aggregator && !ValidParty(actor) {
		return nil, fmt.Errorf("invalid actor %q", actor)
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	return &Run{Root: abs, Actor: actor}, nil
}

// Path joins elements under the run root and checks the actor may access the result.
//
// Private areas: parties/<op>/... and bus/private/<op>/... belong to <op> only.
func (r *Run) Path(elem ...string) (string, error) {
	p := filepath.Join(append([]string{r.Root}, elem...)...)
	rel, err := filepath.Rel(r.Root, p)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path %q escapes the run directory", p)
	}
	parts := strings.Split(filepath.ToSlash(rel), "/")
	owner := ""
	switch {
	case len(parts) >= 2 && parts[0] == "parties":
		owner = parts[1]
	case len(parts) >= 3 && parts[0] == "bus" && parts[1] == "private":
		owner = parts[2]
	}
	if owner != "" && owner != r.Actor {
		return "", fmt.Errorf("isolation: %s may not access %s (private to %s)", r.Actor, rel, owner)
	}
	return p, nil
}

// Standard locations in a run directory.

func (r *Run) ParamsFile() (string, error)  { return r.Path("public", "params.json") }
func (r *Run) CRSFile() (string, error)     { return r.Path("public", "crs_seed") }
func (r *Run) PartiesFile() (string, error) { return r.Path("public", "parties.json") }
func (r *Run) PKFile() (string, error)      { return r.Path("public", "pk.bin") }
func (r *Run) SignPubFile(op string) (string, error) {
	return r.Path("public", "sign_"+op+".pub")
}
func (r *Run) SecretFile(name string) (string, error) {
	return r.Path("parties", r.Actor, "secret", name)
}
func (r *Run) StateFile(name string) (string, error) {
	return r.Path("parties", r.Actor, "state", name)
}
func (r *Run) InboxFile(to, from string) (string, error) {
	return r.Path("bus", "private", to, "from_"+from+".bin")
}
func (r *Run) PKShareFile(op string) (string, error) {
	return r.Path("bus", "setup", "pkshare_"+op+".bin")
}
func (r *Run) RoundDir(round int, elem ...string) (string, error) {
	return r.Path(append([]string{"bus", fmt.Sprintf("round_%d", round)}, elem...)...)
}
func (r *Run) SubmissionFile(round int, op, name string) (string, error) {
	return r.RoundDir(round, "submissions", op, name)
}
