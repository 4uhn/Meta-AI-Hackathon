package protocol

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"time"
)

// DecryptRequest is written by the aggregator to bus/round_r/decrypt/request.json.
// The active set is fixed here, before any share exists.
type DecryptRequest struct {
	RunID        string   `json:"run_id"`
	Round        int      `json:"round"`
	Active       []string `json:"active"`       // exactly t operators
	Contributors []string `json:"contributors"` // whose submissions make up the aggregate
	AggSHA256    string   `json:"agg_sha256"`   // sha256 of ct_agg.bin
}

// Check validates the request's shape against the party list.
func (q DecryptRequest) Check(p Parties) error {
	if len(q.Active) != p.Threshold {
		return fmt.Errorf("active set %v: want exactly %d parties", q.Active, p.Threshold)
	}
	if len(slices.Compact(slices.Sorted(slices.Values(q.Active)))) != len(q.Active) {
		return fmt.Errorf("active set %v repeats a party", q.Active)
	}
	for _, a := range q.Active {
		if _, err := p.Point(a); err != nil {
			return err
		}
	}
	if need := p.MinContributors(); len(q.Contributors) < need {
		return fmt.Errorf("aggregate has %d contributors, need ≥ %d", len(q.Contributors), need)
	}
	seen := map[string]bool{}
	for _, c := range q.Contributors {
		if _, err := p.Point(c); err != nil {
			return err
		}
		if seen[c] {
			return fmt.Errorf("contributor %s listed twice", c)
		}
		seen[c] = true
	}
	if len(q.AggSHA256) != 64 {
		return fmt.Errorf("agg_sha256 %q: want 64 hex characters", q.AggSHA256)
	}
	return nil
}

// MinContributors is the floor on distinct signed submissions an operator will help
// decrypt the sum of. Each contributor can subtract its own update from the result, so
// with 2 contributors each learns the other's update; 3 is the smallest safe value.
// Parties.MinContributors raises it to t+1 against t−1 colluders. In 2-of-3 every
// operator must therefore submit: a round missing a submission is skipped, never decrypted.
const MinContributors = 3

// ShareReceipt is written next to decshare_<op>.bin. The aggregator checks that every
// receipt names the same aggregate it holds.
type ShareReceipt struct {
	RunID       string   `json:"run_id"`
	Round       int      `json:"round"`
	Party       string   `json:"party"`
	Active      []string `json:"active"`
	AggSHA256   string   `json:"agg_sha256"`
	ShareSHA256 string   `json:"share_sha256"`
}

// LogEntry is one line of parties/<op>/state/decrypt_log.jsonl.
type LogEntry struct {
	Time         string   `json:"time"`
	RunID        string   `json:"run_id"`
	Round        int      `json:"round"`
	Active       []string `json:"active"`
	Contributors []string `json:"contributors"`
	AggSHA256    string   `json:"agg_sha256"`
}

// DecryptLog is an operator's append-only record of every decryption share it has
// released. It allows at most one share per (run, round) and never two shares for the
// same ciphertext, whatever the active set. Lattigo's SECURITY.md warns that re-running a
// threshold decryption, even with another subset, can leak the key, and a second request
// in the same round could difference two aggregates ({A,B} vs {A,B,C}) to isolate C.
type DecryptLog struct{ path string }

func OpenDecryptLog(path string) *DecryptLog { return &DecryptLog{path: path} }

// Authorise checks the request against every past entry and, if it is fresh, appends
// and fsyncs the entry before returning. The caller computes the share only after
// this succeeds, so a crash can at worst lose a share, never release a second one.
func (l *DecryptLog) Authorise(q DecryptRequest) error {
	entries, err := l.entries()
	if err != nil {
		return err
	}
	for _, e := range entries {
		if e.RunID == q.RunID && e.Round == q.Round {
			return fmt.Errorf("one-shot: already shared for run %s round %d (active %v, at %s)", e.RunID, e.Round, e.Active, e.Time)
		}
		if e.AggSHA256 == q.AggSHA256 {
			return fmt.Errorf("one-shot: already shared for ciphertext %s… (run %s round %d)", q.AggSHA256[:12], e.RunID, e.Round)
		}
	}
	if err := os.MkdirAll(filepath.Dir(l.path), 0o700); err != nil {
		return err
	}
	f, err := os.OpenFile(l.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()
	b, err := json.Marshal(LogEntry{
		Time: time.Now().UTC().Format(time.RFC3339Nano), RunID: q.RunID, Round: q.Round,
		Active: slices.Clone(q.Active), Contributors: slices.Clone(q.Contributors), AggSHA256: q.AggSHA256,
	})
	if err != nil {
		return err
	}
	if _, err := f.Write(append(b, '\n')); err != nil {
		return err
	}
	return f.Sync()
}

func (l *DecryptLog) entries() ([]LogEntry, error) {
	f, err := os.Open(l.path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []LogEntry
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var e LogEntry
		if err := json.Unmarshal(sc.Bytes(), &e); err != nil {
			return nil, fmt.Errorf("corrupt decrypt log %s: %w", l.path, err)
		}
		out = append(out, e)
	}
	return out, sc.Err()
}
