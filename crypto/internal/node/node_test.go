package node

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/4uhn/Meta-AI-Hackathon/crypto/internal/he"
	"github.com/4uhn/Meta-AI-Hackathon/crypto/internal/protocol"
)

// golden2of3Checks are the check names of the default (2-of-3) selftest, which must
// stay exactly these.
var golden2of3Checks = []string{
	"isolation",
	"sk erased after setup",
	"round 1: submit [op_a op_b op_c], decrypt with [op_a op_b]",
	"round 1: op_a's share alone is noise",
	"round 2: submit [op_a op_b op_c], decrypt with [op_a op_c]",
	"round 2: op_a's share alone is noise",
	"round 3: submit [op_a op_b op_c], decrypt with [op_b op_c]",
	"round 3: op_b's share alone is noise",
	"one-shot: same request again",
	"one-shot: new active set, same round",
	"round 4: op_c skipped → aggregation refused (2 contributors)",
	"refuse: genuine 2-contributor sum (each would learn the other's update)",
	"refuse: individual ciphertext labelled as aggregate",
	"refuse: aggregate with a single contributor",
	"reject: tampered ct (op_c) and replayed round-1 submission (op_b)",
}

func runSelftest(t *testing.T, parties protocol.Parties) []Check {
	t.Helper()
	res, err := Selftest(filepath.Join(t.TempDir(), "run"), 3*16384+7, parties)
	if err != nil {
		t.Fatal(err)
	}
	checks := res["checks"].([]Check)
	for _, c := range checks {
		if !c.OK {
			t.Errorf("FAIL %s: %s", c.Name, c.Detail)
		} else {
			t.Logf("PASS %s: %s", c.Name, c.Detail)
		}
	}
	return checks
}

// TestSelftest runs the full file-based protocol plus attack checks (CLI `selftest`)
// at a small payload, so `make test` covers the node layer end to end.
func TestSelftest(t *testing.T) {
	var names []string
	for _, c := range runSelftest(t, protocol.DefaultParties()) {
		names = append(names, c.Name)
	}
	if !slices.Equal(names, golden2of3Checks) {
		t.Errorf("2-of-3 check list changed:\n got %q\nwant %q", names, golden2of3Checks)
	}
}

// TestSelftestTofN runs the same protocol and attacks for every larger shape we claim.
func TestSelftestTofN(t *testing.T) {
	all := []string{"op_a", "op_b", "op_c", "op_d", "op_e"}
	for _, c := range [][2]int{{4, 3}, {5, 3}, {5, 4}} {
		t.Run(fmt.Sprintf("%dof%d", c[1], c[0]), func(t *testing.T) {
			p, err := protocol.NewParties(all[:c[0]], c[1])
			if err != nil {
				t.Fatal(err)
			}
			checks := runSelftest(t, p)
			rounds := 0
			for _, ch := range checks {
				if len(ch.Name) > 6 && ch.Name[:6] == "round " && bytes.Contains([]byte(ch.Name), []byte("decrypt with")) {
					rounds++
				}
			}
			if want := len(combinations(all[:c[0]], c[1])); rounds != want {
				t.Errorf("%d decrypting rounds, want one per t-subset (%d)", rounds, want)
			}
		})
	}
}

// TestGolden2of3Files: with default flags, init writes params.json and parties.json
// byte-identical to the files every 2-of-3 run carries, and those files still load.
func TestGolden2of3Files(t *testing.T) {
	root := filepath.Join(t.TempDir(), "run")
	if _, err := Init(root, he.DefaultConfig, protocol.DefaultParties()); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"params.json", "parties.json"} {
		got, err := os.ReadFile(filepath.Join(root, "public", f))
		if err != nil {
			t.Fatal(err)
		}
		want, err := os.ReadFile(filepath.Join("testdata", "golden_2of3", f))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Errorf("%s differs from the 2-of-3 golden file:\n%s", f, got)
		}
	}

	old := filepath.Join(t.TempDir(), "old")
	if err := os.MkdirAll(filepath.Join(old, "public"), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"params.json", "parties.json"} {
		b, _ := os.ReadFile(filepath.Join("testdata", "golden_2of3", f))
		if err := os.WriteFile(filepath.Join(old, "public", f), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for _, actor := range []string{protocol.Aggregator, "op_a", "op_b", "op_c"} {
		c, err := Load(old, actor)
		if err != nil {
			t.Fatalf("golden run as %s: %v", actor, err)
		}
		if c.ParamsSHA256[:16] != "ed03de312c948e0d" || c.Parties.Threshold != 2 || c.Parties.N() != 3 {
			t.Fatalf("golden run loads as %s %d-of-%d", c.ParamsSHA256[:16], c.Parties.Threshold, c.Parties.N())
		}
	}
	if _, err := Load(old, "op_d"); err == nil {
		t.Error("op_d accepted in a 2-of-3 run")
	}

	// A 3-of-4 party list next to 2-of-3 params (flood width 85) must not load: the flood
	// would be sized for the wrong number of parties.
	p4, _ := protocol.NewParties([]string{"op_a", "op_b", "op_c", "op_d"}, 3)
	if err := protocol.WriteJSON(filepath.Join(old, "public", "parties.json"), p4, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(old, protocol.Aggregator); err == nil {
		t.Error("3-of-4 parties loaded with a 2-of-3 flood width")
	}
}
