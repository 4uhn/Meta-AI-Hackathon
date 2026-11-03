package protocol

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestPathIsolation(t *testing.T) {
	root := t.TempDir()
	cases := []struct {
		actor string
		elem  []string
		ok    bool
	}{
		{Aggregator, []string{"public", "pk.bin"}, true},
		{Aggregator, []string{"bus", "round_1", "aggregate", "ct_agg.bin"}, true},
		{Aggregator, []string{"parties", "op_a", "secret", "tsk.bin"}, false},
		{Aggregator, []string{"parties", "op_a", "state", "decrypt_log.jsonl"}, false},
		{Aggregator, []string{"bus", "private", "op_b", "from_op_a.bin"}, false},
		{"op_a", []string{"parties", "op_a", "secret", "tsk.bin"}, true},
		{"op_a", []string{"bus", "private", "op_a", "from_op_c.bin"}, true},
		{"op_a", []string{"parties", "op_b", "secret", "tsk.bin"}, false},
		{"op_a", []string{"bus", "private", "op_c", "from_op_b.bin"}, false},
		{"op_a", []string{"..", "elsewhere"}, false},
		{"op_a", []string{"public", "..", "parties", "op_b", "secret"}, false},
	}
	for _, c := range cases {
		r, err := NewRun(root, c.actor)
		if err != nil {
			t.Fatal(err)
		}
		_, err = r.Path(c.elem...)
		if (err == nil) != c.ok {
			t.Errorf("%s → %s: allowed=%v, want %v (%v)", c.actor, strings.Join(c.elem, "/"), err == nil, c.ok, err)
		}
	}
	if _, err := NewRun(root, "../op_a"); err == nil {
		t.Error("malformed actor accepted")
	}
}

func TestStatementSignature(t *testing.T) {
	pub, priv, err := GenSigningKey()
	if err != nil {
		t.Fatal(err)
	}
	s := SubmissionStatement{RunID: "r", Round: 2, Operator: "op_a", CtSHA256: "ab", NValues: 10, NCiphertexts: 1}
	ss, err := Sign(s, priv)
	if err != nil {
		t.Fatal(err)
	}
	if err := ss.Verify(pub); err != nil {
		t.Fatalf("valid signature rejected: %v", err)
	}
	for name, mut := range map[string]func(*SignedStatement){
		"round":    func(x *SignedStatement) { x.Statement.Round = 3 },
		"operator": func(x *SignedStatement) { x.Statement.Operator = "op_b" },
		"hash":     func(x *SignedStatement) { x.Statement.CtSHA256 = "cd" },
	} {
		x := ss
		mut(&x)
		if x.Verify(pub) == nil {
			t.Errorf("tampered %s accepted", name)
		}
	}
	otherPub, _, _ := GenSigningKey()
	if ss.Verify(otherPub) == nil {
		t.Error("signature verified under another operator's key")
	}
}

func TestDecryptLogOneShot(t *testing.T) {
	l := OpenDecryptLog(filepath.Join(t.TempDir(), "state", "decrypt_log.jsonl"))
	q := DecryptRequest{RunID: "r", Round: 1, Active: []string{"op_a", "op_b"}, Contributors: []string{"op_a", "op_b", "op_c"}, AggSHA256: strings.Repeat("a", 64)}
	if err := l.Authorise(q); err != nil {
		t.Fatal(err)
	}
	q2 := q
	q2.Active = []string{"op_a", "op_c"}
	if l.Authorise(q2) == nil {
		t.Error("second share for the same round authorised")
	}
	q3 := q
	q3.Round = 2
	if l.Authorise(q3) == nil {
		t.Error("second share for the same ciphertext authorised")
	}
	q4 := q3
	q4.AggSHA256 = strings.Repeat("b", 64)
	if err := l.Authorise(q4); err != nil {
		t.Errorf("fresh round refused: %v", err)
	}
}

func TestRequestCheck(t *testing.T) {
	p := DefaultParties()
	all3 := []string{"op_a", "op_b", "op_c"}
	h := strings.Repeat("a", 64)
	ok := DecryptRequest{Active: []string{"op_a", "op_c"}, Contributors: all3, AggSHA256: h}
	if err := ok.Check(p); err != nil {
		t.Fatal(err)
	}
	// Each bad request breaks exactly one rule, so each rule is tested on its own.
	bad := []DecryptRequest{
		{Active: []string{"op_a"}, Contributors: all3},                                     // active set too small
		{Active: []string{"op_a", "op_a"}, Contributors: all3},                             // repeated active party
		{Active: []string{"op_a", "op_z"}, Contributors: all3},                             // unknown active party
		{Active: []string{"op_a", "op_b"}, Contributors: []string{"op_a"}},                 // one contributor
		{Active: []string{"op_a", "op_b"}, Contributors: []string{"op_a", "op_b"}},         // two: each learns the other's update
		{Active: []string{"op_a", "op_b"}, Contributors: []string{"op_a", "op_a", "op_b"}}, // duplicate contributor
		{Active: []string{"op_a", "op_b"}, Contributors: []string{"op_a", "op_b", "op_z"}}, // unknown contributor
		{Active: []string{"op_a", "op_b"}, Contributors: all3, AggSHA256: "abc"},           // malformed hash
	}
	for i, q := range bad {
		if q.Check(p) == nil {
			t.Errorf("bad request %d accepted: %+v", i, q)
		}
	}
}

func TestValidateAccessStructure(t *testing.T) {
	names := func(n int) []string {
		var out []string
		for i := range n {
			out = append(out, "op_"+string(rune('a'+i)))
		}
		return out
	}
	for _, c := range [][2]int{{3, 2}, {4, 3}, {5, 3}, {5, 4}, {7, 4}, {7, 6}} {
		p, err := NewParties(names(c[0]), c[1])
		if err != nil {
			t.Errorf("%d-of-%d rejected: %v", c[1], c[0], err)
		}
		if want := max(3, c[1]+1); p.MinContributors() != want {
			t.Errorf("%d-of-%d: min contributors %d, want %d", c[1], c[0], p.MinContributors(), want)
		}
	}
	// t = 1, t > N, t = N (needs N + 1 contributors), 2t ≤ N (disjoint t-subsets), N < 3, N > 7.
	for _, c := range [][2]int{{3, 1}, {3, 4}, {3, 3}, {5, 5}, {4, 2}, {5, 2}, {6, 3}, {2, 2}, {8, 5}} {
		if _, err := NewParties(names(c[0]), c[1]); err == nil {
			t.Errorf("%d-of-%d accepted", c[1], c[0])
		}
	}
	dup := Parties{Threshold: 3, Parties: []Party{{"op_a", 1}, {"op_b", 2}, {"op_c", 2}, {"op_d", 4}}}
	zero := Parties{Threshold: 3, Parties: []Party{{"op_a", 0}, {"op_b", 2}, {"op_c", 3}, {"op_d", 4}}}
	bad := Parties{Threshold: 3, Parties: []Party{{"op_a", 1}, {"op_b", 2}, {"op_c", 3}, {"OP_D", 4}}}
	for name, p := range map[string]Parties{"duplicate point": dup, "zero point": zero, "bad name": bad} {
		if p.Validate() == nil {
			t.Errorf("%s accepted", name)
		}
	}
	if DefaultParties().Validate() != nil || DefaultParties().MinContributors() != 3 {
		t.Error("default 2-of-3 changed")
	}
}

func TestRequestCheckThreeOfFour(t *testing.T) {
	p, err := NewParties([]string{"op_a", "op_b", "op_c", "op_d"}, 3)
	if err != nil {
		t.Fatal(err)
	}
	all4 := []string{"op_a", "op_b", "op_c", "op_d"}
	ok := DecryptRequest{Active: []string{"op_b", "op_c", "op_d"}, Contributors: all4, AggSHA256: strings.Repeat("a", 64)}
	if err := ok.Check(p); err != nil {
		t.Fatal(err)
	}
	bad := []DecryptRequest{
		{Active: []string{"op_a", "op_b"}, Contributors: all4},                                     // t−1 shares
		{Active: []string{"op_a", "op_b", "op_c", "op_d"}, Contributors: all4},                     // more than t
		{Active: []string{"op_a", "op_b", "op_a"}, Contributors: all4},                             // repeated
		{Active: []string{"op_a", "op_b", "op_c"}, Contributors: []string{"op_a", "op_b", "op_c"}}, // 3 < t+1: two colluders would learn the third
	}
	for i, q := range bad {
		if q.Check(p) == nil {
			t.Errorf("bad request %d accepted: %+v", i, q)
		}
	}
}
