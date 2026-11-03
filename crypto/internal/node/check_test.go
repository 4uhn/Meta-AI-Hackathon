package node

import (
	"path/filepath"
	"testing"
)

func TestCheckShapeRules(t *testing.T) {
	cases := []struct {
		n, t       int
		unverified bool
		allowed    bool
		failing    string // name prefix of the rule that must fail
	}{
		{3, 2, false, true, ""},
		{4, 3, false, true, ""},
		{5, 3, false, true, ""},
		{5, 4, false, true, ""},
		{3, 3, false, false, "t ≤ N − 1"},
		{3, 1, false, false, "t ≥ 2"},
		{5, 2, false, false, "2t > N"},
		{6, 5, false, false, "N ≤ 5 measured"},
		{6, 5, true, true, ""},
		{7, 6, true, true, ""},
		{8, 7, true, false, "3 ≤ N ≤ 7"},
	}
	for _, c := range cases {
		res, _, ok := CheckShape(c.n, c.t, c.unverified)
		if ok != c.allowed {
			t.Errorf("%d-of-%d (unverified=%v): allowed=%v, want %v", c.t, c.n, c.unverified, ok, c.allowed)
		}
		if c.failing == "" {
			continue
		}
		found := false
		for _, r := range res["rules"].([]rule) {
			if !r.OK && len(r.Name) >= len(c.failing) && r.Name[:len(c.failing)] == c.failing {
				found = true
			}
		}
		if !found {
			t.Errorf("%d-of-%d: rule %q should fail", c.t, c.n, c.failing)
		}
	}
}

func TestCheckShapeDefaultParameters(t *testing.T) {
	res, _, ok := CheckShape(3, 2, false)
	p := res["params"].(Result)
	if !ok || p["flood_log_width"] != 85 || p["headroom"] != 32.0 || p["headroom_needed"] != 24.0 ||
		p["max_abs_value"] != 4.0 || p["min_contributors"] != 3 || p["err_tolerance"] != 1e-6 {
		t.Fatalf("2-of-3 parameters changed: %v", p)
	}
}

func TestCheckConsortiumRunsTheSelftest(t *testing.T) {
	res, _, err := CheckConsortium(filepath.Join(t.TempDir(), "run"), 3000, 4, 3, false)
	if err != nil || res["selftest"].(Result)["ok_all"] != true {
		t.Fatalf("3-of-4 check failed: %v", err)
	}
	if _, _, err := CheckConsortium(filepath.Join(t.TempDir(), "run"), 3000, 3, 3, false); err == nil {
		t.Fatal("3-of-3 accepted")
	}
}
