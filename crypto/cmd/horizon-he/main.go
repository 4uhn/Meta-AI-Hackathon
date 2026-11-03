// Command horizon-he is Horizon's t-of-N threshold-CKKS tool. Every subcommand acts as exactly
// one actor (the aggregator or one operator) on a run directory, does one protocol step
// and prints a one-line JSON summary on stdout: {"ok":true,"cmd":...,...}. On failure it
// prints {"ok":false,"error":...} and exits 1.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/4uhn/Meta-AI-Hackathon/crypto/internal/he"
	"github.com/4uhn/Meta-AI-Hackathon/crypto/internal/node"
	"github.com/4uhn/Meta-AI-Hackathon/crypto/internal/protocol"
)

const usage = `horizon-he <command> [flags]

Setup (key ceremony, once per run):
  init        --run DIR [--parties L --threshold T]  aggregator: params.json, crs_seed, parties.json
  keygen      --run DIR --party OP               operator:   secret key + Ed25519 signing key
  pk-share    --run DIR --party OP               operator:   public key share
  shamir-deal --run DIR --party OP               operator:   send Shamir shares to private inboxes
  shamir-agg  --run DIR --party OP               operator:   inbox → tsk.bin; erase sk.bin and inbox
  pk-combine  --run DIR                          aggregator: public/pk.bin
  ceremony    --run DIR [--parties L --threshold T]  all of the above, in order (demo convenience)

Round r:
  encrypt     --run DIR --party OP --round R --in update.f64
  aggregate   --run DIR --round R                aggregator: verify + Σ ciphertexts
  request     --run DIR --round R --active op_a,op_c
  dec-share   --run DIR --party OP --round R     operator:   verify, log (one-shot), share
  dec-combine --run DIR --round R [--solo OP[,OP…]]  aggregator: agg.f64 (or fewer-than-t noise demo)

Checks:
  selftest    [--n N] [--keep DIR] [--parties L --threshold T]  protocol + attack checks, synthetic data
  bench       [--n N] [--out FILE] [--parties L --threshold T]  crypto benchmark (default n = Glimmer payload)
  scaling     [--n N] [--out FILE]               selftest + bench for 2-of-3, 3-of-4, 3-of-5, 4-of-5
  check       --parties N --threshold T [--n N] [--unverified]  is this t-of-N safe? rules, parameters,
              then the full selftest. N ≤ 5 is measured; 6–7 need --unverified; N > 7 is not supported.

--parties takes a count (5 = op_a,…,op_e) or a list. Default: op_a,op_b,op_c with threshold 2
(Shamir points 1..N in list order).
`

// GlimmerPayload is lora_B at r=8 on q/k/v/o over 52 layers (4,685,824) plus the weight slot.
const GlimmerPayload = 4_685_824 + 1

func main() {
	if len(os.Args) < 2 || os.Args[1] == "-h" || os.Args[1] == "--help" {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(2)
	}
	cmd := os.Args[1]
	fs := flag.NewFlagSet(cmd, flag.ExitOnError)
	run := fs.String("run", "", "run directory")
	party := fs.String("party", "", "acting operator (op_a, op_b, ...)")
	round := fs.Int("round", 0, "round number (≥ 1)")
	in := fs.String("in", "", "input update.f64")
	active := fs.String("active", "", "comma-separated active set for decryption")
	solo := fs.String("solo", "", "dec-combine: use only these operators' shares (OP[,OP…], fewer than t)")
	n := fs.Int("n", GlimmerPayload, "selftest/bench payload length")
	keep := fs.String("keep", "", "selftest/bench: run directory to keep (default: temp, deleted)")
	out := fs.String("out", "", "bench/scaling: also write the JSON result to this file")
	partyList := fs.String("parties", "", "init/ceremony/selftest/bench: comma-separated operators (default op_a,op_b,op_c)")
	threshold := fs.Int("threshold", 0, "init/ceremony/selftest/bench: t in t-of-N (default 2)")
	unverified := fs.Bool("unverified", false, "check: allow a consortium larger than the measured ones")
	fs.Parse(os.Args[2:])

	if cmd == "check" {
		res, err := check(*partyList, *threshold, *n, *unverified)
		emit(cmd, res, err)
		return
	}
	res, err := dispatch(cmd, *run, *party, *round, *in, *active, *solo, *n, *keep, *out, *partyList, *threshold)
	emit(cmd, res, err)
}

// check runs `horizon-he check` in a temporary directory; the report goes to stderr.
func check(list string, t, n int, unverified bool) (node.Result, error) {
	if list == "" || t == 0 {
		return nil, fmt.Errorf("check: give --parties N and --threshold T")
	}
	count := len(strings.Split(list, ","))
	if k, err := strconv.Atoi(list); err == nil {
		count = k
	}
	tmp, err := os.MkdirTemp("", "horizon-check-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmp)
	res, text, err := node.CheckConsortium(filepath.Join(tmp, "run"), n, count, t, unverified)
	fmt.Fprint(os.Stderr, text)
	return res, err
}

// parties builds the party list from --parties/--threshold; neither flag means 2-of-3.
func parties(list string, t int) (protocol.Parties, error) {
	if list == "" && t == 0 {
		return protocol.DefaultParties(), nil
	}
	if list == "" || t == 0 {
		return protocol.Parties{}, fmt.Errorf("--parties and --threshold go together")
	}
	if k, err := strconv.Atoi(list); err == nil {
		if k < 1 || k > 26 {
			return protocol.Parties{}, fmt.Errorf("--parties %d: out of range", k)
		}
		return protocol.NewParties(node.PartyNames(k), t)
	}
	return protocol.NewParties(strings.Split(list, ","), t)
}

func dispatch(cmd, run, party string, round int, in, active, solo string, n int, keep, out, partyList string, threshold int) (node.Result, error) {
	need := func(cond bool, what string) error {
		if !cond {
			return fmt.Errorf("%s: missing or invalid %s", cmd, what)
		}
		return nil
	}
	asParty := func(f func(*node.Ctx) (node.Result, error)) (node.Result, error) {
		if err := need(run != "" && party != "", "--run/--party"); err != nil {
			return nil, err
		}
		c, err := node.Load(run, party)
		if err != nil {
			return nil, err
		}
		return f(c)
	}
	asAgg := func(f func(*node.Ctx) (node.Result, error)) (node.Result, error) {
		if err := need(run != "", "--run"); err != nil {
			return nil, err
		}
		c, err := node.Load(run, protocol.Aggregator)
		if err != nil {
			return nil, err
		}
		return f(c)
	}
	roundOK := need(round >= 1, "--round")
	ps, err := parties(partyList, threshold)
	if err != nil {
		return nil, err
	}

	switch cmd {
	case "init":
		if err := need(run != "", "--run"); err != nil {
			return nil, err
		}
		return node.Init(run, he.DefaultConfig, ps)
	case "ceremony":
		if err := need(run != "", "--run"); err != nil {
			return nil, err
		}
		return node.Ceremony(run, he.DefaultConfig, ps)
	case "keygen":
		return asParty(node.Keygen)
	case "pk-share":
		return asParty(node.PKShare)
	case "shamir-deal":
		return asParty(node.ShamirDeal)
	case "shamir-agg":
		return asParty(node.ShamirAgg)
	case "pk-combine":
		return asAgg(node.PKCombine)
	case "encrypt":
		if roundOK != nil {
			return nil, roundOK
		}
		if err := need(in != "", "--in"); err != nil {
			return nil, err
		}
		return asParty(func(c *node.Ctx) (node.Result, error) { return node.Encrypt(c, round, in) })
	case "aggregate":
		if roundOK != nil {
			return nil, roundOK
		}
		return asAgg(func(c *node.Ctx) (node.Result, error) { return node.Aggregate(c, round) })
	case "request":
		if roundOK != nil {
			return nil, roundOK
		}
		if err := need(active != "", "--active"); err != nil {
			return nil, err
		}
		return asAgg(func(c *node.Ctx) (node.Result, error) {
			return node.RequestDecrypt(c, round, strings.Split(active, ","))
		})
	case "dec-share":
		if roundOK != nil {
			return nil, roundOK
		}
		return asParty(func(c *node.Ctx) (node.Result, error) { return node.DecShare(c, round) })
	case "dec-combine":
		if roundOK != nil {
			return nil, roundOK
		}
		return asAgg(func(c *node.Ctx) (node.Result, error) { return node.DecCombine(c, round, solo) })
	case "selftest", "bench", "scaling":
		if err := need(n >= 2, "--n"); err != nil {
			return nil, err
		}
		dir := keep
		if dir == "" {
			tmp, err := os.MkdirTemp("", "horizon-"+cmd+"-")
			if err != nil {
				return nil, err
			}
			defer os.RemoveAll(tmp)
			dir = filepath.Join(tmp, "run")
		}
		if cmd == "scaling" {
			res, err := node.Scaling(filepath.Dir(dir), n, node.ScalingConfigs)
			if err == nil {
				table := node.ScalingTable(res)
				fmt.Fprint(os.Stderr, table)
				if out != "" {
					if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
						return nil, err
					}
					if err := protocol.WriteJSON(out, res, 0o644); err != nil {
						return nil, err
					}
					err = os.WriteFile(strings.TrimSuffix(out, filepath.Ext(out))+".txt", []byte(table), 0o644)
				}
			}
			return res, err
		}
		if cmd == "selftest" {
			res, err := node.Selftest(dir, n, ps)
			if err == nil {
				fmt.Fprint(os.Stderr, node.PrettyChecks(res))
				if res["ok_all"] != true {
					err = fmt.Errorf("selftest: some checks failed")
				}
			}
			return res, err
		}
		res, err := node.Bench(dir, n, ps)
		if err == nil && out != "" {
			if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
				return nil, err
			}
			err = protocol.WriteJSON(out, res, 0o644)
		}
		return res, err
	}
	return nil, fmt.Errorf("unknown command %q\n%s", cmd, usage)
}

func emit(cmd string, res node.Result, err error) {
	if res == nil {
		res = node.Result{}
	}
	res["cmd"] = cmd
	res["ok"] = err == nil
	if err != nil {
		res["error"] = err.Error()
	}
	b, _ := json.Marshal(res)
	fmt.Println(string(b))
	if err != nil {
		os.Exit(1)
	}
}
