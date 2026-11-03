package node

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/multiparty"

	"github.com/4uhn/Meta-AI-Hackathon/crypto/internal/he"
	"github.com/4uhn/Meta-AI-Hackathon/crypto/internal/protocol"
)

// AggregateInfo is bus/round_r/aggregate/aggregate.json.
type AggregateInfo struct {
	RunID        string            `json:"run_id"`
	Round        int               `json:"round"`
	Contributors []string          `json:"contributors"`
	Rejected     map[string]string `json:"rejected"`
	AggSHA256    string            `json:"agg_sha256"`
	NValues      int               `json:"n_values"`
	NCiphertexts int               `json:"n_ciphertexts"`
}

// Encrypt (operator) encrypts a plaintext update under the collective key and writes
// ct.bin plus a signed statement.json into its submission slot for the round.
func Encrypt(c *Ctx, round int, inPath string) (Result, error) {
	t0 := time.Now()
	values, err := ReadF64(inPath)
	if err != nil {
		return nil, err
	}
	if len(values) == 0 {
		return nil, fmt.Errorf("%s is empty", inPath)
	}
	pk, pkHash, err := c.readPK()
	if err != nil {
		return nil, err
	}
	tEnc := time.Now()
	cts, err := he.Encrypt(c.Params, pk, c.Parties.N(), values)
	if err != nil {
		return nil, err
	}
	encMs := since(tEnc)
	b, err := he.MarshalList(he.MagicCiphertexts, cts)
	if err != nil {
		return nil, err
	}
	key, err := c.signingKey()
	if err != nil {
		return nil, err
	}
	stmt, err := protocol.Sign(protocol.SubmissionStatement{
		RunID: c.RunID, Round: round, Operator: c.Run.Actor, CtSHA256: he.SHA256Hex(b),
		NValues: len(values), NCiphertexts: len(cts), ParamsSHA256: c.ParamsSHA256, PKSHA256: pkHash,
	}, key)
	if err != nil {
		return nil, err
	}
	ctf, err := c.Run.SubmissionFile(round, c.Run.Actor, "ct.bin")
	if err != nil {
		return nil, err
	}
	if err := writeNew(ctf, b, 0o644); err != nil {
		return nil, err
	}
	sf, _ := c.Run.SubmissionFile(round, c.Run.Actor, "statement.json")
	if err := protocol.WriteJSON(sf, stmt, 0o644); err != nil {
		return nil, err
	}
	return Result{"party": c.Run.Actor, "round": round, "n_values": len(values), "n_ciphertexts": len(cts),
		"bytes": len(b), "expansion_vs_fp16": float64(len(b)) / float64(2*len(values)),
		"ct_sha256": stmt.Statement.CtSHA256, "encrypt_ms": encMs, "ms": since(t0)}, nil
}

func (c *Ctx) signingKey() ([]byte, error) {
	f, err := c.Run.SecretFile("ed25519.key")
	if err != nil {
		return nil, err
	}
	return protocol.ReadPrivateKey(f)
}

// verifySubmission loads op's round-r submission and checks the signature and every
// binding in the statement. Both the aggregator and the decrypting operators run it.
func (c *Ctx) verifySubmission(round int, op, pkHash string) ([]*rlwe.Ciphertext, protocol.SubmissionStatement, error) {
	var none protocol.SubmissionStatement
	sf, err := c.Run.SubmissionFile(round, op, "statement.json")
	if err != nil {
		return nil, none, err
	}
	var ss protocol.SignedStatement
	if err := protocol.ReadJSON(sf, &ss); err != nil {
		return nil, none, fmt.Errorf("statement: %w", err)
	}
	pf, err := c.Run.SignPubFile(op)
	if err != nil {
		return nil, none, err
	}
	pub, err := protocol.ReadPublicKey(pf)
	if err != nil {
		return nil, none, err
	}
	if err := ss.Verify(pub); err != nil {
		return nil, none, err
	}
	s := ss.Statement
	switch {
	case s.RunID != c.RunID:
		return nil, none, fmt.Errorf("statement run %q, want %q", s.RunID, c.RunID)
	case s.Round != round:
		return nil, none, fmt.Errorf("statement round %d, want %d (replayed submission?)", s.Round, round)
	case s.Operator != op:
		return nil, none, fmt.Errorf("statement operator %q in %s's slot", s.Operator, op)
	case s.ParamsSHA256 != c.ParamsSHA256:
		return nil, none, fmt.Errorf("statement params hash differs")
	case s.PKSHA256 != pkHash:
		return nil, none, fmt.Errorf("statement public key hash differs")
	}
	ctf, _ := c.Run.SubmissionFile(round, op, "ct.bin")
	b, err := os.ReadFile(ctf)
	if err != nil {
		return nil, none, err
	}
	if he.SHA256Hex(b) != s.CtSHA256 {
		return nil, none, fmt.Errorf("ct.bin sha256 does not match the signed statement")
	}
	cts, err := he.ReadCiphertexts(bytes.NewReader(b), c.Params)
	if err != nil {
		return nil, none, err
	}
	slots := he.Slots(c.Params)
	if len(cts) != s.NCiphertexts || len(cts) != (s.NValues+slots-1)/slots {
		return nil, none, fmt.Errorf("%d ciphertexts for %d values (statement says %d)", len(cts), s.NValues, s.NCiphertexts)
	}
	return cts, s, nil
}

// Aggregate (aggregator) verifies every submission present for the round and sums the
// valid ones. Invalid submissions are rejected and reported, not fatal.
func Aggregate(c *Ctx, round int) (Result, error) {
	t0 := time.Now()
	_, pkHash, err := c.readPK()
	if err != nil {
		return nil, err
	}
	info := AggregateInfo{RunID: c.RunID, Round: round, Rejected: map[string]string{}}
	var subs [][]*rlwe.Ciphertext
	for _, op := range c.Parties.Names() {
		ctf, _ := c.Run.SubmissionFile(round, op, "ct.bin")
		if _, err := os.Stat(ctf); os.IsNotExist(err) {
			continue // did not submit this round
		}
		cts, s, err := c.verifySubmission(round, op, pkHash)
		if err == nil && info.NValues != 0 && s.NValues != info.NValues {
			err = fmt.Errorf("%d values, others have %d (layout mismatch)", s.NValues, info.NValues)
		}
		if err != nil {
			info.Rejected[op] = err.Error()
			continue
		}
		info.NValues, info.NCiphertexts = s.NValues, s.NCiphertexts
		info.Contributors = append(info.Contributors, op)
		subs = append(subs, cts)
	}
	if need := c.Parties.MinContributors(); len(subs) < need {
		return nil, fmt.Errorf("round %d: %d valid submissions (rejected: %v), need ≥ %d", round, len(subs), info.Rejected, need)
	}
	tSum := time.Now()
	agg, err := he.Aggregate(c.Params, subs)
	if err != nil {
		return nil, err
	}
	sumMs := since(tSum)
	b, err := he.MarshalList(he.MagicCiphertexts, agg)
	if err != nil {
		return nil, err
	}
	info.AggSHA256 = he.SHA256Hex(b)
	af, err := c.Run.RoundDir(round, "aggregate", "ct_agg.bin")
	if err != nil {
		return nil, err
	}
	if err := writeNew(af, b, 0o644); err != nil {
		return nil, err
	}
	jf, _ := c.Run.RoundDir(round, "aggregate", "aggregate.json")
	if err := protocol.WriteJSON(jf, info, 0o644); err != nil {
		return nil, err
	}
	return Result{"round": round, "contributors": info.Contributors, "rejected": info.Rejected,
		"agg_sha256": info.AggSHA256, "bytes": len(b), "sum_ms": sumMs, "ms": since(t0)}, nil
}

func (c *Ctx) readAggregateInfo(round int) (AggregateInfo, error) {
	var info AggregateInfo
	f, err := c.Run.RoundDir(round, "aggregate", "aggregate.json")
	if err != nil {
		return info, err
	}
	return info, protocol.ReadJSON(f, &info)
}

// RequestDecrypt (aggregator) fixes the active set for the round's one decryption.
func RequestDecrypt(c *Ctx, round int, active []string) (Result, error) {
	info, err := c.readAggregateInfo(round)
	if err != nil {
		return nil, err
	}
	q := protocol.DecryptRequest{RunID: c.RunID, Round: round, Active: active, Contributors: info.Contributors, AggSHA256: info.AggSHA256}
	if err := q.Check(c.Parties); err != nil {
		return nil, err
	}
	f, err := c.Run.RoundDir(round, "decrypt", "request.json")
	if err != nil {
		return nil, err
	}
	if _, err := os.Stat(f); err == nil {
		return nil, fmt.Errorf("round %d already has a decryption request (one-shot)", round)
	}
	if err := os.MkdirAll(filepath.Dir(f), 0o755); err != nil {
		return nil, err
	}
	if err := protocol.WriteJSON(f, q, 0o644); err != nil {
		return nil, err
	}
	return Result{"round": round, "active": active, "contributors": q.Contributors, "agg_sha256": q.AggSHA256}, nil
}

// DecShare (operator) checks the request, recomputes the aggregate from the signed
// submissions itself, records the decryption in its one-shot log, and only then releases
// its flooded decryption share.
func DecShare(c *Ctx, round int) (Result, error) {
	t0 := time.Now()
	me := c.Run.Actor
	rf, err := c.Run.RoundDir(round, "decrypt", "request.json")
	if err != nil {
		return nil, err
	}
	var q protocol.DecryptRequest
	if err := protocol.ReadJSON(rf, &q); err != nil {
		return nil, fmt.Errorf("request: %w", err)
	}
	if err := q.Check(c.Parties); err != nil {
		return nil, fmt.Errorf("refuse: %w", err)
	}
	if q.RunID != c.RunID || q.Round != round {
		return nil, fmt.Errorf("refuse: request is for run %s round %d", q.RunID, q.Round)
	}
	if !slices.Contains(q.Active, me) {
		return nil, fmt.Errorf("refuse: %s is not in the active set %v", me, q.Active)
	}

	// Rebuild the aggregate from the signed submissions; never trust ct_agg.bin.
	tVerify := time.Now()
	_, pkHash, err := c.readPK()
	if err != nil {
		return nil, err
	}
	var subs [][]*rlwe.Ciphertext
	nValues := 0
	for _, op := range q.Contributors {
		cts, s, err := c.verifySubmission(round, op, pkHash)
		if err != nil {
			return nil, fmt.Errorf("refuse: submission of %s: %w", op, err)
		}
		if nValues != 0 && s.NValues != nValues {
			return nil, fmt.Errorf("refuse: contributors disagree on payload size")
		}
		nValues = s.NValues
		subs = append(subs, cts)
	}
	agg, err := he.Aggregate(c.Params, subs)
	if err != nil {
		return nil, err
	}
	b, err := he.MarshalList(he.MagicCiphertexts, agg)
	if err != nil {
		return nil, err
	}
	if h := he.SHA256Hex(b); h != q.AggSHA256 {
		return nil, fmt.Errorf("refuse: recomputed aggregate %s… ≠ requested %s…", h[:12], q.AggSHA256[:12])
	}
	verifyMs := since(tVerify)

	// Log before computing, so no crash or retry can release a second share.
	lf, err := c.Run.StateFile("decrypt_log.jsonl")
	if err != nil {
		return nil, err
	}
	if err := protocol.OpenDecryptLog(lf).Authorise(q); err != nil {
		return nil, fmt.Errorf("refuse: %w", err)
	}

	tf, err := c.Run.SecretFile("tsk.bin")
	if err != nil {
		return nil, err
	}
	var tsk multiparty.ShamirSecretShare
	if err := he.ReadObject(tf, &tsk); err != nil {
		return nil, err
	}
	own, _ := c.Parties.Point(me)
	var active []he.Point
	for _, a := range q.Active {
		p, _ := c.Parties.Point(a)
		active = append(active, p)
	}
	tShare := time.Now()
	shares, err := he.DecryptionShare(c.Params, tsk, own, active, c.Parties.Threshold, c.Parties.N(), agg)
	if err != nil {
		return nil, err
	}
	shareMs := since(tShare)
	sb, err := he.MarshalList(he.MagicShares, shares)
	if err != nil {
		return nil, err
	}
	sf, _ := c.Run.RoundDir(round, "decrypt", "decshare_"+me+".bin")
	if err := writeNew(sf, sb, 0o644); err != nil {
		return nil, err
	}
	rcpt := protocol.ShareReceipt{RunID: c.RunID, Round: round, Party: me, Active: q.Active, AggSHA256: q.AggSHA256, ShareSHA256: he.SHA256Hex(sb)}
	jf, _ := c.Run.RoundDir(round, "decrypt", "decshare_"+me+".json")
	if err := protocol.WriteJSON(jf, rcpt, 0o644); err != nil {
		return nil, err
	}
	return Result{"party": me, "round": round, "active": q.Active, "contributors": q.Contributors,
		"verify_ms": verifyMs, "share_ms": shareMs, "bytes": len(sb), "ms": since(t0)}, nil
}

// DecCombine (aggregator) combines the active parties' shares into agg.f64. With solo
// set (one operator, or a comma-separated list of fewer than t) it combines only those
// shares, to show they give noise, into solo_<op>[+<op>…].f64 and reports the relative
// error against agg.f64 if it exists.
func DecCombine(c *Ctx, round int, solo string) (Result, error) {
	t0 := time.Now()
	rf, err := c.Run.RoundDir(round, "decrypt", "request.json")
	if err != nil {
		return nil, err
	}
	var q protocol.DecryptRequest
	if err := protocol.ReadJSON(rf, &q); err != nil {
		return nil, err
	}
	info, err := c.readAggregateInfo(round)
	if err != nil {
		return nil, err
	}
	af, _ := c.Run.RoundDir(round, "aggregate", "ct_agg.bin")
	ab, err := os.ReadFile(af)
	if err != nil {
		return nil, err
	}
	if he.SHA256Hex(ab) != q.AggSHA256 {
		return nil, fmt.Errorf("ct_agg.bin does not match the request hash")
	}
	agg, err := he.ReadCiphertexts(bytes.NewReader(ab), c.Params)
	if err != nil {
		return nil, err
	}
	use := q.Active
	if solo != "" {
		use = strings.Split(solo, ",")
		if len(use) >= c.Parties.Threshold {
			return nil, fmt.Errorf("--solo takes fewer than t = %d operators, got %v", c.Parties.Threshold, use)
		}
		for i, op := range use {
			if !slices.Contains(q.Active, op) {
				return nil, fmt.Errorf("%s did not take part in round %d's decryption", op, round)
			}
			if slices.Contains(use[:i], op) {
				return nil, fmt.Errorf("--solo lists %s twice", op)
			}
		}
	}
	var shares [][]multiparty.KeySwitchShare
	for _, op := range use {
		jf, _ := c.Run.RoundDir(round, "decrypt", "decshare_"+op+".json")
		var r protocol.ShareReceipt
		if err := protocol.ReadJSON(jf, &r); err != nil {
			return nil, fmt.Errorf("receipt of %s: %w", op, err)
		}
		if r.Party != op || r.RunID != c.RunID || r.Round != round || r.AggSHA256 != q.AggSHA256 || !slices.Equal(r.Active, q.Active) {
			return nil, fmt.Errorf("receipt of %s does not match the request", op)
		}
		sf, _ := c.Run.RoundDir(round, "decrypt", "decshare_"+op+".bin")
		sb, err := os.ReadFile(sf)
		if err != nil {
			return nil, err
		}
		if he.SHA256Hex(sb) != r.ShareSHA256 {
			return nil, fmt.Errorf("share of %s does not match its receipt", op)
		}
		s, err := he.ReadShares(bytes.NewReader(sb), c.Params)
		if err != nil {
			return nil, err
		}
		shares = append(shares, s)
	}
	tComb := time.Now()
	v, err := he.CombineDecryption(c.Params, agg, shares, info.NValues)
	if err != nil {
		return nil, err
	}
	combMs := since(tComb)
	res := Result{"round": round, "used": use, "n_values": len(v), "combine_ms": combMs}
	if solo == "" {
		of, _ := c.Run.RoundDir(round, "decrypt", "agg.f64")
		if err := WriteF64(of, v, 0o644); err != nil {
			return nil, err
		}
		res["weight_slot"] = v[len(v)-1]
	} else {
		of, _ := c.Run.RoundDir(round, "decrypt", "solo_"+strings.Join(use, "+")+".f64")
		if err := WriteF64(of, v, 0o644); err != nil {
			return nil, err
		}
		gf, _ := c.Run.RoundDir(round, "decrypt", "agg.f64")
		if good, err := ReadF64(gf); err == nil && len(good) == len(v) {
			res["rel_err_vs_true"] = relErr(v, good)
		}
	}
	res["ms"] = since(t0)
	return res, nil
}

func relErr(got, want []float64) float64 {
	var num, den float64
	for i := range want {
		d := got[i] - want[i]
		num += d * d
		den += want[i] * want[i]
	}
	return math.Sqrt(num / den)
}
