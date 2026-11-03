// Package node implements every protocol step as a function acting as exactly one actor
// on a run directory. The CLI and the selftest both call these, so the selftest
// exercises the same file-based flow (and the same isolation checks) as a real run.
package node

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/multiparty"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"

	"github.com/4uhn/Meta-AI-Hackathon/crypto/internal/he"
	"github.com/4uhn/Meta-AI-Hackathon/crypto/internal/protocol"
)

// Result is the JSON summary every command prints on stdout.
type Result map[string]any

// Ctx is the public context every actor loads: verified parameters and the party list.
type Ctx struct {
	Run          *protocol.Run
	RunID        string
	Params       ckks.Parameters
	ParamsSHA256 string
	Parties      protocol.Parties
}

func since(t time.Time) float64 { return float64(time.Since(t).Microseconds()) / 1000 }

// Init (aggregator) creates public/params.json, public/crs_seed and public/parties.json.
// The flood width in params.json is sized for the number of parties.
func Init(root string, cfg he.Config, parties protocol.Parties) (Result, error) {
	t0 := time.Now()
	if err := parties.Validate(); err != nil {
		return nil, err
	}
	run, err := protocol.NewRun(root, protocol.Aggregator)
	if err != nil {
		return nil, err
	}
	pf, _ := run.ParamsFile()
	if _, err := os.Stat(pf); err == nil {
		return nil, fmt.Errorf("%s exists: refusing to overwrite a run", pf)
	}
	params, err := he.NewParameters(cfg)
	if err != nil {
		return nil, err
	}
	w, err := he.FloodLogWidth(params, parties.N())
	if err != nil {
		return nil, err
	}
	h, err := paramsHash(params)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Dir(pf), 0o755); err != nil {
		return nil, err
	}
	if err := protocol.WriteJSON(pf, protocol.ParamsFile{
		LogN: cfg.LogN, LogQ: cfg.LogQ, LogScale: cfg.LogScale, RingType: params.RingType().String(),
		FloodLogW: w, StatSecurity: he.StatSecurity, ParamsSHA256: h,
	}, 0o644); err != nil {
		return nil, err
	}
	seed, err := he.NewCRSSeed()
	if err != nil {
		return nil, err
	}
	cf, _ := run.CRSFile()
	if err := os.WriteFile(cf, seed, 0o644); err != nil {
		return nil, err
	}
	pp, _ := run.PartiesFile()
	if err := protocol.WriteJSON(pp, parties, 0o644); err != nil {
		return nil, err
	}
	return Result{"run_id": filepath.Base(run.Root), "params_sha256": h, "flood_log_width": w,
		"slots": he.Slots(params), "parties": parties.Names(), "ms": since(t0)}, nil
}

func paramsHash(params ckks.Parameters) (string, error) {
	b, err := params.MarshalBinary()
	if err != nil {
		return "", err
	}
	return he.SHA256Hex(b), nil
}

// Load reads and verifies the public context as the given actor.
func Load(root, actor string) (*Ctx, error) {
	run, err := protocol.NewRun(root, actor)
	if err != nil {
		return nil, err
	}
	// The party list comes first: the flood width depends on how many parties there are.
	pp, _ := run.PartiesFile()
	var parties protocol.Parties
	if err := protocol.ReadJSON(pp, &parties); err != nil {
		return nil, fmt.Errorf("parties: %w", err)
	}
	if err := parties.Validate(); err != nil {
		return nil, err
	}
	pf, _ := run.ParamsFile()
	var p protocol.ParamsFile
	if err := protocol.ReadJSON(pf, &p); err != nil {
		return nil, fmt.Errorf("params: %w", err)
	}
	params, err := he.NewParameters(he.Config{LogN: p.LogN, LogQ: p.LogQ, LogScale: p.LogScale})
	if err != nil {
		return nil, err
	}
	h, err := paramsHash(params)
	if err != nil {
		return nil, err
	}
	if h != p.ParamsSHA256 {
		return nil, fmt.Errorf("params hash mismatch: file says %s, this build derives %s", p.ParamsSHA256, h)
	}
	if w, _ := he.FloodLogWidth(params, parties.N()); w != p.FloodLogW {
		return nil, fmt.Errorf("flood width mismatch: file says %d, this build derives %d for %d parties", p.FloodLogW, w, parties.N())
	}
	if actor != protocol.Aggregator {
		if _, err := parties.Point(actor); err != nil {
			return nil, err
		}
	}
	return &Ctx{Run: run, RunID: filepath.Base(run.Root), Params: params, ParamsSHA256: h, Parties: parties}, nil
}

func (c *Ctx) crs() ([]byte, error) {
	cf, _ := c.Run.CRSFile()
	b, err := os.ReadFile(cf)
	if err == nil && len(b) != 32 {
		err = fmt.Errorf("crs_seed has %d bytes, want 32", len(b))
	}
	return b, err
}

// writeNew creates a file that must not exist yet (no silent overwrite of keys or
// submissions), with the parent directory created 0700.
func writeNew(path string, data []byte, perm os.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, perm)
	if err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// Keygen (operator) generates the operator's RLWE secret key and Ed25519 signing key.
func Keygen(c *Ctx) (Result, error) {
	t0 := time.Now()
	sk := he.GenSecretKey(c.Params)
	b, err := sk.MarshalBinary()
	if err != nil {
		return nil, err
	}
	skf, err := c.Run.SecretFile("sk.bin")
	if err != nil {
		return nil, err
	}
	if err := writeNew(skf, b, 0o600); err != nil {
		return nil, err
	}
	pub, priv, err := protocol.GenSigningKey()
	if err != nil {
		return nil, err
	}
	kf, _ := c.Run.SecretFile("ed25519.key")
	if err := writeNew(kf, priv, 0o600); err != nil {
		return nil, err
	}
	pf, _ := c.Run.SignPubFile(c.Run.Actor)
	if err := writeNew(pf, pub, 0o644); err != nil {
		return nil, err
	}
	return Result{"party": c.Run.Actor, "sk_bytes": len(b), "sign_pub": he.SHA256Hex(pub)[:16], "ms": since(t0)}, nil
}

func (c *Ctx) readSK() (*rlwe.SecretKey, error) {
	f, err := c.Run.SecretFile("sk.bin")
	if err != nil {
		return nil, err
	}
	sk := new(rlwe.SecretKey)
	return sk, he.ReadObject(f, sk)
}

// PKShare (operator) publishes -a·sk_i + e_i for the collective public key.
func PKShare(c *Ctx) (Result, error) {
	t0 := time.Now()
	sk, err := c.readSK()
	if err != nil {
		return nil, err
	}
	seed, err := c.crs()
	if err != nil {
		return nil, err
	}
	share, err := he.PublicKeyShare(c.Params, sk, seed)
	if err != nil {
		return nil, err
	}
	b, err := share.MarshalBinary()
	if err != nil {
		return nil, err
	}
	f, err := c.Run.PKShareFile(c.Run.Actor)
	if err != nil {
		return nil, err
	}
	if err := writeNew(f, b, 0o644); err != nil {
		return nil, err
	}
	return Result{"party": c.Run.Actor, "bytes": len(b), "ms": since(t0)}, nil
}

// ShamirDeal (operator) splits sk_i into one Shamir share per party and sends each to
// its recipient's private inbox. Sending into another party's inbox is write-only: the
// file is created exclusively and never read back by the sender.
func ShamirDeal(c *Ctx) (Result, error) {
	t0 := time.Now()
	sk, err := c.readSK()
	if err != nil {
		return nil, err
	}
	shares, err := he.DealShamirShares(c.Params, sk, c.Parties.Threshold, c.Parties.Points())
	if err != nil {
		return nil, err
	}
	for _, p := range c.Parties.Parties {
		b, err := shares[he.Point(p.Point)].MarshalBinary()
		if err != nil {
			return nil, err
		}
		if err := c.send(p.Name, b); err != nil {
			return nil, err
		}
	}
	return Result{"party": c.Run.Actor, "sent_to": c.Parties.Names(), "ms": since(t0)}, nil
}

// send writes into bus/private/<to>/from_<me>.bin. It bypasses Run.Path's ownership check
// on purpose (that check guards reads), and is the only place that does.
func (c *Ctx) send(to string, data []byte) error {
	if !protocol.ValidParty(to) {
		return fmt.Errorf("invalid recipient %q", to)
	}
	p := filepath.Join(c.Run.Root, "bus", "private", to, "from_"+c.Run.Actor+".bin")
	return writeNew(p, data, 0o600)
}

// ShamirAgg (operator) sums the shares in its own inbox into tsk = S(α_me). It then
// deletes the inbox and the original sk_i: from here on, only tsk exists. The PK share
// must already be published, since it needs sk_i.
func ShamirAgg(c *Ctx) (Result, error) {
	t0 := time.Now()
	me := c.Run.Actor
	pkf, _ := c.Run.PKShareFile(me)
	if _, err := os.Stat(pkf); err != nil {
		return nil, fmt.Errorf("publish the PK share before aggregating Shamir shares (sk is erased after): %w", err)
	}
	var received []multiparty.ShamirSecretShare
	var inbox []string
	for _, from := range c.Parties.Names() {
		f, err := c.Run.InboxFile(me, from)
		if err != nil {
			return nil, err
		}
		var s multiparty.ShamirSecretShare
		if err := he.ReadObject(f, &s); err != nil {
			return nil, fmt.Errorf("share from %s: %w", from, err)
		}
		received = append(received, s)
		inbox = append(inbox, f)
	}
	tsk, err := he.AggregateShamirShares(c.Params, received)
	if err != nil {
		return nil, err
	}
	b, err := tsk.MarshalBinary()
	if err != nil {
		return nil, err
	}
	tf, _ := c.Run.SecretFile("tsk.bin")
	if err := writeNew(tf, b, 0o600); err != nil {
		return nil, err
	}
	skf, _ := c.Run.SecretFile("sk.bin")
	for _, f := range append(inbox, skf) {
		if err := os.Remove(f); err != nil {
			return nil, fmt.Errorf("erase %s: %w", f, err)
		}
	}
	return Result{"party": me, "tsk_bytes": len(b), "erased": []string{"sk.bin", "inbox"}, "ms": since(t0)}, nil
}

// PKCombine (aggregator) sums the public key shares into public/pk.bin.
func PKCombine(c *Ctx) (Result, error) {
	t0 := time.Now()
	seed, err := c.crs()
	if err != nil {
		return nil, err
	}
	var shares []multiparty.PublicKeyGenShare
	for _, p := range c.Parties.Names() {
		f, err := c.Run.PKShareFile(p)
		if err != nil {
			return nil, err
		}
		var s multiparty.PublicKeyGenShare
		if err := he.ReadObject(f, &s); err != nil {
			return nil, fmt.Errorf("pk share of %s: %w", p, err)
		}
		shares = append(shares, s)
	}
	pk, err := he.CombinePublicKey(c.Params, seed, shares)
	if err != nil {
		return nil, err
	}
	b, err := pk.MarshalBinary()
	if err != nil {
		return nil, err
	}
	f, _ := c.Run.PKFile()
	if err := writeNew(f, b, 0o644); err != nil {
		return nil, err
	}
	return Result{"pk_sha256": he.SHA256Hex(b), "bytes": len(b), "ms": since(t0)}, nil
}

func (c *Ctx) readPK() (*rlwe.PublicKey, string, error) {
	f, _ := c.Run.PKFile()
	b, err := os.ReadFile(f)
	if err != nil {
		return nil, "", err
	}
	pk := new(rlwe.PublicKey)
	return pk, he.SHA256Hex(b), pk.UnmarshalBinary(b)
}

// ReadF64 reads raw little-endian float64 (update.f64, agg.f64).
func ReadF64(path string) ([]float64, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b)%8 != 0 {
		return nil, fmt.Errorf("%s: %d bytes is not a whole number of float64", path, len(b))
	}
	v := make([]float64, len(b)/8)
	for i := range v {
		v[i] = math.Float64frombits(binary.LittleEndian.Uint64(b[8*i:]))
	}
	return v, nil
}

// WriteF64 writes raw little-endian float64.
func WriteF64(path string, v []float64, perm os.FileMode) error {
	var buf bytes.Buffer
	buf.Grow(8 * len(v))
	var w [8]byte
	for _, x := range v {
		binary.LittleEndian.PutUint64(w[:], math.Float64bits(x))
		buf.Write(w[:])
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	return os.WriteFile(path, buf.Bytes(), perm)
}
