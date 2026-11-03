package protocol

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
)

// SubmissionStatement is what an operator signs when it submits a ciphertext. It binds
// the ciphertext hash to the run, round, operator and the exact crypto setup.
// Field order is fixed by the struct, so json.Marshal gives canonical bytes.
type SubmissionStatement struct {
	RunID        string `json:"run_id"`
	Round        int    `json:"round"`
	Operator     string `json:"operator"`
	CtSHA256     string `json:"ct_sha256"`
	NValues      int    `json:"n_values"`
	NCiphertexts int    `json:"n_ciphertexts"`
	ParamsSHA256 string `json:"params_sha256"`
	PKSHA256     string `json:"pk_sha256"`
}

// SignedStatement is statement.json next to ct.bin.
type SignedStatement struct {
	Statement SubmissionStatement `json:"statement"`
	Signature string              `json:"signature"` // base64 Ed25519
}

// Domain separation: a signature over a submission can never be replayed as anything else.
const statementDomain = "horizon/submission/v1\n"

func (s SubmissionStatement) message() ([]byte, error) {
	b, err := json.Marshal(s)
	if err != nil {
		return nil, err
	}
	return append([]byte(statementDomain), b...), nil
}

// Sign signs the statement with the operator's Ed25519 key.
func Sign(s SubmissionStatement, key ed25519.PrivateKey) (SignedStatement, error) {
	m, err := s.message()
	if err != nil {
		return SignedStatement{}, err
	}
	return SignedStatement{Statement: s, Signature: base64.StdEncoding.EncodeToString(ed25519.Sign(key, m))}, nil
}

// Verify checks the signature against the operator's published public key.
func (ss SignedStatement) Verify(pub ed25519.PublicKey) error {
	sig, err := base64.StdEncoding.DecodeString(ss.Signature)
	if err != nil {
		return fmt.Errorf("signature encoding: %w", err)
	}
	m, err := ss.Statement.message()
	if err != nil {
		return err
	}
	if !ed25519.Verify(pub, m, sig) {
		return fmt.Errorf("bad signature on %s's round-%d submission", ss.Statement.Operator, ss.Statement.Round)
	}
	return nil
}

// GenSigningKey draws an Ed25519 key pair from crypto/rand.
func GenSigningKey() (ed25519.PublicKey, ed25519.PrivateKey, error) {
	return ed25519.GenerateKey(rand.Reader)
}

// ReadPublicKey and ReadPrivateKey load raw Ed25519 keys.
func ReadPublicKey(path string) (ed25519.PublicKey, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b) != ed25519.PublicKeySize {
		return nil, fmt.Errorf("%s: %d bytes, want %d", path, len(b), ed25519.PublicKeySize)
	}
	return ed25519.PublicKey(b), nil
}

func ReadPrivateKey(path string) (ed25519.PrivateKey, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b) != ed25519.PrivateKeySize {
		return nil, fmt.Errorf("%s: %d bytes, want %d", path, len(b), ed25519.PrivateKeySize)
	}
	return ed25519.PrivateKey(b), nil
}

// WriteJSON writes v as indented JSON.
func WriteJSON(path string, v any, perm os.FileMode) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), perm)
}

// ReadJSON reads JSON into v and rejects unknown fields.
func ReadJSON(path string, v any) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	d := json.NewDecoder(f)
	d.DisallowUnknownFields()
	return d.Decode(v)
}
