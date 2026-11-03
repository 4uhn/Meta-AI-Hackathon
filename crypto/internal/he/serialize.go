package he

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"os"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/multiparty"
)

// File format for lists of Lattigo objects (ct.bin, ct_agg.bin, decshare_*.bin):
//
//	magic [4]byte | count uint32 | count × (length uint64 | MarshalBinary bytes)
//
// little endian. Lattigo's marshalling is canonical (fixed layout, no maps), so equal
// objects give equal bytes. Operators rely on this when they recompute the aggregate
// and compare its hash with the one they are asked to decrypt.
var (
	MagicCiphertexts = [4]byte{'H', 'Z', 'C', '1'}
	MagicShares      = [4]byte{'H', 'Z', 'S', '1'}
)

const (
	maxListCount  = 1 << 20
	maxObjectSize = 1 << 30
)

// WriteList writes objects in the list format above.
func WriteList[T encoding.BinaryMarshaler](w io.Writer, magic [4]byte, items []T) error {
	bw := bufio.NewWriter(w)
	if _, err := bw.Write(magic[:]); err != nil {
		return err
	}
	if err := binary.Write(bw, binary.LittleEndian, uint32(len(items))); err != nil {
		return err
	}
	for i, it := range items {
		b, err := it.MarshalBinary()
		if err != nil {
			return fmt.Errorf("marshal item %d: %w", i, err)
		}
		if err := binary.Write(bw, binary.LittleEndian, uint64(len(b))); err != nil {
			return err
		}
		if _, err := bw.Write(b); err != nil {
			return err
		}
	}
	return bw.Flush()
}

// readList reads the list format, calling decode on each object's bytes.
func readList(r io.Reader, magic [4]byte, decode func([]byte) error) (int, error) {
	br := bufio.NewReader(r)
	var m [4]byte
	if _, err := io.ReadFull(br, m[:]); err != nil {
		return 0, fmt.Errorf("read magic: %w", err)
	}
	if m != magic {
		return 0, fmt.Errorf("bad magic %q, want %q", m[:], magic[:])
	}
	var n uint32
	if err := binary.Read(br, binary.LittleEndian, &n); err != nil {
		return 0, err
	}
	if n > maxListCount {
		return 0, fmt.Errorf("list count %d too large", n)
	}
	for i := uint32(0); i < n; i++ {
		var l uint64
		if err := binary.Read(br, binary.LittleEndian, &l); err != nil {
			return 0, fmt.Errorf("item %d length: %w", i, err)
		}
		if l > maxObjectSize {
			return 0, fmt.Errorf("item %d size %d too large", i, l)
		}
		b := make([]byte, l)
		if _, err := io.ReadFull(br, b); err != nil {
			return 0, fmt.Errorf("item %d: %w", i, err)
		}
		if err := decode(b); err != nil {
			return 0, fmt.Errorf("decode item %d: %w", i, err)
		}
	}
	if _, err := br.ReadByte(); err != io.EOF {
		return 0, fmt.Errorf("trailing bytes after %d items", n)
	}
	return int(n), nil
}

// ReadCiphertexts reads a ciphertext list and checks every ciphertext matches params.
func ReadCiphertexts(r io.Reader, params rlwe.ParameterProvider) ([]*rlwe.Ciphertext, error) {
	p := params.GetRLWEParameters()
	var out []*rlwe.Ciphertext
	_, err := readList(r, MagicCiphertexts, func(b []byte) error {
		ct := new(rlwe.Ciphertext)
		if err := ct.UnmarshalBinary(b); err != nil {
			return err
		}
		if ct.Degree() != 1 || ct.Level() != p.MaxLevel() || ct.Value[0].N() != p.N() {
			return fmt.Errorf("ciphertext shape (degree %d, level %d, N %d) does not match parameters", ct.Degree(), ct.Level(), ct.Value[0].N())
		}
		out = append(out, ct)
		return nil
	})
	return out, err
}

// ReadShares reads a decryption-share list.
func ReadShares(r io.Reader, params rlwe.ParameterProvider) ([]multiparty.KeySwitchShare, error) {
	p := params.GetRLWEParameters()
	var out []multiparty.KeySwitchShare
	_, err := readList(r, MagicShares, func(b []byte) error {
		var s multiparty.KeySwitchShare
		if err := s.UnmarshalBinary(b); err != nil {
			return err
		}
		if s.Value.N() != p.N() || s.Level() != p.MaxLevel() {
			return fmt.Errorf("share shape (N %d, level %d) does not match parameters", s.Value.N(), s.Level())
		}
		out = append(out, s)
		return nil
	})
	return out, err
}

// MarshalList returns the list encoding in memory.
func MarshalList[T encoding.BinaryMarshaler](magic [4]byte, items []T) ([]byte, error) {
	var buf bytes.Buffer
	if err := WriteList(&buf, magic, items); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// SHA256Hex returns the lowercase hex sha256 of b.
func SHA256Hex(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

// ReadObject reads one Lattigo object from a file.
func ReadObject(path string, obj encoding.BinaryUnmarshaler) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return obj.UnmarshalBinary(b)
}
