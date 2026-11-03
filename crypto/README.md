# horizon-he: t-of-N threshold CKKS for federated aggregation

`horizon-he` is a Go command-line tool built on [Lattigo v6](https://github.com/tuneinsight/lattigo)
(multiparty CKKS). It lets N organisations add up their private vectors, such as model updates, so
that:

- the server sums ciphertexts and holds no key material;
- any t organisations together can decrypt the **sum**, and fewer than t learn nothing;
- no organisation, and not the server, can decrypt another organisation's individual vector.

There is no trusted dealer. Each organisation creates its own secret key and deals Shamir shares of
it to the others. The public key combines their public shares, and the whole key never exists in one
place.

Every subcommand acts as exactly one party (the aggregator or one operator) on a run directory and
prints a one-line JSON summary. Python drives it through `horizon/he_bridge.py`; nothing in Python
touches key material.

## Build

```bash
make he          # needs Go ≥ 1.24 → crypto/bin/horizon-he (this machine) + horizon-he-linux-amd64 (containers)
make test-go     # go vet + unit tests (make test-go-long adds the 10M-value precision test, race detector)
```

Without Go, download `horizon-he-<os>-<arch>` and `SHA256SUMS` from the repository's Releases
page (built by `make release`). Check the file with `shasum -a 256 -c SHA256SUMS --ignore-missing`,
then save it as `crypto/bin/horizon-he`, or point `HORIZON_HE` at it. On macOS, clear the download
flag with `xattr -d com.apple.quarantine <file>`. Modal runs also need the Linux build at
`crypto/bin/horizon-he-linux-amd64`.

## Commands

| Phase | Command | Who | What |
|---|---|---|---|
| Setup | `init --run DIR [--parties N --threshold T]` | aggregator | public parameters, CRS seed, party list |
| | `keygen`, `pk-share`, `shamir-deal`, `shamir-agg` `--run DIR --party OP` | each operator | own secret key, public share, Shamir shares to the others' inboxes, threshold key share (then the secret key is erased) |
| | `pk-combine --run DIR` | aggregator | collective public key |
| Round r | `encrypt --run DIR --party OP --round R --in update.f64` | operator | encrypt and sign its update |
| | `aggregate --run DIR --round R` | aggregator | verify signatures, sum the ciphertexts |
| | `request --run DIR --round R --active op_a,op_c` | aggregator | name the t parties that decrypt this round |
| | `dec-share --run DIR --party OP --round R` | operator | recompute the sum itself, check the rules, release one decryption share |
| | `dec-combine --run DIR --round R` | aggregator | combine t shares into the decrypted sum |
| Checks | `check --parties N --threshold T [--unverified]` | anyone | is this t-of-N safe? rules, parameters, full selftest |
| | `selftest`, `bench`, `scaling` | anyone | protocol and attack checks, timings, the scaling table |

`ceremony --run DIR` runs the whole setup in one process, for tests and demos. `--parties` takes a
count (`5` means op_a … op_e) or a list.

## Parameters

| | Value |
|---|---|
| Ring | conjugate-invariant, degree 2^14: 16,384 real values per ciphertext |
| Modulus | Q = two 60-bit primes (log2 Q = 120), no special modulus; additions only |
| Security | ≥ 128 bits: log2 QP = 120, under the 438-bit limit for degree 2^14 (HE Security Standard) |
| Scale | Δ = 2^114; headroom Q/2Δ = 32 |
| Value range | each value in [−4, 4] for N = 3, [−2, 2] for N = 4 to 7 (LoRA updates are far smaller) |
| Noise flooding | each decryption share adds exact uniform noise 2^(λ+1) times the proven bound on the ciphertext noise, λ = 64 (statistical distance ≤ 2^−64 per coefficient) |
| Precision | max abs error 8.0e-7 at 2-of-3 on 4,685,825 values (below) |

## Protocol rules

Each operator enforces these itself before releasing a decryption share:

1. **At least t + 1 contributors** (and at least 3). Members learn the decrypted sum, so t − 1
   colluders could subtract their own updates from it. With t + 1 contributors they are left with a
   sum of at least two other updates, never one.
2. **2t > N.** Any two decrypting groups share a member. Because each member releases one share per
   round, a round cannot be decrypted twice and the two results differenced.
3. **t ≤ N − 1.** Rule 1 needs t + 1 contributors, which t = N could never reach.
4. **One share per round, ever.** Repeating a partial decryption leaks key information: Lattigo's
   SECURITY.md warns that retrying any multiparty protocol is insecure, even with a different
   t-subset. Each operator logs what it has shared and refuses a second request for the same round.
   If a decryption fails, the round is skipped.
5. **Verify before sharing.** The operator recomputes the aggregate from the signed (Ed25519)
   submissions and checks it byte for byte, so a server cannot pass off one organisation's ciphertext
   as "the aggregate".

Secret material lives only in `parties/<op>/secret/`. Every file access goes through one path guard,
which refuses another party's directory.

## Choosing t-of-N

```bash
make check-tofn PARTIES=5 THRESHOLD=4      # = crypto/bin/horizon-he check --parties 5 --threshold 4
```

`check` validates the rules, prints the parameters for that size, then runs the full selftest at the
Muse Glimmer payload (4,685,825 values; 3-of-4 takes under a minute on an M2 Pro). The selftest checks:

- every t-subset decrypts;
- t − 1 shares give noise;
- a skipped round is refused;
- each refusal holds: a 2-contributor sum, an individual ciphertext labelled as the aggregate, a
  tampered or replayed submission, a second decryption of a round.

It exits non-zero with the reason when a shape is unsafe or unsupported.

Measured at 4,685,825 values (`results/crypto/scaling.json`, M2 Pro, 10 cores, crypto only):

| t-of-N | Contributors needed | Offline at decryption | Flood | Max abs error | Selftest | Encrypt ms | Bytes per submission |
|---|---|---|---|---|---|---|---|
| 2-of-3 | 3 | 1 | 2^85 | 8.0e-7 | 15/15 | 359 | 150.6 MB |
| 3-of-4 | 4 | 1 | 2^86 | 1.8e-6 | 21/21 | 337 | 150.6 MB |
| 3-of-5 | 4 | 2 | 2^87 | 4.0e-6 | 39/39 | 391 | 150.6 MB |
| 4-of-5 | 5 | 1 | 2^87 | 4.3e-6 | 24/24 | 363 | 150.6 MB |

**t = N − 1 (recommended).** Every member submits every round, so each round has exactly one sum.
The guarantee then holds even if the server recruits up to t − 1 members who ignore rule 4.

**Smaller t (e.g. 3-of-5).** Up to N − t − 1 members may skip a round, and N − t may be offline at
decryption. But several different sums of one round can now exist. If c members collude with the
server and release a second share, two of them could be decrypted and differenced, unless
2t > N + c. Use a smaller t only when every member is trusted to follow rule 4.

## Going beyond five organisations

N ≤ 5 is what we measured. This build accepts N = 6 or 7 if you verify it yourself:

```bash
crypto/bin/horizon-he check --parties 7 --threshold 6 --unverified        # selftest at the Glimmer payload
crypto/bin/horizon-he bench --parties 7 --threshold 6 --out bench_6of7.json
make test-go-long                                                         # precision at 10M values, race detector
```

Then set `fl: {operators: [op_a … op_g], threshold: 6, unverified: true}` in the run config. Check
two things:

- every selftest line passes;
- the max abs error, which must stay within the printed tolerance (1.4e-5 at 6-of-7), is
  acceptable for your model.

**N > 7 needs new parameters.** Three quantities change with N:

- **Headroom:** a sum of N updates must stay below Q/2Δ = 32. The per-value range already shrinks
  to ±2 at N = 4, falls to ±1 at N = 8, and the budget runs out at N = 16.
- **Flooding:** its width grows about 2·log2 N bits, because the noise bound scales with N². The
  decryption error grows with it.
- **Cost:** setup sends N² Shamir shares, and the selftest decrypts every t-subset.

Raising `MaxParties` in `internal/he/params.go` is therefore not enough. Re-derive Δ and Q for your
N and update range, re-run the noise-bound and precision tests (`internal/he`), run `check` and
`scaling` at your size, and have the parameter choice reviewed.

## Layout

```
cmd/horizon-he/       CLI (one party per call, JSON out)
internal/he/          CKKS parameters, noise bound and flooding, encrypt, aggregate, threshold decrypt
internal/protocol/    run-directory layout and path guard, party list and rules, signed statements
internal/node/        each protocol step as one party; selftest, bench, scaling, check
```
