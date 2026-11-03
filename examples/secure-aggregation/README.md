# Secure aggregation of plain vectors

The encryption layer on its own, with no model and no training.

N organisations each hold a private vector. The server adds up their ciphertexts, and any t of the
organisations together decrypt the sum and nothing else. Use this layer directly if you already
have a federated-learning loop and only need the aggregation step.

**Needs:** Go ≥ 1.24 (`make he`) and numpy (`pip install -e .`). No account and no GPU.

## Run

```bash
make he
python examples/secure-aggregation/secure_sum.py                                       # 2-of-3, 100k values
python examples/secure-aggregation/secure_sum.py --parties 5 --threshold 4 --size 4685825   # one Glimmer update each
```

Expected output, from the second command on an M2 Pro:

```
key ceremony: 4-of-5, no trusted dealer  (0.3 s)
5 parties encrypted 4,685,825 values each
decrypted by op_a, op_b, op_c, op_d: sum matches the plaintext sum, max abs error 3.8e-06
3 share(s) only: relative error 2.6e+05 (noise)
done in 9.6 s
```

## What the script does

`secure_sum.py` is about 50 lines of `horizon.he_bridge.HorizonHE` calls. Each one runs `horizon-he`
as exactly one party:

| Step | Call | Party |
|---|---|---|
| setup | `ceremony_stepwise(names, t)`: keygen, public-key share, Shamir deal and combine | each organisation, then the server |
| submit | `encrypt(party, round, vector)` | each organisation |
| sum | `aggregate(round)` | server |
| decrypt | `request(round, active)`, then `dec_share(party, round)` per active party, then `dec_combine(round)` | server, t organisations, server |

Values must lie in [−4, 4] for N = 3 and [−2, 2] for N = 4–7. Model updates are far smaller. To
pick t-of-N safely, run `make check-tofn PARTIES=N THRESHOLD=t` (see `crypto/README.md`).

## Adapt it

Replace the random vectors with your own updates, flattened to float64. The run directory is the
message bus: each party reads and writes only its own files (layout in `docs/how-it-works.md`), so
a real deployment moves the same files over its own transport.
