# Security model

Horizon is a research prototype. This page states what it protects, against whom, and what it does
not protect. Read it before using Horizon with real data.

## What is protected

An organisation's **individual model update** (its LoRA B matrices and weight) is encrypted on its
own machine under a collective CKKS public key before it leaves. The server sees only ciphertexts and
signed manifests. A round's update can be decrypted only as part of a **sum of at least t + 1
organisations' updates** (and at least 3), and only when t organisations cooperate.

Why it matters: from one plaintext Glimmer update, a server can tell which attack classes that
operator holds. The attack is a logit lens using public weights only:
- presence AUC 0.993 over 18 updates;
- the top 3 classes exact for 16 of 18.

On the decrypted aggregate, the same attack scores 0.508 (chance). Source:
`results/5g-nidd-glimmer/update_leakage.json`.

## Threat model

- **Server (aggregator): curious, and not trusted to follow the rules.** It may try to get more
  than the sum decrypted. Operators enforce the rules below themselves, so the attempts we test are refused:
  - labelling one organisation's ciphertext as the aggregate;
  - a sum of too few contributors;
  - tampered or replayed submissions;
  - a second decryption of a round.
- **Organisations: follow the protocol; up to t − 1 may collude with each other or the server.**
  Together they still learn only sums of at least two honest organisations' updates (with
  t = N − 1; for a smaller t see Limits).
- **Not covered:** an organisation that sends a poisoned update (no Byzantine robustness), and
  network attackers (the reference implementation exchanges files on one machine or through
  per-organisation storage; a deployment must add an authenticated, encrypted transport).

## Protocol rules (enforced by each operator before it releases a decryption share)

1. At least t + 1 distinct, correctly signed contributors (and at least 3).
2. 2t > N, so any two decrypting groups overlap.
3. t ≤ N − 1.
4. One decryption share per round, ever. Repeating a partial decryption leaks key information (see
   Lattigo's [SECURITY.md](https://github.com/tuneinsight/lattigo/blob/main/SECURITY.md), "On the
   insecurity of retries"). A failed decryption means the round is skipped, never retried.
5. The operator recomputes the aggregate from the signed submissions and checks it byte for byte.

Noise flooding: each decryption share carries exact uniform noise 2^65 times the proven ciphertext
noise bound (statistical distance ≤ 2^−64 per coefficient), so shares reveal nothing about secret
keys beyond the decrypted sum. Parameters: `crypto/README.md`.

## Limits

- **Any t colluding organisations can decrypt an individual update.** That is the meaning of a
  t-of-N threshold. Choose t accordingly. We recommend t = N − 1, where every member must submit
  every round. With a smaller t, read the note in `crypto/README.md`.
- **The decrypted sum and the final model are shared with every member.** Encryption protects
  updates in transit and at the server, not what the trained model itself reveals.
  - We measured membership inference on the final models (`results/*/membership_inference.json`):
    - 5G-NIDD: AUC 0.505, against a base-model control of 0.500 (chance);
    - aviation reports: AUC 0.541, against a control of 0.514 (a small signal).
  - **Formal differential privacy is not included yet.** It is in development as an add-on:
    each organisation would clip and add calibrated noise before encrypting, and the encrypted sum
    carries it unchanged.
- **Metadata is visible to the server:** who submitted in which round, when, and how many bytes.
- **One launcher acts for every organisation in the reference implementation.** Every
  organisation's process runs under one user on one machine, and the Modal launcher reads each
  organisation's signing key to pass it to that organisation's container only. In a deployment,
  each organisation runs its own steps, with its own keys, on its own infrastructure.
- **The plaintext reference path** (`--insecure-reference`, used to produce E2 and to measure
  leakage) writes unencrypted updates by design. It is for experiments only, and every use is logged
  as `insecure_reference_written`.
- **`make demo`** re-decrypts the final round of a finished, archived run in a sandbox copy, to show
  the protocol live. It deliberately repeats a decryption, which rule 4 forbids for keys in use.
  Never do this with a live consortium's keys.
- The cryptographic parameters and protocol have not had an independent audit.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting (Security → Report a vulnerability) on this
repository rather than a public issue.
