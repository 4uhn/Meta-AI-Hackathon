GO ?= go
HE_PKG := ./cmd/horizon-he
LDFLAGS := -s -w

.PHONY: help release crypto-scaling test test-go test-py test-slow test-go-long he fmt selftest bench crypto-report setup data data-asrs e1a \
	fl-local e2 e3 e5 baselines equivalence report modal-provision modal-test demo check-tofn doctor

PY := .venv/bin/python

help:
	@echo "Horizon: federated LoRA fine-tuning with t-of-N threshold-encrypted aggregation"
	@echo ""
	@echo "Set up"
	@echo "  make setup            Python env (.venv) with every extra        PYTHON=python3.11 if needed"
	@echo "  make he               build the crypto CLI (Go >= 1.24)"
	@echo "  make doctor           check this machine and your Modal account; prints the next step"
	@echo "  source .venv/bin/activate   so that 'python' below is the project environment"
	@echo "Try it (no account)"
	@echo "  python examples/secure-aggregation/secure_sum.py          encryption layer on plain vectors"
	@echo "  python examples/federated-fine-tuning/run.py              encrypted FL on 5G-NIDD, tiny model, CPU"
	@echo "  make check-tofn PARTIES=5 THRESHOLD=4                     is this t-of-N safe? (rules + selftest)"
	@echo "Data (downloaded, sha256-checked)"
	@echo "  make data             5G-NIDD → data/processed/ (one shard per organisation + test set)"
	@echo "  make data-asrs        NASA ASRS → data/asrs/ (3 organisations), data/asrs4/ (4)"
	@echo "On your Modal account (docs/run-on-modal.md)"
	@echo "  python examples/federated-fine-tuning/run.py --on modal --model tiny|llama|glimmer"
	@echo "  make e3 | e2 | e5 BACKEND=modal CFG=configs/<config>.yaml"
	@echo "Tests and measurements"
	@echo "  make test | test-slow | test-go-long"
	@echo "  make selftest | bench | crypto-scaling | crypto-report"

test: he test-go test-py

# Python unit tests (bridge tests drive the real horizon-he binary; build it with `make he`).
test-py:
	$(PY) -m pytest -q

# End-to-end FL loop (E3, E2, E5) on CPU with a tiny random Llama + real horizon-he (~2 min, no GPU).
test-slow:
	$(PY) -m pytest -q -m slow

# Python environment (.venv) with every extra. PYTHON=python3.11 if 3.12 isn't installed.
PYTHON ?= python3.12
setup:
	$(PYTHON) -m venv .venv && $(PY) -m pip install -q --upgrade pip && $(PY) -m pip install -q -e '.[train,modal,dev]'

# Download the raw dataset (no account; sha256-checked), then split it into one shard per organisation.
# 5G-NIDD → data/processed/{op_a,op_b,op_c}/, test.jsonl, stats.json
data:
	$(PY) -m horizon.data.download nidd5g
	$(PY) -m horizon.data.prepare --config configs/partitions/5g-nidd.yaml
# NASA ASRS → data/asrs/ (3 organisations); data/asrs4/ adds a fourth (business aviation)
data-asrs:
	$(PY) -m horizon.data.download asrs
	$(PY) -m horizon.data.asrs --config configs/partitions/asrs.yaml
	$(PY) -m horizon.data.asrs --config configs/partitions/asrs4.yaml

# Model runs. Real models run on Modal (BACKEND=modal CFG=configs/5g-nidd-glimmer.yaml); locally, the
# tiny model checks the pipeline (build it first: python -m horizon.model.tiny --out models/tiny).
CFG ?= configs/5g-nidd-tiny.yaml
BACKEND ?= local

# op_a local-only baseline (E1a) → the config's results_dir/E1a.json
e1a:
	$(PY) -m horizon.fl.baseline --exp E1a --config $(CFG)

# E0, E1a–c, E4 (non-federated baselines), one model process at a time.
baselines:
	for e in E0 E1a E1b E1c E4; do $(PY) -m horizon.fl.baseline --exp $$e --config $(CFG) || exit 1; done

# Federated runs: E3 = encrypted (the headline), E2 = plaintext reference, E5 = async scenario.
fl-local e3:
	$(PY) -m horizon.fl.run --config $(CFG) --backend $(BACKEND) --mode encrypted --fresh
e2:
	$(PY) -m horizon.fl.run --config $(CFG) --backend $(BACKEND) --mode plaintext --fresh
e5:
	$(PY) -m horizon.fl.run --config $(CFG) --backend $(BACKEND) --mode encrypted --scenario e5 --fresh

# Modal: one-time volumes (shards, test set, tiny test model), then the CPU pipeline test (cents).
modal-provision:
	$(PY) -m modal_app.provision data && $(PY) -m modal_app.provision tiny
modal-test:
	$(PY) -m horizon.fl.run --backend modal --config configs/5g-nidd-tiny-modal.yaml --mode encrypted \
		--max-steps 2 --limit-test 20 --fresh

# Equivalence: replay E2's plaintext updates through the real threshold path and compare with E3
# (prediction agreement at fp32). Report: the detection matrix and blind-spot summary.
AGREE_CFG ?= configs/5g-nidd-llama-fp32-eval.yaml
equivalence:
	$(PY) -m horizon.eval.equivalence --e2 results/5g-nidd-llama/E2.json --e3 results/5g-nidd-llama/E3.json --replay \
		--config configs/5g-nidd-llama.yaml --agreement-config $(AGREE_CFG) --backend $(BACKEND)
report:
	$(PY) -m horizon.eval.report --results results/5g-nidd-llama

# Fast suite (skips T4, the 10M-value precision test).
test-go:
	cd crypto && $(GO) vet ./... && $(GO) test -short ./...

# Everything, including T4, under the race detector for the parallel code paths.
test-go-long:
	cd crypto && $(GO) test -race -timeout 30m ./...

# horizon-he for the Mac (darwin/arm64) and for the Modal image (linux/amd64, static).
he:
	cd crypto && $(GO) build -trimpath -ldflags '$(LDFLAGS)' -o bin/horizon-he $(HE_PKG)
	cd crypto && CGO_ENABLED=0 GOOS=linux GOARCH=amd64 $(GO) build -trimpath -ldflags '$(LDFLAGS)' -o bin/horizon-he-linux-amd64 $(HE_PKG)

# Prebuilt horizon-he for a GitHub Release (attach dist/* to the release): static, 4 platforms + SHA256SUMS.
release:
	rm -rf dist && mkdir -p dist
	cd crypto && for t in darwin/arm64 darwin/amd64 linux/amd64 linux/arm64; do \
		CGO_ENABLED=0 GOOS=$${t%/*} GOARCH=$${t#*/} $(GO) build -trimpath -ldflags '$(LDFLAGS)' \
			-o ../dist/horizon-he-$${t%/*}-$${t#*/} $(HE_PKG) || exit 1; done
	cp "$$(cd crypto && $(GO) list -m -f '{{.Dir}}' github.com/tuneinsight/lattigo/v6)/NOTICE" dist/NOTICE-lattigo
	cd dist && shasum -a 256 horizon-he-* > SHA256SUMS && cat SHA256SUMS

selftest: he
	crypto/bin/horizon-he selftest

# Crypto benchmark: Glimmer payload size (default) and 10M values → results/.
bench: he
	crypto/bin/horizon-he bench --out results/crypto/bench.json
	crypto/bin/horizon-he bench --n 10000000 --out results/crypto/bench_10M.json

# Measured test outputs (noise vs bound, flood uniformity, T1–T5 errors, T4 at 10M) → results/.
crypto-report: he
	cd crypto && HORIZON_SWEEP=1 $(GO) test -count=1 -v -run 'Sweep|Noise|Flood|T1|T2|T3|T4|T5|Canonical|RowNorm' ./internal/he/ | grep -E '^\s+\w+_test.go|^(--- |ok|FAIL)' > ../results/crypto/tests.txt; \
		! grep -q '^FAIL' ../results/crypto/tests.txt
	crypto/bin/horizon-he selftest --n 200000 2> results/crypto/selftest_200k.txt > results/crypto/selftest_200k.json

# t-of-N scaling at the Glimmer payload: selftest + bench for 2-of-3, 3-of-4, 3-of-5, 4-of-5
# → results/crypto/scaling.json + .txt (needs ~10 GB of temporary disk).
crypto-scaling: he
	crypto/bin/horizon-he scaling --out results/crypto/scaling.json

# Is a t-of-N consortium safe to run? Rules, parameters, then the full selftest at the Glimmer payload.
# make check-tofn PARTIES=5 THRESHOLD=4   (N ≤ 5 measured; N = 6 or 7 also needs UNVERIFIED=1)
PARTIES ?= 3
THRESHOLD ?= 2
check-tofn: he
	crypto/bin/horizon-he check --parties $(PARTIES) --threshold $(THRESHOLD) $(if $(UNVERIFIED),--unverified) > /dev/null

# What is installed and configured (Go, binaries, venv, data, Modal account and volumes), and what to run next.
doctor:
	python3 -m horizon.doctor

fmt:
	cd crypto && gofmt -w .

# Live, narrated replay of the Glimmer run's final round on its real encrypted updates (~1 min).
# STEP=1: press Enter to advance. DETAIL=1: also show hashes, timings, verbatim replies.
demo:
	$(PY) -m horizon.demo $(if $(STEP),--step) $(if $(DETAIL),--detail) $(DEMO_ARGS)
