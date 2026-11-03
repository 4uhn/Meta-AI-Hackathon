"""Download a dataset's raw file to where its partition config expects it, and check its pinned sha256.

    python -m horizon.data.download nidd5g    # 5G-NIDD Combined.csv (Etsin, CC BY 4.0): 28 MB zip -> 275 MB
    python -m horizon.data.download asrs      # NASA ASRS export (pinned Hugging Face mirror): 103 MB

No account is needed for either. An existing file with the right hash is kept.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import yaml

from horizon.data.prepare import sha256_file

CONFIGS = {"nidd5g": "configs/partitions/5g-nidd.yaml", "asrs": "configs/partitions/asrs.yaml"}
# 5G-NIDD on Etsin (doi:10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361): open access via a download token.
ETSIN_AUTHORIZE = "https://etsin.fairdata.fi/api/download/authorize"
ETSIN_DATASET, ETSIN_FILE, ETSIN_MEMBER = (
    "9d13ef28-2ca7-44b0-9950-225359afac65",
    "/Combined.zip",
    "Combined.csv",
)
UA = {"User-Agent": "horizon-data-download"}


def _fetch(url: str, dest: Path, data: bytes | None = None, headers: dict | None = None) -> None:
    req = urllib.request.Request(url, data=data, headers=UA | (headers or {}))
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)


def _nidd5g(dest: Path, tmp: Path) -> None:
    body = json.dumps({"cr_id": ETSIN_DATASET, "file": ETSIN_FILE}).encode()
    _fetch(ETSIN_AUTHORIZE, tmp / "token.json", body, {"Content-Type": "application/json"})
    url = json.loads((tmp / "token.json").read_text())["url"]
    _fetch(url, tmp / "Combined.zip")
    with zipfile.ZipFile(tmp / "Combined.zip") as z, z.open(ETSIN_MEMBER) as src, open(dest, "wb") as out:
        shutil.copyfileobj(src, out, length=1 << 20)


def _asrs(dest: Path, source: str) -> None:
    # hf://datasets/<owner>/<name>@<revision>/<file>
    repo, _, rest = source.removeprefix("hf://datasets/").partition("@")
    rev, _, name = rest.partition("/")
    _fetch(f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{name}", dest)


def download(dataset: str, config: str | None = None) -> dict:
    cfg = yaml.safe_load(open(config or CONFIGS[dataset]))
    raw, want = Path(cfg["raw_csv"]).expanduser(), cfg["raw_sha256"]
    if raw.is_file() and sha256_file(raw) == want:
        return {"dataset": dataset, "path": str(raw), "sha256": want, "status": "already present"}
    raw.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=raw.parent) as d:
        part = Path(d) / raw.name
        if dataset == "nidd5g":
            _nidd5g(part, Path(d))
        else:
            _asrs(part, cfg["raw_source"])
        got = sha256_file(part)
        if got != want:
            raise SystemExit(f"{dataset}: downloaded sha256 {got} != pinned {want}; not saved")
        part.replace(raw)
    return {"dataset": dataset, "path": str(raw), "sha256": want, "status": "downloaded"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Download a dataset's raw file (sha256-checked).")
    ap.add_argument("dataset", choices=sorted(CONFIGS))
    ap.add_argument(
        "--config", help="partition config to read raw_csv/raw_sha256 from (default: the dataset's)"
    )
    a = ap.parse_args()
    print(json.dumps(download(a.dataset, a.config)))


if __name__ == "__main__":
    main()
