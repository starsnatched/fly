#!/usr/bin/env python3
"""Bootstrap the connectome binary for a fresh clone.

The engine cannot boot without data/fly-brain-full.bin (~295 MB, not in git).
This script finds or fetches it, in order:

  1. already on disk (hash-checked, no download)
  2. this repo's GitHub Releases asset `fly-brain-full.bin` (works for anyone
     with repo access; set GITHUB_TOKEN for private repos), or an explicit
     URL via the FLY_BRAIN_URL env var
  3. explains how to rebuild from the upstream Janelia MaleCNS exports

Run:  python scripts/get_brain.py
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "fly-brain-full.bin"
SHA256 = "dafefb6bb4bf20a870499a9347c64ab5d62cecb31972c85763a3cf8beec39cee"
SIZE = 309_150_560
UPSTREAM = "https://www.janelia.org/project-team/flyem/male-cns-connectome"


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_slug() -> str | None:
    try:
        url = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output, text=True, check=True, cwd=ROOT).stdout.strip()
    except Exception:
        return None
    for prefix in ("https://github.com/", "git@github.com:"):
        if url.startswith(prefix):
            return url[len(prefix):].removesuffix(".git")
    return None


def download(url: str, token: str | None, dst: Path) -> None:
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    print(f"downloading {url}")
    with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {done / 1e6:7.1f} / {SIZE / 1e6:.1f} MB", end="", flush=True)
    print()


def main() -> int:
    if OUT.exists():
        got = sha256_of(OUT)
        if got == SHA256:
            print(f"OK: {OUT} already present and hash-verified")
            return 0
        print(f"WARNING: {OUT} exists but its hash differs — refetching")

    url = os.environ.get("FLY_BRAIN_URL")
    if not url:
        slug = repo_slug()
        if slug:
            url = f"https://github.com/{slug}/releases/latest/download/fly-brain-full.bin"
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("FLY_BRAIN_TOKEN")

    if url:
        tmp = OUT.with_suffix(".bin.part")
        try:
            download(url, token, tmp)
        except Exception as e:
            tmp.unlink(missing_ok=True)
            print(f"download failed: {e}")
        else:
            got = sha256_of(tmp)
            if got != SHA256:
                tmp.unlink(missing_ok=True)
                print(f"hash mismatch after download ({got[:12]}…), discarding")
            else:
                OUT.parent.mkdir(exist_ok=True)
                tmp.replace(OUT)
                print(f"OK: {OUT} ({SIZE / 1e6:.0f} MB, hash verified)")
                return 0

    print(
        "No web source available. Two ways to get the connectome:\n"
        f"  1. Upload data/fly-brain-full.bin as a release asset named\n"
        f"     'fly-brain-full.bin' on this repo's GitHub Releases, then rerun.\n"
        f"  2. Rebuild from upstream: fetch the raw MaleCNS exports\n"
        f"     (malecns-connectome.feather, malecns-annotations.feather,\n"
        f"      neuron-nt.json, tbar-nt.feather) into data/ from {UPSTREAM},\n"
        f"     then: python scripts/extract_full_brain.py"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
