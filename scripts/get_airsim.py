"""Download and install an AirSim prebuilt environment for the flybrain bridge.

Hash-checks what is already on disk (by size — GitHub does not publish
checksums for the release zips), downloads the zip with resume support,
extracts it, and installs config/airsim.settings.json into
~/Documents/AirSim/settings.json if that file does not exist yet.

Usage:
  python scripts/get_airsim.py                # Blocks (259 MB, fast start)
  python scripts/get_airsim.py airsimnh       # realistic neighborhood (1.7 GB)
  python scripts/get_airsim.py --list         # available environments
  python scripts/get_airsim.py airsimnh --force-settings
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "airsim"
DOCS_SETTINGS = Path.home() / "Documents" / "AirSim" / "settings.json"

BASE = "https://github.com/microsoft/AirSim/releases/download/v1.8.1-windows"

ENVS: dict[str, dict] = {
    "blocks": {
        "zip": "Blocks.zip",
        "size": 259_463_081,
        "desc": "light blocky obstacle course; quick to try",
    },
    "airsimnh": {
        "zip": "AirSimNH.zip",
        "size": 1_709_782_828,
        "desc": "realistic suburban neighborhood (houses, trees, roads)",
    },
    "africa": {
        "zip": "Africa.zip",
        "size": 712_371_132,
        "desc": "natural landscape with scattered structures",
    },
    "mountains": {
        "zip": "LandscapeMountains.zip",
        "size": 750_698_256,
        "desc": "open mountain terrain",
    },
}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(url: str, dest: Path, expected_size: int) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    already = part.stat().st_size if part.exists() else 0
    if already:
        print(f"resuming from {human(already)}")
    req = urllib.request.Request(url, headers={"Range": f"bytes={already}-"})
    start = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp, \
            open(part, "ab") as f:
        total = expected_size
        done = already
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if time.time() - start > 5:
                rate = (done - already) / (time.time() - start)
                eta = (total - done) / max(rate, 1)
                print(f"\r{human(done)} / {human(total)}  "
                      f"{human(rate)}/s  eta {eta:3.0f}s   ", end="", flush=True)
    print()
    if done != expected_size:
        sys.exit(f"size mismatch: got {done}, expected {expected_size}; "
                 "delete the .part file and retry if this persists")
    part.rename(dest)
    print(f"saved {dest.name} ({human(done)})")


def find_exe(dest_dir: Path) -> Path | None:
    """Locate the environment launcher exe (name varies per environment:
    Blocks.exe, AirSimNH.exe, ...). Skips engine/binaries helpers."""
    hits = [p for p in sorted(dest_dir.glob("*/WindowsNoEditor/*.exe"))
            if "Binaries" not in p.parts and "Engine" not in p.parts]
    return hits[0] if hits else None


def extract(zip_path: Path, dest_dir: Path) -> None:
    if find_exe(dest_dir) is not None:
        print("already extracted; skipping")
        return
    print(f"extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
    print(f"extracted to {dest_dir}")


def install_settings(force: bool) -> None:
    src = ROOT / "config" / "airsim.settings.json"
    DOCS_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    if DOCS_SETTINGS.exists() and not force:
        print(f"settings already present at {DOCS_SETTINGS} "
              "(--force-settings to overwrite)")
        return
    shutil.copy(src, DOCS_SETTINGS)
    print(f"installed {DOCS_SETTINGS}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("env", nargs="?", default="blocks", choices=sorted(ENVS),
                    help="environment to download (default: blocks)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force-settings", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        for name, meta in ENVS.items():
            print(f"  {name:10s} {human(meta['size']):>9s}  {meta['desc']}")
        return 0

    meta = ENVS[args.env]
    zip_path = DEST / meta["zip"]
    DEST.mkdir(exist_ok=True)

    if zip_path.exists() and zip_path.stat().st_size == meta["size"]:
        print(f"{zip_path.name} already on disk")
    else:
        download(f"{BASE}/{meta['zip']}", zip_path, meta["size"])

    extract(zip_path, DEST)
    install_settings(args.force_settings)

    exe = find_exe(DEST)
    if exe is None:
        sys.exit("extraction finished but no environment exe found; report this")
    print("\nready. launch with:")
    print(f"  {exe.as_posix()} -opengl4")
    print("then:")
    print("  bash scripts/start_airsim_stack.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
