from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

logger = logging.getLogger(__name__)


DEFAULT_PATTERNS = [
    "README.md",
    "requirements.txt",
    "battery_analysis.ipynb",
    "src/**/*.py",
    "tests/**/*.py",
    "dashboard/**/*.py",
    "scripts/**/*.py",
    ".github/workflows/*.yml",
    "data/processed/modeling/*",
    "trained_models/*",
]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _collect_files(root: Path, patterns: list[str]) -> list[Path]:
    files: set[Path] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                files.add(path)
    return sorted(files)


def make_snapshot(
    root: Path,
    out_dir: Path,
    tag: str | None = None,
    patterns: list[str] | None = None,
) -> tuple[Path, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"snapshot_{ts}{suffix}.zip"
    manifest_path = out_dir / f"snapshot_{ts}{suffix}_manifest.json"
    latest_path = out_dir / "latest_snapshot.txt"

    files = _collect_files(root, patterns or DEFAULT_PATTERNS)
    entries = []

    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        for file_path in files:
            rel = file_path.relative_to(root)
            blob = file_path.read_bytes()
            zf.writestr(str(rel).replace("\\", "/"), blob)
            entries.append(
                {
                    "path": str(rel).replace("\\", "/"),
                    "size": len(blob),
                    "sha256": _sha256_bytes(blob),
                }
            )

    manifest = {
        "created_at": datetime.now().isoformat(),
        "snapshot_zip": str(zip_path.relative_to(root)).replace("\\", "/"),
        "file_count": len(entries),
        "files": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    latest_path.write_text(
        (
            f"snapshot_zip={zip_path.name}\n"
            f"manifest={manifest_path.name}\n"
            f"created_at={manifest['created_at']}\n"
        ),
        encoding="utf-8",
    )

    return zip_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a reproducible project snapshot.")
    parser.add_argument("--tag", default=None, help="Optional suffix (e.g., pre_demo).")
    parser.add_argument("--out-dir", default="releases", help="Output directory for snapshots.")
    parser.add_argument(
        "--root",
        default=None,
        help="Project root (defaults to parent of this script).",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    script_dir = Path(__file__).resolve().parent
    root = Path(args.root).resolve() if args.root else script_dir.parent
    out_dir = (root / args.out_dir).resolve()

    zip_path, manifest_path = make_snapshot(root=root, out_dir=out_dir, tag=args.tag)
    logger.info(f"Snapshot created: {zip_path}")
    logger.info(f"Manifest created: {manifest_path}")


if __name__ == "__main__":
    main()
