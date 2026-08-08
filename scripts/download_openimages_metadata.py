"""Download the two small official Open Images V4 ontology files."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import paths  # noqa: E402


FILES = {
    paths.OPENIMAGES_HIERARCHY: (
        "https://storage.googleapis.com/openimages/2018_04/"
        "bbox_labels_600_hierarchy.json"
    ),
    paths.OPENIMAGES_CLASS_DESCRIPTIONS: (
        "https://storage.googleapis.com/openimages/2018_04/"
        "class-descriptions-boxable.csv"
    ),
}


def download(destination: Path, url: str, *, force: bool = False) -> None:
    if destination.is_file() and not force:
        print(f"skip existing {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, temporary)
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"Downloaded empty file from {url}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"downloaded {url} -> {destination}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for destination, url in FILES.items():
        download(destination, url, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
