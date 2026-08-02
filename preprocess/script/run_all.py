from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all metadata preprocessing stages.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--boilerplate-threshold", type=float, default=0.80)
    return parser.parse_args()


def run(script_dir: Path, script_name: str, *args: str) -> None:
    command = [sys.executable, str(script_dir / script_name), *args]
    print("\n>", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    input_dir = str(args.input_dir)
    output_dir = str(args.output_dir)

    run(script_dir, "01_audit_metadata.py", "--input-dir", input_dir, "--output-dir", output_dir)
    run(script_dir, "02_profile_metadata.py", "--input-dir", input_dir, "--output-dir", output_dir)
    run(
        script_dir,
        "03_clean_metadata.py",
        "--input-dir",
        input_dir,
        "--output-dir",
        output_dir,
        "--boilerplate-threshold",
        str(args.boilerplate_threshold),
    )
    run(
        script_dir,
        "04_extract_derived_fields.py",
        "--input-file",
        str(args.output_dir / "03_clean_metadata.jsonl"),
        "--output-dir",
        output_dir,
    )
    run(
        script_dir,
        "05_build_search_documents.py",
        "--input-file",
        str(args.output_dir / "04_enriched_metadata.jsonl"),
        "--output-dir",
        output_dir,
    )
    run(
        script_dir,
        "06_build_bm25_index.py",
        "--input-file",
        str(args.output_dir / "05_metadata_search_documents.jsonl"),
        "--output-file",
        str(args.output_dir / "06_metadata_bm25.pkl"),
    )

    print("\nAll metadata preprocessing stages completed.")


if __name__ == "__main__":
    main()
