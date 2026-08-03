"""Migrate legacy Task 1 metadata into the canonical versioned schema."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from schemas import (
        DEFAULT_METADATA_PATH,
        SCHEMA_VERSION,
        FrameMetadata,
        load_metadata_file,
        validate_metadata_records,
    )
except ImportError:
    from .schemas import (
        DEFAULT_METADATA_PATH,
        SCHEMA_VERSION,
        FrameMetadata,
        load_metadata_file,
        validate_metadata_records,
    )

DEFAULT_OUTPUT_PATH = DEFAULT_METADATA_PATH.with_name("metadata_v1.json")


def migrate_records(raw_records: list[dict]) -> list[dict]:
    records: list[FrameMetadata] = validate_metadata_records(raw_records)
    # mode="json" converts enums/datetime/tuples to JSON-compatible values.
    return [record.model_dump(mode="json") for record in records]


def write_json_atomically(path: Path, data: list[dict], overwrite: bool = False) -> None:
    path = path.resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; choose another --output or pass --force"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def migrate_file(
    input_path: str | Path,
    output_path: str | Path,
    overwrite: bool = False,
) -> list[dict]:
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if input_path == output_path:
        raise ValueError("input and output paths must differ; migration never overwrites source")
    migrated = migrate_records(load_metadata_file(input_path))
    write_json_atomically(output_path, migrated, overwrite=overwrite)
    return migrated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate legacy semantic metadata to schema v1 without overwriting source."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacing an existing output file (the input is never replaced).",
    )
    args = parser.parse_args()

    try:
        migrated = migrate_file(args.input, args.output, overwrite=args.force)
    except (ValueError, FileExistsError) as exc:
        parser.exit(status=1, message=f"Migration failed: {exc}\n")
    print(
        f"Migrated {len(migrated)} records to schema v{SCHEMA_VERSION}: "
        f"{args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
