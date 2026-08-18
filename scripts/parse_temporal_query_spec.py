"""Parse a TRAKE query into the predicate schema used by temporal encoders."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic_pipeline.retrieval.temporal_query_expander import (  # noqa: E402
    GeminiTemporalQueryParser,
    temporal_query_to_retrieval_spec,
)
from semantic_pipeline.retrieval.temporal_query_parser import parse_temporal_query  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--without-gemini", action="store_true")
    args = parser.parse_args()

    parsed = parse_temporal_query(args.query_file.read_text(encoding="utf-8"))
    mode = "deterministic_fallback"
    if not args.without_gemini:
        expander = GeminiTemporalQueryParser(model=args.model)
        parsed = expander.parse(parsed)
        mode = expander.mode
        parser_error = expander.last_error
    else:
        parser_error = None
    output = temporal_query_to_retrieval_spec(parsed)
    output["parser_mode"] = mode
    if parser_error:
        output["parser_error"] = parser_error
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
