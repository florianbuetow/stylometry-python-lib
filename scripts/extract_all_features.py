"""Extract every deterministic stylometry feature this library offers from a text file.

Usage:
    uv run scripts/extract_all_features.py <input_text_file> [output_file]

Reads the input file as a single document, runs the library's full deterministic
feature extractor over it, and writes the resulting feature/value pairs to a CSV
file. When no output path is given, the output is written next to the input file
using the same basename with a ".features.csv" suffix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from stylometry_python_lib import default_deterministic_extractor

TEXT_COLUMN = "text"


def resolve_paths(argv: list[str]) -> tuple[Path, Path]:
    """Resolve the input and output paths from CLI arguments, failing fast."""
    if len(argv) < 2:
        raise SystemExit("Usage: uv run scripts/extract_all_features.py <input_text_file> [output_file]")
    input_path = Path(argv[1])
    if not input_path.is_file():
        raise SystemExit(f"Input file does not exist: {input_path}")
    output_path = Path(argv[2]) if len(argv) >= 3 else input_path.with_name(f"{input_path.stem}.features.csv")
    return input_path, output_path


def extract_features(text: str) -> pd.DataFrame:
    """Run every deterministic feature block over a single document."""
    document_frame = pd.DataFrame({TEXT_COLUMN: [text]})
    extractor = default_deterministic_extractor(text_column=TEXT_COLUMN, output="pandas")
    features = extractor.fit_transform(document_frame, None)
    if features.shape[0] != 1:
        raise SystemExit(f"Expected exactly one feature row, got {features.shape[0]}")
    return features


def main() -> None:
    """Read the input document, extract all features, and write them to disk."""
    input_path, output_path = resolve_paths(sys.argv)
    text = input_path.read_text(encoding="utf-8")
    features = extract_features(text)

    # Transpose the single feature row into a readable (feature, value) table.
    feature_table = features.iloc[0].rename("value").rename_axis("feature").reset_index()
    feature_table.to_csv(output_path, index=False)

    print(f"Extracted {feature_table.shape[0]} features from {input_path}")
    print(f"Wrote features to {output_path}")


if __name__ == "__main__":
    main()
