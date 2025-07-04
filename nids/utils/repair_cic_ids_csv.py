"""Repair CIC-IDS2017 CSV files that contain stray newlines.

Some rows in the raw dataset are split across multiple physical lines because
of embedded newline characters.  This helper rewrites such files so that each
logical record occupies a single line.

It can be used as a library function::

    from nids.utils.repair_cic_ids_csv import repair_file
    repair_file("raw.csv", "fixed.csv")

or executed as a CLI::

    python -m nids.utils.repair_cic_ids_csv input_dir output_dir
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Final

__all__: list[str] = ["repair_file", "main"]


def repair_file(in_path: str | os.PathLike[str], out_path: str | os.PathLike[str]) -> None:  # noqa: D401
    """Rewrite *in_path* to *out_path* ensuring rows are intact."""
    in_path = Path(in_path)
    out_path = Path(out_path)
    if not in_path.is_file():
        raise FileNotFoundError(in_path)

    with in_path.open("r", encoding="utf-8", errors="ignore") as f_in, out_path.open(
        "w", encoding="utf-8", newline=""
    ) as f_out:
        header = f_in.readline().rstrip("\n\r")
        expected_commas: Final[int] = header.count(",")
        f_out.write(header + "\n")

        buffer = ""
        for raw_line in f_in:
            line = raw_line.strip("\n\r")
            if not line:
                continue

            buffer += line
            if buffer.count(",") >= expected_commas:
                f_out.write(buffer + "\n")
                buffer = ""

        if buffer:
            f_out.write(buffer + "\n")

    print(f"[Repair] {in_path.name} → {out_path}")


def main(argv: list[str] | None = None) -> None:  # noqa: D401
    """CLI entry-point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", help="Directory with the original CSV files")
    parser.add_argument(
        "output_dir", help="Directory where the repaired CSVs will be written (created if absent)"
    )
    args = parser.parse_args(argv)

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    csv_files = glob.glob(str(in_dir / "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {in_dir}")

    for csv_path in csv_files:
        repair_file(csv_path, out_dir / Path(csv_path).name)

    print("[Repair] All files processed successfully.")


if __name__ == "__main__":  # pragma: no cover
    main() 