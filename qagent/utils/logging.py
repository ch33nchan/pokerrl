from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Optional


class TrainingLogger:
    """Writes per-iteration training metrics to CSV and JSONL files."""

    def __init__(
        self,
        output_dir: str,
        base_filename: str = "metrics",
        write_csv: bool = True,
        write_jsonl: bool = True,
        overwrite: bool = False,
    ) -> None:
        if not write_csv and not write_jsonl:
            raise ValueError("At least one of write_csv or write_jsonl must be True.")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_filename = base_filename
        self.write_csv = write_csv
        self.write_jsonl = write_jsonl
        self.overwrite = overwrite

        self._csv_path: Optional[Path] = None
        self._csv_file = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._fieldnames: Optional[Iterable[str]] = None

        self._jsonl_path: Optional[Path] = None
        self._jsonl_file = None

        if self.write_csv:
            self._csv_path = self._prepare_path("csv")
        if self.write_jsonl:
            self._jsonl_path = self._prepare_path("jsonl")

    def log(self, metrics: Dict[str, float]) -> None:
        if not metrics:
            return

        if self.write_csv:
            self._write_csv(metrics)
        if self.write_jsonl:
            self._write_jsonl(metrics)

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        if self._jsonl_file is not None:
            self._jsonl_file.close()
            self._jsonl_file = None

    def __enter__(self) -> "TrainingLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_path(self, extension: str) -> Path:
        filename = f"{self.base_filename}.{extension}"
        path = self.output_dir / filename
        if path.exists() and not self.overwrite:
            raise FileExistsError(
                f"Logging target '{path}' already exists. Pass overwrite=True to replace it."
            )
        if path.exists():
            path.unlink()
        return path

    def _write_csv(self, metrics: Dict[str, float]) -> None:
        if self._csv_writer is None:
            assert self._csv_path is not None
            self._csv_file = self._csv_path.open("w", newline="")
            fieldnames = list(metrics.keys())
            self._fieldnames = fieldnames
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
            self._csv_writer.writeheader()
        else:
            fieldnames = list(metrics.keys())
            if list(self._fieldnames) != fieldnames:
                raise ValueError(
                    "Metric keys changed between iterations. Ensure consistent logging keys."
                )
        assert self._csv_writer is not None
        self._csv_writer.writerow(metrics)
        if self._csv_file is not None:
            self._csv_file.flush()

    def _write_jsonl(self, metrics: Dict[str, float]) -> None:
        if self._jsonl_file is None:
            assert self._jsonl_path is not None
            self._jsonl_file = self._jsonl_path.open("w")
        json.dump(metrics, self._jsonl_file)
        self._jsonl_file.write("\n")
        self._jsonl_file.flush()


__all__ = ["TrainingLogger"]
