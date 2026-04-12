import csv
import json
from datetime import datetime
from pathlib import Path


class Logger:
    """A lightweight structured logger for CSV metrics and JSON metadata."""

    fieldnames = ["timestamp", "method", "dataset", "metric", "value"]

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "logs.csv"
        self.json_path = self.log_dir / "metadata.json"
        self.records = []
        self._ensure_csv_header()

    def _ensure_csv_header(self):
        """Create the CSV file with a header when it does not exist yet."""
        if self.csv_path.exists():
            return
        with open(self.csv_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, method, dataset, metric, value):
        """Append a single metric record to the CSV log."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "dataset": dataset,
            "metric": metric,
            "value": value,
        }
        self.records.append(record)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(record)

    def save(self):
        """Save the in-memory records to JSON for optional downstream inspection."""
        with open(self.json_path, "w", encoding="utf-8") as file:
            json.dump(self.records, file, indent=4)
