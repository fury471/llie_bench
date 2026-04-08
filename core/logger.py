import csv
import json
from datetime import datetime
from pathlib import Path

class Logger:
    """A simple logger that saves logs to a CSV file and can also save metadata to a JSON file."""
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "logs.csv"
        self.json_path = self.log_dir / "metadata.json"
        self.fieldnames = None
        self.records = []

    def log(self, method, dataset, metric, value):
        """Log a single record to the CSV file."""
        if self.fieldnames is None:
            self.fieldnames = ['timestamp', 'method', 'dataset', 'metric', 'value']
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'method': method,
            'dataset': dataset,
            'metric': metric,
            'value': value
        }
        self.records.append(record)
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(record)

    def save(self):
        """Save all records to a JSON file."""
        with open(self.json_path, 'w') as f:
            json.dump(self.records, f, indent=4)