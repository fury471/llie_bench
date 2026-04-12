import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Export benchmark results as a table")
    parser.add_argument("--log_dir", type=str, required=True, help="Path to a logs folder")
    parser.add_argument("--output", type=str, default="results/table.csv", help="Path to save the exported table")
    args = parser.parse_args()

    log_path = Path(args.log_dir) / "logs.csv"
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    dataframe = pd.read_csv(log_path)
    table = dataframe.pivot_table(
        index=["method", "dataset"],
        columns="metric",
        values="value",
    ).reset_index()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)

    print(table.to_string(index=False))
    print(f"\nTable saved to {output_path}")


if __name__ == "__main__":
    main()
