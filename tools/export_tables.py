import pandas as pd
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Export benchmark results as a table")
    parser.add_argument("--log_dir", type=str, required=True, help="Path to logs folder")
    parser.add_argument("--output", type=str, default="results/table.csv", help="Path to save table")
    args = parser.parse_args()

    # read the CSV file
    log_path = Path(args.log_dir) / "logs.csv"
    df = pd.read_csv(log_path)

    # pivot the table so metrics become columns
    table = df.pivot_table(
        index=["method", "dataset"],
        columns="metric",
        values="value"
    ).reset_index()

    print(table.to_string(index=False))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"\nTable saved to {args.output}")

if __name__ == "__main__":
    main()