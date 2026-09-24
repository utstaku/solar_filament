"""Print the report for tables fetched from a run:  uv run python -m error_analysis outputs/error-analysis-cnn-errors"""

import json
import sys
from pathlib import Path

import pandas as pd

from error_analysis.report import breakdowns, summary, what_if


def main(directory: Path) -> int:
    gt = pd.read_csv(directory / "errors_gt.csv")
    pred = pd.read_csv(directory / "errors_pred.csv")
    pd.set_option("display.width", 160)
    print(json.dumps({k: v for k, v in summary(gt, pred).items() if k != "what_if"}, indent=1))
    print("\n== what if ==")
    print(what_if(gt, pred).to_string(index=False))
    for name, table in breakdowns(gt, pred).items():
        print(f"\n== {name} ==")
        print(table.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/error-analysis-cnn-errors")))
