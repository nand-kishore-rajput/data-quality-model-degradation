import pandas as pd
from pathlib import Path

METADATA_DIR = Path("datasets/metadata")

files = [
    "openml_metadata.csv",
    "uci_metadata.csv",
    "folktables_metadata.csv",
    "cleanml_metadata.csv",
    "tableshift_metadata.csv"
]

all_columns = {}

for file in files:
    df = pd.read_csv(METADATA_DIR / file)

    all_columns[file] = list(df.columns)

max_cols = max(len(cols) for cols in all_columns.values())

comparison = {}

for file, cols in all_columns.items():
    comparison[file] = cols + [""] * (max_cols - len(cols))

comparison_df = pd.DataFrame(comparison)

comparison_df.to_csv(
    METADATA_DIR / "metadata_column_comparison.csv",
    index=False
)

print(comparison_df)