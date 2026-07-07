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

canonical_columns = [
    "source",
    "dataset_id",
    "dataset_name",
    "rows",
    "columns",
    "feature_count",
    "target",
    "missing_cells",
    "missing_percent"
]

catalog = []

for file in files:

    df = pd.read_csv(METADATA_DIR / file)

    for col in canonical_columns:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[canonical_columns]

    catalog.append(df)

benchmark_catalog = pd.concat(
    catalog,
    ignore_index=True
)

benchmark_catalog.to_csv(
    METADATA_DIR / "benchmark_catalog.csv",
    index=False
)

print("Saved benchmark_catalog.csv")
print("Rows:", len(benchmark_catalog))