import pandas as pd

from src.registry import TABLESHIFT_DATASETS

from src.loaders.tableshift_loader import (
    TableShiftLoader,
    extract_metadata
)

loader = TableShiftLoader()

metadata_rows = []

for dataset_name, config in TABLESHIFT_DATASETS.items():

    try:

        print(
            f"Processing {dataset_name}"
        )

        df = loader.load_dataset(
            config["path"]
        )

        loader.save_dataset(
            df,
            dataset_name
        )

        metadata_rows.append(
            extract_metadata(
                dataset_name=dataset_name,
                target=config["target"],
                domain_split_variable=config[
                    "domain_split_variable"
                ],
                df=df
            )
        )

        print(
            f"Completed {dataset_name}"
        )

    except Exception as e:

        print(
            f"Failed {dataset_name}: {e}"
        )

metadata_df = pd.DataFrame(
    metadata_rows
)

metadata_df.to_csv(
    "datasets/metadata/tableshift_metadata.csv",
    index=False
)

print(
    "\nAll TableShift datasets processed."
)