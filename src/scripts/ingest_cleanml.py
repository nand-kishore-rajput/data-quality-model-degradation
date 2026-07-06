import pandas as pd

from src.registry import CLEANML_DATASETS

from src.loaders.cleanml_loader import (
    CleanMLLoader,
    extract_metadata
)

loader = CleanMLLoader()

metadata_rows = []

for dataset_name in CLEANML_DATASETS:

    print(
        f"\nProcessing {dataset_name}"
    )

    #
    # RAW
    #

    try:

        df = loader.load_csv(
            dataset_name,
            "raw",
            "raw.csv"
        )

        loader.save_dataset(
            df,
            dataset_name,
            "raw"
        )

        metadata_rows.append(
            extract_metadata(
                dataset_name,
                "raw",
                "raw",
                df
            )
        )

    except Exception as e:

        print(
            f"Raw not found: {e}"
        )

metadata_df = pd.DataFrame(
    metadata_rows
)

metadata_df.to_csv(
    "datasets/metadata/cleanml_metadata.csv",
    index=False
)

print("\nDone")