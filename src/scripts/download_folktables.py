import pandas as pd

from src.registry import FOLKTABLES_DATASETS

from src.loaders.folktables_loader import FolktablesLoader, extract_metadata

loader = FolktablesLoader()

metadata_rows = []

for dataset_name in FOLKTABLES_DATASETS:

    try:

        print(f"Downloading {dataset_name}")

        df, metadata = loader.load_dataset(dataset_name)

        loader.save_dataset(df, dataset_name)

        metadata_rows.append(extract_metadata(metadata, df))

        print(f"Completed {dataset_name}")

    except Exception as e:

        print(f"Failed {dataset_name}: {e}")

metadata_df = pd.DataFrame(metadata_rows)

metadata_df.to_csv("datasets/metadata/folktables_metadata.csv", index=False)

print("\nAll downloads complete.")
