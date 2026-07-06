import pandas as pd

from src.registry import OPENML_DATASETS

from src.loaders.openml_loader import OpenMLLoader, extract_metadata

loader = OpenMLLoader()

metadata_rows = []

for name, dataset_id in OPENML_DATASETS.items():

    try:

        print(f"Downloading {name}")

        df, dataset = loader.load_dataset(dataset_id)

        loader.save_dataset(df, name)

        meta = extract_metadata(dataset, df)

        metadata_rows.append(meta)

        print(f"Completed {name}")

    except Exception as e:

        print(f"Failed {name}: {e}")

metadata_df = pd.DataFrame(metadata_rows)

metadata_df.to_csv("datasets/metadata/openml_metadata.csv", index=False)

print("\nAll downloads complete.")
