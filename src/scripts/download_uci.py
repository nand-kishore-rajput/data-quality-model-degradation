import pandas as pd

from src.registry import UCI_DATASETS

from src.loaders.uci_loader import UCILoader, extract_metadata

loader = UCILoader()

metadata_rows = []

for name, dataset_id in UCI_DATASETS.items():

    try:

        print(f"Downloading {name}")

        df, dataset = loader.load_dataset(dataset_id)

        loader.save_dataset(df, name)

        metadata_rows.append(extract_metadata(dataset, df))

        print(f"Completed {name}")

    except Exception as e:

        print(f"Failed {name}: {e}")

metadata_df = pd.DataFrame(metadata_rows)

metadata_df.to_csv("datasets/metadata/uci_metadata.csv", index=False)

print("\nAll downloads complete.")
