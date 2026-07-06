import os
import pandas as pd

from ucimlrepo import fetch_ucirepo


class UCILoader:

    def __init__(self, raw_data_dir="datasets/raw/uci"):

        self.raw_data_dir = raw_data_dir

        os.makedirs(self.raw_data_dir, exist_ok=True)

    def load_dataset(self, dataset_id):

        dataset = fetch_ucirepo(id=dataset_id)

        X = dataset.data.features
        y = dataset.data.targets

        target = y.columns[0]

        df = X.copy()

        df[target] = y.iloc[:, 0]

        return df, dataset

    def save_dataset(self, df, dataset_name):

        path = os.path.join(self.raw_data_dir, f"{dataset_name}.parquet")

        df.to_parquet(path, index=False)

        return path


def extract_metadata(dataset, df):

    target = dataset.data.targets.columns[0]

    metadata = {
        "source": "uci",
        "dataset_id": dataset.metadata.uci_id,
        "dataset_name": dataset.metadata.name,
        "rows": len(df),
        "columns": df.shape[1],
        "target": target,
        "missing_cells": int(df.isna().sum().sum()),
        "missing_percent": float(
            df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        ),
        "feature_count": df.shape[1] - 1,
    }

    return metadata
