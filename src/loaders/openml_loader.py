import os
import pandas as pd
import openml


class OpenMLLoader:

    def __init__(self, raw_data_dir="datasets/raw/openml"):

        self.raw_data_dir = raw_data_dir

        os.makedirs(self.raw_data_dir, exist_ok=True)

    def load_dataset(self, dataset_id):

        dataset = openml.datasets.get_dataset(dataset_id)

        target = dataset.default_target_attribute

        X, y, categorical, attribute_names = dataset.get_data(target=target)

        df = X.copy()

        df[target] = y

        return df, dataset

    def save_dataset(self, df, dataset_name):

        path = os.path.join(self.raw_data_dir, f"{dataset_name}.parquet")

        df.to_parquet(path, index=False)

        return path


def extract_metadata(dataset, df):

    target = dataset.default_target_attribute

    metadata = {
        "source": "openml",
        "dataset_id": dataset.dataset_id,
        "dataset_name": dataset.name,
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
