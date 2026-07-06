import os
import pandas as pd

from folktables import ACSDataSource, ACSIncome, ACSPublicCoverage


class FolktablesLoader:

    def __init__(self, raw_data_dir="datasets/raw/folktables"):

        self.raw_data_dir = raw_data_dir

        os.makedirs(self.raw_data_dir, exist_ok=True)

    def load_dataset(self, dataset_name):

        data_source = ACSDataSource(
            survey_year="2018", horizon="1-Year", survey="person"
        )

        acs_data = data_source.get_data(states=["CA"], download=True)

        if dataset_name == "ACSIncome":

            X, y, group = ACSIncome.df_to_numpy(acs_data)

            feature_names = ACSIncome.features

        elif dataset_name == "ACSPublicCoverage":

            X, y, group = ACSPublicCoverage.df_to_numpy(acs_data)

            feature_names = ACSPublicCoverage.features

        else:

            raise ValueError(f"Unknown dataset {dataset_name}")

        df = pd.DataFrame(X, columns=feature_names)

        df["target"] = y

        metadata = {"dataset_name": dataset_name, "feature_names": feature_names}

        return df, metadata

    def save_dataset(self, df, dataset_name):

        path = os.path.join(self.raw_data_dir, f"{dataset_name}.parquet")

        df.to_parquet(path, index=False)

        return path


def extract_metadata(metadata, df):

    return {
        "source": "folktables",
        "dataset_name": metadata["dataset_name"],
        "rows": len(df),
        "columns": df.shape[1],
        "target": "target",
        "missing_cells": int(df.isna().sum().sum()),
        "missing_percent": float(
            df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        ),
        "feature_count": df.shape[1] - 1,
    }
