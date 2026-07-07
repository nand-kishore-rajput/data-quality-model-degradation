import os
import pandas as pd


class TableShiftLoader:

    def __init__(self, raw_data_dir="datasets/raw"):

        self.raw_data_dir = raw_data_dir

        os.makedirs(self.raw_data_dir, exist_ok=True)

    def load_dataset(self, file_path):

        df = pd.read_csv(file_path)

        return df

    def save_dataset(self, df, dataset_name):

        path = os.path.join(
            self.raw_data_dir,
            f"{dataset_name}.parquet"
        )

        df.to_parquet(
            path,
            index=False
        )

        return path


def extract_metadata(
    dataset_name,
    target,
    domain_split_variable,
    df
):

    metadata = {

        "source": "tableshift",

        "dataset_name": dataset_name,

        "rows": len(df),

        "columns": df.shape[1],

        "target": target,

        "domain_split_variable": domain_split_variable,

        "feature_count": df.shape[1] - 1,

        "missing_cells": int(
            df.isna().sum().sum()
        ),

        "missing_percent": float(
            df.isna().sum().sum()
            /
            (df.shape[0] * df.shape[1])
            * 100
        ),

        "numerical_features": len(
            df.select_dtypes(
                include="number"
            ).columns
        ),

        "categorical_features": len(
            df.select_dtypes(
                exclude="number"
            ).columns
        )
    }

    return metadata