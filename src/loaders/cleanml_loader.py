import os
import pandas as pd
from src.registry import CLEANML_DATASET_INFO


class CleanMLLoader:

    def __init__(self, cleanml_root="datasets/cleanml"):

        self.cleanml_root = cleanml_root

    def load_csv(
        self,
        dataset_name,
        subfolder,
        filename
    ):

        path = os.path.join(
            self.cleanml_root,
            dataset_name,
            subfolder,
            filename
        )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        return pd.read_csv(path)

    def save_dataset(
        self,
        df,
        dataset_name,
        version_name,
        output_dir="datasets/raw"
    ):

        os.makedirs(output_dir, exist_ok=True)

        path = os.path.join(
            output_dir,
            f"{dataset_name}_{version_name}.parquet"
        )

        df.to_parquet(
            path,
            index=False
        )

        return path
    

def extract_metadata(
    dataset_name,
    corruption_type,
    version,
    df
):

    dataset_info = CLEANML_DATASET_INFO.get(
        dataset_name,
        {}
    )

    target = dataset_info.get("target")

    metadata = {

        "source": "cleanml",

        "dataset_name": dataset_name,

        "target": target,

        "task_type": dataset_info.get(
            "task_type"
        ),

        "target_definition": dataset_info.get(
            "target_definition"
        ),

        "corruption_type": corruption_type,

        "version": version,

        "rows": len(df),

        "columns": df.shape[1],

        "feature_count": (
            df.shape[1] - 1
            if target and target in df.columns
            else None
        ),

        "missing_cells": int(
            df.isna().sum().sum()
        ),

        "missing_percent": float(
            df.isna().sum().sum()
            /
            (df.shape[0] * df.shape[1])
            * 100
        )
    }

    return metadata