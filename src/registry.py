OPENML_DATASETS = {
    "phoneme": 1489,
    "bank_marketing": 1461,
    "magic_telescope": 1120,
    "churn": 40701,
    "heloc": 46932,
    "electricity": 151,
    "jm1": 1053,
    "road_safety": 42803,
    "kick": 41162,
    "coil2000": 298,
    "seismic_bumps": 46956,
    "house_16H": 821,
}

UCI_DATASETS = {
    "default_credit": 350,
    "online_shoppers": 468,
    "taiwanese_bankruptcy": 572,
    "polish_bankruptcy": 365,
    "diabetes_readmission": 296
}

FOLKTABLES_DATASETS = {
    "ACSIncome": "income",
    "ACSPublicCoverage": "public_coverage",
}

CLEANML_DATASETS = {
    "Airbnb": "datasets/cleanml/Airbnb",
    "EEG": "datasets/cleanml/EEG",
    "Credit": "datasets/cleanml/Credit",
    "Movie": "datasets/cleanml/Movie",
    "Restaurant": "datasets/cleanml/Restaurant",
}

CLEANML_DATASET_INFO = {
    "Airbnb": {
        "target": "Rating",
        "task_type": "binary_classification"
    },
    "EEG": {
        "target": "Eye",
        "task_type": "binary_classification"
    },
    "Credit": {
        "target": "SeriousDlqin2yrs",
        "task_type": "binary_classification"
    },
    "Movie": {
        "target": "Rating",
        "task_type": "binary_classification"
    },
    "Restaurant": {
        "target": "is_match",
        "task_type": "binary_classification"
    }
}

TABLESHIFT_DATASETS = {

    "assistments": {
        "path": (
            "datasets/kaggle/assistments/"
            "2012-2013-data-with-predictions-4-final.csv"
        ),
        "target": "correct",
        "domain_split_variable": "school_id"
    }

}


'''
TableShift Datasets:

| Dataset              | Source     |
| -------------------- | ---------- |
| diabetes_readmission | UCI 296    |
| assistments          | Kaggle     | https://www.kaggle.com/datasets/nicolaswattiez/skillbuilder-data-2009-2010/https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect
| ACSIncome            | Folktables |
| ACSPublicCoverage    | Folktables |
| HELOC                | OpenML     |
'''