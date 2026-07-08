OPENML_DATASETS = {
    "phoneme": 1489,
    "bank_marketing": 1461,
    "magic_telescope": 1120,
    "churn": 40701,
    "heloc": 45023,
    "electricity": 151,
    "jm1": 1053,
    "road_safety": 44038,
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