from src.loaders.openml_loader import OpenMLLoader, extract_metadata

loader = OpenMLLoader()

df, dataset = loader.load_dataset(1461)

metadata = extract_metadata(dataset, df)

print(metadata)

loader.save_dataset(df, "bank_marketing")
