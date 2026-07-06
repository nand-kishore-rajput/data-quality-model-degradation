from src.loaders.cleanml_loader import CleanMLLoader

loader = CleanMLLoader()

df = loader.load_csv(
    dataset_name="Airbnb",
    subfolder="raw",
    filename="dirty_train.csv"
)

print(df.shape)

print(df.head())