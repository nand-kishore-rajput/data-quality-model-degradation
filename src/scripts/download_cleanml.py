import shutil
import zipfile
from pathlib import Path

import requests


CLEANML_URL = (
    "https://www.dropbox.com/scl/fi/"
    "p4atgd4383fowik1cce5l/"
    "CleanML-datasets-2020.zip"
    "?rlkey=9aiwy8cwgq4tz89fg4jzceerc&dl=1"
)


EXPECTED_DATASETS = [
    "Airbnb",
    "EEG",
    "KDD",
]


def download_file(url: str, destination: Path) -> None:

    print("Downloading CleanML datasets...")

    response = requests.get(
        url,
        stream=True,
        timeout=300,
    )

    response.raise_for_status()

    with open(destination, "wb") as f:

        for chunk in response.iter_content(
            chunk_size=8192
        ):

            if chunk:
                f.write(chunk)

    print("Download complete.")


def create_directory_structure(datasets_dir: Path) -> None:

    required_dirs = [
        datasets_dir / "raw",
        datasets_dir / "processed",
        datasets_dir / "corrupted",
        datasets_dir / "metadata",
    ]

    for directory in required_dirs:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def verify_extraction(cleanml_path: Path) -> list[str]:

    missing = []

    for dataset in EXPECTED_DATASETS:

        if not (cleanml_path / dataset).exists():

            missing.append(dataset)

    return missing


def main():

    datasets_dir = Path("datasets")

    datasets_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    create_directory_structure(
        datasets_dir
    )

    zip_path = (
        datasets_dir
        / "CleanML-datasets-2020.zip"
    )

    cleanml_path = (
        datasets_dir
        / "cleanml"
    )

    if cleanml_path.exists():

        print(
            "CleanML already exists. Skipping download."
        )

        return

    try:

        download_file(
            CLEANML_URL,
            zip_path,
        )

        print("Extracting archive...")

        with zipfile.ZipFile(
            zip_path,
            "r",
        ) as zip_ref:

            zip_ref.extractall(
                datasets_dir
            )

        print(
            "Extraction complete."
        )

        macosx_dir = (
            datasets_dir
            / "__MACOSX"
        )

        if macosx_dir.exists():

            shutil.rmtree(
                macosx_dir
            )

            print(
                "Removed __MACOSX."
            )

        data_dir = (
            datasets_dir
            / "data"
        )

        if not data_dir.exists():

            raise FileNotFoundError(
                "Expected extracted folder "
                "'datasets/data' not found."
            )

        data_dir.rename(
            cleanml_path
        )

        print(
            "Renamed data -> cleanml"
        )

        missing = verify_extraction(
            cleanml_path
        )

        if missing:

            raise RuntimeError(
                f"Verification failed. "
                f"Missing datasets: {missing}"
            )

        if zip_path.exists():

            zip_path.unlink()

            print(
                "Removed ZIP file."
            )

        print(
            "\nCleanML setup completed successfully."
        )

        print(
            f"Location: {cleanml_path}"
        )

    except Exception as e:

        print(
            f"\nSetup failed: {e}"
        )

        raise


if __name__ == "__main__":
    main()