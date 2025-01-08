import gdown
import zipfile


def fetch_dataset():
    file_ids = [
        "1ki-aYI07KEbi7mWsPWxeAvmMfVsWrBCq",
        "1aIzlyrPTgrzLKwGS1_3alvNJKgLvbGQ3",
    ]
    file_names = ["patients.zip", "classes.zip"]

    for id, name in zip(reversed(file_ids), reversed(file_names)):
        url = f"https://drive.google.com/uc?id={id}"
        gdown.download(url, name, quiet=False)
        with zipfile.ZipFile(name, "r") as zip_ref:
            zip_ref.extractall("./")
