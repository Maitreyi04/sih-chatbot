# dataset_loader.py
import pandas as pd

def load_suicide_data(path="datasets/merged_suicide_dataset.csv"):
    """
    Load and clean suicide dataset.
    Expects columns: text, label
    where label = 1 (suicidal), 0 (non-suicidal).
    """
    df = pd.read_csv(path)

    # Drop bad rows like "Title"
    df = df[df["text"].str.lower() != "title"]

    # Ensure labels are integers (0 or 1)
    df["label"] = df["label"].astype(int)

    print("✅ Suicide dataset loaded and cleaned!")
    print("Shape:", df.shape)
    print("Label distribution:\n", df["label"].value_counts())
    print("\nSample:\n", df.head())

    return df

if __name__ == "__main__":
    df = load_suicide_data()
