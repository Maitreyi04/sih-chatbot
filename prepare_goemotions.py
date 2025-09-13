import os
import pandas as pd

# Path to your data folder
DATA_DIR = "data"

# 1) Read emotions.txt -> build id2emotion dictionary
EMOTIONS_FILE = os.path.join("emotions.txt")
with open(EMOTIONS_FILE, "r", encoding="utf-8") as f:
    emotions = [line.strip() for line in f if line.strip()]

id2emotion = {i: e for i, e in enumerate(emotions)}
print("Loaded", len(id2emotion), "emotions")
print("Example mapping:", list(id2emotion.items())[:5])

# 2) Function to decode a label string like "0,1,26" -> ["admiration", "amusement", "sadness"]
def decode_label_ids(label_str):
    if pd.isna(label_str):
        return []
    ids = [int(x) for x in str(label_str).split(",") if x.strip().isdigit()]
    return [id2emotion[i] for i in ids if i in id2emotion]
# 4) Define chatbot buckets (grouped emotions)
BUCKETS = {
    "Sadness/Depression": {"sadness", "disappointment", "remorse", "grief", "embarrassment"},
    "Anxiety/Stress": {"nervousness", "fear", "confusion", "realization", "desire"},
    "Anger/Frustration": {"anger", "annoyance", "disgust", "disapproval"},
    "Loneliness/Isolation": {"loneliness"},
    "Positive/Neutral": {"joy", "love", "optimism", "admiration", "gratitude", "relief", "neutral"},
}

def map_to_buckets(emotion_list):
    mapped = set()
    for emo in emotion_list:
        for bucket, emos in BUCKETS.items():
            if emo in emos:
                mapped.add(bucket)
    return list(mapped) if mapped else ["Other"]


# 3) Process one dataset file
def process_tsv(path):
    df = pd.read_csv(path, sep="\t", header=None, names=["text","labels","comment_id"], engine="python", encoding="utf-8")
    df["emotions"] = df["labels"].apply(decode_label_ids)
    df["buckets"] = df["emotions"].apply(map_to_buckets)
    df = df.drop(columns=["comment_id"])  # drop comment_id
    return df

    

if __name__ == "__main__":
    train_df = process_tsv("train.tsv")
    dev_df = process_tsv("dev.tsv")
    test_df = process_tsv("test.tsv")

    # Save processed datasets
    train_df.to_csv("train_clean.csv", index=False)
    dev_df.to_csv("dev_clean.csv", index=False)
    test_df.to_csv("test_clean.csv", index=False)

    print("✅ Clean datasets saved: train_clean.csv, dev_clean.csv, test_clean.csv")
    print(train_df.head(10))

