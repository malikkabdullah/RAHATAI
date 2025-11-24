"""
STEP 2: PREPROCESS DATA
This step cleans the text and prepares the data for machine learning models.
"""

import pandas as pd
import re
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json

print("=" * 70)
print("STEP 2: PREPROCESSING DATA")
print("=" * 70)

# ============================================================
# 1. LOAD COMBINED DATASET
# ============================================================
print("\nLoading combined dataset...")

train_df = pd.read_csv("Data/Combined/combined_train.csv")
dev_df = pd.read_csv("Data/Combined/combined_dev.csv")
test_df = pd.read_csv("Data/Combined/combined_test.csv")

print("Loaded:")
print("   Training:", len(train_df), "samples")
print("   Dev:     ", len(dev_df), "samples")
print("   Test:    ", len(test_df), "samples")

# ============================================================
# 2. TEXT CLEANING FUNCTION
# ============================================================
def clean_text(text):
    """
    This function cleans the input text.
    It removes URLs, usernames, extra spaces, and converts text to lowercase.
    """

    # If text is empty or NaN, return empty string
    if pd.isna(text) or text == "":
        return ""

    # Convert to string
    text = str(text)

    # Remove URLs (http/https/www)
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove special placeholders like 'httpAddress'
    text = re.sub(r'httpAddress', '', text)

    # Remove @usernames or usrId
    text = re.sub(r'@\w+|usrId', '', text)

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Remove space from start/end
    text = text.strip()

    return text

# ============================================================
# 3. APPLY TEXT CLEANING
# ============================================================
print("\nCleaning text...")

train_df['text_cleaned'] = train_df['text'].apply(clean_text)
dev_df['text_cleaned'] = dev_df['text'].apply(clean_text)
test_df['text_cleaned'] = test_df['text'].apply(clean_text)

# Remove empty rows
train_df = train_df[train_df['text_cleaned'].str.len() > 0]
dev_df = dev_df[dev_df['text_cleaned'].str.len() > 0]
test_df = test_df[test_df['text_cleaned'].str.len() > 0]

print("After cleaning:", len(train_df), "training samples")

# ============================================================
# 4. ENCODE LABELS (Convert words → numbers)
# ============================================================
print("\nEncoding labels...")

label_encoder = LabelEncoder()

# Fit encoder on training labels only
label_encoder.fit(train_df['label'])

# Transform (convert)
train_df['label_encoded'] = label_encoder.transform(train_df['label'])
dev_df['label_encoded'] = label_encoder.transform(dev_df['label'])
test_df['label_encoded'] = label_encoder.transform(test_df['label'])

# Create mappings
label_to_idx = {}
idx_to_label = {}

for index, label in enumerate(label_encoder.classes_):
    label_to_idx[label] = index
    idx_to_label[index] = label

print("Number of classes:", len(label_to_idx))
print("Label mapping:")
for idx in sorted(idx_to_label.keys()):
    print("   ", idx, ":", idx_to_label[idx])

# ============================================================
# 5. SPLIT TRAINING DATA (Train / Validation)
# ============================================================
print("\nSplitting training data...")

X_train, X_val, y_train, y_val = train_test_split(
    train_df['text_cleaned'].values,
    train_df['label_encoded'].values,
    test_size=0.2,
    random_state=42,
    stratify=train_df['label_encoded']
)

print("Training:   ", len(X_train), "samples")
print("Validation: ", len(X_val), "samples")
print("Test:       ", len(test_df), "samples")

# ============================================================
# 6. SAVE PREPROCESSED DATA
# ============================================================
print("\nSaving preprocessed data...")

output_dir = Path("Data/Preprocessed")
output_dir.mkdir(exist_ok=True)

# Save training set
train_output = pd.DataFrame({
    'text': X_train,
    'label_encoded': y_train,
    'label': [idx_to_label[i] for i in y_train]
})
train_output.to_csv(output_dir / "train_preprocessed.csv", index=False)

# Save validation set
val_output = pd.DataFrame({
    'text': X_val,
    'label_encoded': y_val,
    'label': [idx_to_label[i] for i in y_val]
})
val_output.to_csv(output_dir / "val_preprocessed.csv", index=False)

# Save test set
test_output = pd.DataFrame({
    'text': test_df['text_cleaned'].values,
    'label_encoded': test_df['label_encoded'].values,
    'label': test_df['label'].values
})
test_output.to_csv(output_dir / "test_preprocessed.csv", index=False)

# Save label mappings as JSON
with open(output_dir / "label_mappings.json", "w") as f:
    json.dump({
        "label_to_idx": label_to_idx,
        "idx_to_label": {str(k): v for k, v in idx_to_label.items()}
    }, f, indent=2)

print("Saved files to:", output_dir)

# ============================================================
# 7. FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("PREPROCESSING SUMMARY")
print("=" * 70)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(test_df))
print("Number of classes:", len(label_to_idx))

print("\nLabel distribution (Training):")
print(train_output['label'].value_counts())

print("\n" + "=" * 70)
print("DONE! Ready to train models.")
print("=" * 70)
