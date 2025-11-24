"""
STEP 1.3: PREPARE COMBINED DATASET
This script loads CrisisNLP and Kaggle datasets,
cleans them, maps labels, combines them, and saves the final dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Print step header
print("=" * 70)
print("STEP 1.3: PREPARING COMBINED DATASET")
print("=" * 70)

# ============================================================
# 1. LOAD DATASETS
# ============================================================
print("\nLoading datasets...")

# Load CrisisNLP CSV files
crisisnlp_train = pd.read_csv("Data/CrisisNLP/sample_prccd_train.csv")
crisisnlp_dev = pd.read_csv("Data/CrisisNLP/sample_prccd_dev.csv")
crisisnlp_test = pd.read_csv("Data/CrisisNLP/sample_prccd_test.csv")

# Print dataset sizes
print("CrisisNLP loaded:")
print("   Train:", len(crisisnlp_train), "samples")
print("   Dev:  ", len(crisisnlp_dev), "samples")
print("   Test: ", len(crisisnlp_test), "samples")

# Load Kaggle CSV files
kaggle_train = pd.read_csv("Data/Kaggle/train.csv")
kaggle_test = pd.read_csv("Data/Kaggle/test.csv")

print("\nKaggle loaded:")
print("   Train:", len(kaggle_train), "samples")
print("   Test: ", len(kaggle_test), "samples (no labels)")

# ============================================================
# 2. PREPARE CRISISNLP DATA
# ============================================================
print("\nPreparing CrisisNLP dataset...")

# CrisisNLP contains: item_id, item, label
# We want only: text, label, source

# Clean CrisisNLP TRAIN
crisisnlp_train_clean = pd.DataFrame()
crisisnlp_train_clean['text'] = crisisnlp_train['item']
crisisnlp_train_clean['label'] = crisisnlp_train['label']
crisisnlp_train_clean['source'] = 'CrisisNLP'

# Clean CrisisNLP DEV
crisisnlp_dev_clean = pd.DataFrame()
crisisnlp_dev_clean['text'] = crisisnlp_dev['item']
crisisnlp_dev_clean['label'] = crisisnlp_dev['label']
crisisnlp_dev_clean['source'] = 'CrisisNLP'

# Clean CrisisNLP TEST
crisisnlp_test_clean = pd.DataFrame()
crisisnlp_test_clean['text'] = crisisnlp_test['item']
crisisnlp_test_clean['label'] = crisisnlp_test['label']
crisisnlp_test_clean['source'] = 'CrisisNLP'

# Print label statistics
unique_labels_crisis = crisisnlp_train['label'].unique()
print("   CrisisNLP has", len(unique_labels_crisis), "unique labels")
print("   Labels:", list(unique_labels_crisis))

# ============================================================
# 3. PREPARE KAGGLE DATA
# ============================================================
print("\nPreparing Kaggle dataset...")

# Kaggle labels are 0 (not disaster) and 1 (disaster)
# Convert numbers to readable names

kaggle_train_clean = pd.DataFrame()
kaggle_train_clean['text'] = kaggle_train['text']
kaggle_train_clean['label'] = kaggle_train['target'].map({
    0: 'Not disaster',
    1: 'Disaster related'
})
kaggle_train_clean['source'] = 'Kaggle'

# Print label counts
print("   Kaggle has 2 labels: Not disaster, Disaster related")
print("   Distribution:")
print(kaggle_train_clean['label'].value_counts())

# ============================================================
# 4. COMBINE DATASETS
# ============================================================
print("\n" + "=" * 70)
print("COMBINING DATASETS")
print("=" * 70)

# ---------------------- OPTION 1 ----------------------------
print("\nOPTION 1: Keep all labels separate")

# Combine CrisisNLP + Kaggle as they are
combined_train_v1 = pd.concat(
    [crisisnlp_train_clean, kaggle_train_clean],
    ignore_index=True
)

# Dev and Test remain only from CrisisNLP
combined_dev_v1 = crisisnlp_dev_clean.copy()
combined_test_v1 = crisisnlp_test_clean.copy()

# Print stats for Option 1
print("   Combined Training:", len(combined_train_v1), "samples")
print("   Total unique labels:",
      len(combined_train_v1['label'].unique()))
print("   Labels:",
      list(combined_train_v1['label'].unique()))

# ---------------------- OPTION 2 ----------------------------
print("\nOPTION 2: Map Kaggle to CrisisNLP labels")

# Copy the Kaggle dataset
kaggle_train_mapped = kaggle_train_clean.copy()

# Map Kaggle to CrisisNLP-like labels
kaggle_train_mapped['label'] = kaggle_train_mapped['label'].map({
    'Disaster related': 'Other Useful Information',
    'Not disaster': 'Not related or irrelevant'
})

# Combine CrisisNLP and mapped Kaggle
combined_train_v2 = pd.concat(
    [crisisnlp_train_clean, kaggle_train_mapped],
    ignore_index=True
)

# Print stats for Option 2
print("   Combined Training:", len(combined_train_v2), "samples")
print("   Total unique labels:",
      len(combined_train_v2['label'].unique()))
print("   Labels:",
      list(combined_train_v2['label'].unique()))

print("\n   Label distribution:")
print(combined_train_v2['label'].value_counts())

# ============================================================
# 5. SAVE FINAL DATASETS
# ============================================================
print("\n" + "=" * 70)
print("SAVING COMBINED DATASET")
print("=" * 70)

output_dir = Path("Data/Combined")

# Create folder if not exists
output_dir.mkdir(exist_ok=True)

# Save CSV files
combined_train_v2.to_csv(output_dir / "combined_train.csv",
                         index=False)
combined_dev_v1.to_csv(output_dir / "combined_dev.csv",
                       index=False)
combined_test_v1.to_csv(output_dir / "combined_test.csv",
                        index=False)

print("\nSaved combined dataset to", output_dir, "/")
print("   - combined_train.csv (", len(combined_train_v2), "samples )")
print("   - combined_dev.csv   (", len(combined_dev_v1), "samples )")
print("   - combined_test.csv  (", len(combined_test_v1), "samples )")

# ============================================================
# 6. FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("DATASET STATISTICS:\n")

print("Training Set:")
print("  - Total samples:", len(combined_train_v2))
print("  - From CrisisNLP:", len(crisisnlp_train_clean))
print("  - From Kaggle:   ", len(kaggle_train_clean))
print("  - Unique labels: ", len(combined_train_v2['label'].unique()))
print()

print("Dev Set:")
print("  - Total samples:", len(combined_dev_v1))
print("  - From CrisisNLP only")
print()

print("Test Set:")
print("  - Total samples:", len(combined_test_v1))
print("  - From CrisisNLP only")
print()

print("LABELS:")
for label in sorted(combined_train_v2['label'].unique()):
    print("  -", label)

print("\nNEXT STEP:")
print("Ready to preprocess and train models!")

print("=" * 70)
