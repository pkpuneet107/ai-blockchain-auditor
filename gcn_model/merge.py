import os
import json

# Corrected folder-to-label mapping
label_map = {
    "reentrancy": 1,
    "timestamp": 2,
    "integeroverflow": 3  # ← fixed here
}

combined_train = []
combined_valid = []

for category, label in label_map.items():
    for split in ["train", "valid"]:
        path = f"train_data/{category}/{split}.json"
        if not os.path.exists(path):
            print(f"⚠️ Skipping missing file: {path}")
            continue

        with open(path, "r") as f:
            samples = json.load(f)

        for sample in samples:
            sample["vuln_type"] = label
            if split == "train":
                combined_train.append(sample)
            else:
                combined_valid.append(sample)

# Create output directory
os.makedirs("train_data/combined", exist_ok=True)

with open("train_data/combined/train.json", "w") as f:
    json.dump(combined_train, f, indent=2)

with open("train_data/combined/valid.json", "w") as f:
    json.dump(combined_valid, f, indent=2)

print(f"✅ Merged {len(combined_train)} training and {len(combined_valid)} validation samples.")
