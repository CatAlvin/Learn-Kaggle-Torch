from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# D:\Programming\Python\python_workspace\Learn-Kaggle-Torch
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "titanic"
OUTPUT_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "titanic"
OUTPUT_LOG_DIR = PROJECT_ROOT / "outputs" / "logs" / "titanic"
OUTPUT_SUBMISSION_DIR = PROJECT_ROOT / "outputs" / "submissions" / "titanic"

# read data
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
sample_submission_df = pd.read_csv(DATA_DIR / "gender_submission.csv")

# predict by rule-based baseline model
predicted_survived = np.array(test_df["Sex"] == "female", dtype="int64")

# create df
submission_df = pd.DataFrame()
submission_df["PassengerId"] = test_df["PassengerId"]
submission_df["Survived"] = predicted_survived
print(f"Submission Shape: {submission_df.shape}")
print("Submission Columns:")
print(submission_df.columns)
print("Submission Head:")
print(submission_df.head())
print("Predicted Survived Ratio:")
print(submission_df["Survived"].value_counts(normalize=True))
print(f"Sample Submission Shape: {sample_submission_df.shape}")
print("Submission Columns:")
print(sample_submission_df.columns)

# save csv
OUTPUT_SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
submission_df.to_csv(OUTPUT_SUBMISSION_DIR / "rule_gender_submission.csv", index=False)