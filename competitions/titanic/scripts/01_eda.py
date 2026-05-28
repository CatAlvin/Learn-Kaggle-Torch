from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# D:\Programming\Python\python_workspace\Learn-Kaggle-Torch
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "titanic"
OUTPUT_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "titanic"
OUTPUT_LOG_DIR = PROJECT_ROOT / "outputs" / "logs" / "titanic"

# read data
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
sample_submission_df = pd.read_csv(DATA_DIR / "gender_submission.csv")

# see shape
print("train shape:", train_df.shape)
print("test shape:", test_df.shape)
print("sample submission shape:", sample_submission_df.shape)
print()

# see columns
print("train columns:", train_df.columns)
print("test columns:", test_df.columns)
print("sample submission columns:", sample_submission_df.columns)
print("train vs. test column(s):", np.setdiff1d(train_df.columns, test_df.columns))
print()

# see NA
print("train NA count:", train_df.isna().sum())
print("test NA count:", test_df.isna().sum())
print()

# see target distribution
print("Survived count:", train_df["Survived"].value_counts())
print("Survived ratio:", train_df["Survived"].value_counts(normalize=True))
print()

# group analyze
print("Group by Sex:")
print(train_df.groupby("Sex")["Survived"].mean())
print()
print("Group by Pclass:")
print(train_df.groupby("Pclass")["Survived"].mean())
print()
print("Group by Embarked:")
print(train_df.groupby("Embarked")["Survived"].mean())
print()
# Because Here class is 0 and 1, mean() represent survive rate

# Sex vs. Survival Rate
sex_survival_rate = train_df.groupby("Sex")["Survived"].mean()
OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

plt.figure()
plt.bar(sex_survival_rate.index, sex_survival_rate.values)
plt.xlabel("Sex")
plt.ylabel("Survival Rate")
plt.title("Survival Rate by Sex")
plt.ylim(0, 1)

plt.savefig(OUTPUT_FIGURE_DIR / "sex_survival_rate.png", dpi=150, bbox_inches="tight")
plt.close()

# save important result
OUTPUT_LOG_DIR.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_LOG_DIR / "eda_summary.txt", "w", encoding="utf-8") as f:
    f.write(
        f"train shape: {train_df.shape}\n\n"
        f"test shape: {test_df.shape}\n\n"
        f"train NA count:\n"
        f"{train_df.isna().sum()}\n\n"
        f"test NA count:\n"
        f"{test_df.isna().sum()}\n\n"
        f"survival distribution:\n"
        f"{train_df["Survived"].value_counts(normalize=True)}\n\n"
        f"survival rate by sex:\n"
        f"{train_df.groupby("Sex")["Survived"].mean()}\n\n"
        f"survival rate by Pclass:\n"
        f"{train_df.groupby("Pclass")["Survived"].mean()}\n\n"
        f"survival rate by Embarked:\n"
        f"{train_df.groupby("Embarked")["Survived"].mean()}\n\n"
    )