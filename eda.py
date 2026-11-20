
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# 6001CMD - Exploratory Data Analysis
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "bank-additional-full.csv"
FIGURE_DIR = BASE_DIR / "figures"

FIGURE_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH, sep=";")

sns.set_theme(style="whitegrid")

print("=" * 55)
print("6001CMD - EXPLORATORY DATA ANALYSIS")
print("=" * 55)

# 1. DESCRIPTIVE STATISTICS
print("\n1. DESCRIPTIVE STATISTICS")
print(df.describe(include="all").to_string())

# Save descriptive statistics
df.describe(include="all").to_csv(
    FIGURE_DIR / "descriptive_statistics.csv"
)

# 2. MISSING VALUES
print("\n2. MISSING VALUES")
print(df.isnull().sum())

# 3. UNKNOWN VALUES
print("\n3. UNKNOWN VALUES")
unknown_counts = (df == "unknown").sum()
print(unknown_counts[unknown_counts > 0])

# 4. DUPLICATE ROWS
print("\n4. DUPLICATE ROWS")
print("Duplicates:", df.duplicated().sum())

# 5. TARGET DISTRIBUTION
print("\n5. TARGET DISTRIBUTION")
print(df["y"].value_counts())
print(df["y"].value_counts(normalize=True) * 100)

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="y", order=["no", "yes"])
plt.title("Term Deposit Subscription Distribution")
plt.xlabel("Subscription")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "01_subscription_distribution.png",
    dpi=300
)
plt.close()

# 6. AGE DISTRIBUTION
plt.figure(figsize=(9, 5))
sns.histplot(data=df, x="age", bins=30, kde=True)
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "02_age_distribution.png",
    dpi=300
)
plt.close()

# 7. CORRELATION HEATMAP
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(12, 9))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Numerical Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "03_correlation_heatmap.png",
    dpi=300
)
plt.close()

# 8. OUTLIER ANALYSIS
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="age")
plt.title("Age Outlier Analysis")
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "04_age_outliers.png",
    dpi=300
)
plt.close()

plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="campaign")
plt.title("Campaign Contact Outlier Analysis")
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "05_campaign_outliers.png",
    dpi=300
)
plt.close()

# 9. JOB CATEGORY VS SUBSCRIPTION
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x="job", hue="y")
plt.title("Subscription Distribution by Job Category")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "06_subscription_by_job.png",
    dpi=300
)
plt.close()

# 10. SAVE EDA SUMMARY
summary = pd.DataFrame({
    "Metric": [
        "Total Records",
        "Total Columns",
        "Missing Values",
        "Unknown Values",
        "Duplicate Rows",
        "Subscribed Yes",
        "Subscribed No"
    ],
    "Value": [
        len(df),
        len(df.columns),
        int(df.isnull().sum().sum()),
        int((df == "unknown").sum().sum()),
        int(df.duplicated().sum()),
        int((df["y"] == "yes").sum()),
        int((df["y"] == "no").sum())
    ]
})

summary.to_csv(
    FIGURE_DIR / "eda_summary.csv",
    index=False
)

print("\nEDA SUMMARY")
print(summary.to_string(index=False))

print("\nEDA completed successfully!")
print("All graphs saved in the figures folder.")
