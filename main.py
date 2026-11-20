
import pandas as pd
from pathlib import Path

# ==========================================
# 6001CMD Machine Learning Group Project
# Step 2: Dataset Loading and Verification
# ==========================================

# Locate the dataset
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "bank-additional-full.csv"

# Load the CSV file
df = pd.read_csv(DATA_PATH, sep=";")

# Display project information
print("=" * 55)
print("6001CMD MACHINE LEARNING GROUP PROJECT")
print("Bank Term Deposit Subscription Prediction")
print("=" * 55)

# Check dataset dimensions
print("\n1. DATASET SHAPE")
print(df.shape)

# Display the first five rows
print("\n2. FIRST FIVE ROWS")
print(df.head().to_string())

# Display column names
print("\n3. COLUMN NAMES")
print(df.columns.tolist())

# Display target distribution
print("\n4. TARGET DISTRIBUTION")
print(df["y"].value_counts())

# Check missing values
print("\n5. MISSING VALUES")
print(df.isnull().sum())

print("\nDataset loaded successfully!")
