# Exploratory Data Analysis (EDA) Project
# Run with: python eda_analysis.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# =========================
# CONFIGURATION
# =========================
DATA_SOURCE = "titanic"  # Change to "csv" for your own file
CSV_PATH = "data/your_dataset.csv"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# STEP 1: LOAD DATA
# =========================
if DATA_SOURCE == "titanic":
    df = sns.load_dataset("titanic")
else:
    df = pd.read_csv(CSV_PATH)

print("\n===== STEP 1: INITIAL INSPECTION =====")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nColumn names:")
print(df.columns.tolist())
print("\nGeneral information:")
df.info()

# =========================
# STEP 2: SUMMARY STATISTICS
# =========================
print("\n===== STEP 2: SUMMARY STATISTICS =====")

numeric_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns

print("\nNumerical summary:")
print(df[numeric_cols].describe().T)

print("\nCategorical summary:")
if len(categorical_cols) > 0:
    print(df[categorical_cols].describe().T)
else:
    print("No categorical columns found.")

# =========================
# STEP 3: MISSING VALUES & DUPLICATES
# =========================
print("\n===== STEP 3: DATA QUALITY =====")

missing = df.isnull().sum().sort_values(ascending=False)
missing_percent = (df.isnull().mean() * 100).sort_values(ascending=False)

missing_table = pd.DataFrame({
    "missing_count": missing,
    "missing_percent": missing_percent.round(2)
})

print("\nMissing values:")
print(missing_table[missing_table["missing_count"] > 0])

print("\nDuplicate rows:", df.duplicated().sum())

# Make a cleaned copy.
df_clean = df.drop_duplicates().copy()

# Simple, general-purpose missing-value treatment:
# numerical -> median
# categorical -> mode
for col in numeric_cols:
    if df_clean[col].isnull().any():
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

for col in categorical_cols:
    if df_clean[col].isnull().any():
        mode = df_clean[col].mode(dropna=True)
        if len(mode) > 0:
            df_clean[col] = df_clean[col].fillna(mode.iloc[0])
        else:
            df_clean[col] = df_clean[col].fillna("Unknown")

print("\nCleaned shape:", df_clean.shape)
print("Remaining missing values:", int(df_clean.isnull().sum().sum()))

# =========================
# STEP 4: UNIVARIATE ANALYSIS
# =========================
print("\n===== STEP 4: UNIVARIATE ANALYSIS =====")

for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(df_clean[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/hist_{col}.png", dpi=150)
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df_clean[col])
    plt.title(f"Box Plot of {col}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/box_{col}.png", dpi=150)
    plt.show()
    plt.close()

for col in categorical_cols:
    if df_clean[col].nunique() <= 20:
        plt.figure(figsize=(9, 5))
        order = df_clean[col].value_counts().index
        sns.countplot(data=df_clean, x=col, order=order)
        plt.title(f"Category Counts: {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/count_{col}.png", dpi=150)
        plt.show()
        plt.close()

# =========================
# STEP 5: BIVARIATE & MULTIVARIATE
# =========================
print("\n===== STEP 5: RELATIONSHIPS =====")

if len(numeric_cols) >= 2:
    corr = df_clean[numeric_cols].corr()
    print("\nCorrelation matrix:")
    print(corr.round(2))

    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=150)
    plt.show()
    plt.close()

    sns.pairplot(df_clean[numeric_cols].dropna())
    plt.savefig(f"{OUTPUT_DIR}/pairplot.png", dpi=150)
    plt.show()
    plt.close()

# Example scatter plot using the first two numerical columns.
if len(numeric_cols) >= 2:
    x_col = numeric_cols[0]
    y_col = numeric_cols[1]
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df_clean, x=x_col, y=y_col)
    plt.title(f"{x_col} vs {y_col}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/scatter_{x_col}_vs_{y_col}.png", dpi=150)
    plt.show()
    plt.close()

# =========================
# STEP 6: AUTOMATIC INSIGHT HELPERS
# =========================
print("\n===== STEP 6: INSIGHT EXTRACTION =====")

print("\nNumerical variables:", list(numeric_cols))
print("Categorical variables:", list(categorical_cols))

if len(numeric_cols) >= 2:
    corr_pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    strongest = corr_pairs.stack().sort_values(key=np.abs, ascending=False)

    print("\nStrongest numerical correlations:")
    print(strongest.head(10).round(3))

print("\nTop category frequencies:")
for col in categorical_cols:
    if df_clean[col].nunique() <= 20:
        print(f"\n{col}:")
        print(df_clean[col].value_counts().head(10))

print("\nEDA completed. Check the 'outputs' folder for saved charts.")
