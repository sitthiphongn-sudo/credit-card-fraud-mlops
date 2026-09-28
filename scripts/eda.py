from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path("data/raw/creditcard.csv")
REPORT_DIR = Path("reports")
FIGURE_DIR = REPORT_DIR / "figures"


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    total_rows = len(df)
    fraud_rows = int(df["Class"].sum())
    fraud_ratio = fraud_rows / total_rows
    duplicate_rows = int(df.duplicated().sum())
    zero_amount_rows = int((df["Amount"] == 0).sum())

    print(f"Total rows: {total_rows:,}")
    print(f"Fraud rows: {fraud_rows:,}")
    print(f"Fraud ratio: {fraud_ratio:.6%}")
    print(f"Duplicate rows: {duplicate_rows:,}")
    print(f"Amount = 0 rows: {zero_amount_rows:,}")

    # -----------------------------
    # 1. Amount distribution
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(df["Amount"], bins=100)
    plt.xlabel("Amount")
    plt.ylabel("Count")
    plt.title("Distribution of Transaction Amount")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "amount_distribution.png")
    plt.close()

    # -----------------------------
    # 2. Time distribution
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(df["Time"], bins=100)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Count")
    plt.title("Distribution of Transaction Time")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "time_distribution.png")
    plt.close()

    # -----------------------------
    # 3. Fraud ratio by hour
    # -----------------------------
    df["Hour"] = ((df["Time"] // 3600) % 24).astype(int)

    fraud_by_hour = (
        df.groupby("Hour")["Class"]
        .mean()
        .reindex(range(24), fill_value=0)
    )

    plt.figure(figsize=(9, 5))
    plt.bar(fraud_by_hour.index, fraud_by_hour.values)
    plt.xlabel("Hour of day")
    plt.ylabel("Fraud ratio")
    plt.title("Fraud Ratio by Hour")
    plt.xticks(range(24))
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fraud_ratio_by_hour.png")
    plt.close()

    # -----------------------------
    # 4. Fraud count by class
    # -----------------------------
    class_counts = df["Class"].value_counts().sort_index()

    plt.figure(figsize=(6, 5))
    plt.bar(["Normal", "Fraud"], class_counts.values)
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "class_distribution.png")
    plt.close()

    report = f"""# Exploratory Data Analysis

## Dataset summary

- Total rows: {total_rows:,}
- Fraud rows: {fraud_rows:,}
- Fraud ratio: {fraud_ratio:.6%}
- Duplicate rows: {duplicate_rows:,}
- Rows with Amount = 0: {zero_amount_rows:,}

## Observations

### Class imbalance

The dataset is highly imbalanced. Fraud transactions represent only
{fraud_ratio:.6%} of all transactions.

### Duplicate rows

There are {duplicate_rows:,} duplicated rows in the complete dataset.

For the project, duplicated rows will be removed only from the training
set to reduce the chance that the model memorizes repeated transactions.
Duplicates will remain in validation and test data so those sets better
represent incoming real-world data.

### Amount

The transaction amount distribution is strongly right-skewed. Most
transactions have relatively small amounts, while a small number of
transactions have very large values.

### Time

The Time column records elapsed seconds. Transaction activity is not
uniform across time.

### Fraud by hour

Fraud ratio varies by hour. The hourly ratio is calculated as the number
of fraudulent transactions divided by the total number of transactions
within each hour.

## Figures

![Amount distribution](figures/amount_distribution.png)

![Time distribution](figures/time_distribution.png)

![Fraud ratio by hour](figures/fraud_ratio_by_hour.png)

![Class distribution](figures/class_distribution.png)
"""

    (REPORT_DIR / "eda.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()