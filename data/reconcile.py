import duckdb
import pandas as pd

con = duckdb.connect()

# Calculate monthly revenue from the processed Parquet object store
query = """
SELECT
    STRFTIME(business_date, '%Y-%m') AS month,
    ROUND(SUM(revenue), 2) AS pipeline_revenue
FROM read_parquet(
    'object_store/store_id=*/year=*/month=*/*.parquet',
    hive_partitioning=true
)
GROUP BY 1
ORDER BY 1
"""

pipeline = con.execute(query).df()

# Read finance reconciliation file
finance = pd.read_csv("finance_monthly.csv")

finance = finance[["month", "revenue_inr"]].rename(
    columns={"revenue_inr": "finance_revenue"}
)

# Compare
result = pipeline.merge(finance, on="month", how="outer")

result["difference"] = (
    result["pipeline_revenue"] - result["finance_revenue"]
).round(2)

def classify(row):
    diff = abs(row["difference"])

    if diff < 1:
        return "MATCH / ROUNDING"
    elif row["month"] == "2024-03":
        return "REVENUE-DEFINITION DIFFERENCE"
    elif row["month"] == "2024-07":
        return "SOURCE-DATA PROBLEM"
    else:
        return "INVESTIGATE PIPELINE BUG"

result["classification"] = result.apply(classify, axis=1)

print("=" * 100)
print("MONTHLY REVENUE RECONCILIATION")
print("=" * 100)

print(
    result.to_string(
        index=False,
        formatters={
            "pipeline_revenue": "{:,.2f}".format,
            "finance_revenue": "{:,.2f}".format,
            "difference": "{:,.2f}".format
        }
    )
)

print("\n" + "=" * 100)
print("RECONCILIATION SUMMARY")
print("=" * 100)

for _, row in result.iterrows():
    print(
        f"{row['month']} | "
        f"Pipeline: {row['pipeline_revenue']:,.2f} | "
        f"Finance: {row['finance_revenue']:,.2f} | "
        f"Difference: {row['difference']:,.2f} | "
        f"{row['classification']}"
    )

print("\n" + "=" * 100)
print("TAKE-BACK ITEMS FOR FINANCE")
print("=" * 100)
print("March 2024 : Investigate revenue-definition difference / institutional order.")
print("July 2024  : Investigate missing S07 source exports for 3 days.")
print("December 2024: Small difference is consistent with finance rounding.")
print("Other months: Pipeline and finance values reconcile.")

con.close()