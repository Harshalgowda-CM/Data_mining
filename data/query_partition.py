import pandas as pd
from pathlib import Path

folder = Path("object_store/store_id=S01/year=2024/month=10")

files = list(folder.glob("*.parquet"))

print("========================================")
print("PARTITION QUERY")
print("========================================")
print("Store: S01")
print("Year: 2024")
print("Month: October")
print("Parquet files read:", len(files))

total_rows = 0
total_revenue = 0

for file in files:
    df = pd.read_parquet(file)

    total_rows += len(df)
    total_revenue += df["revenue"].sum()

print("Rows read:", total_rows)
print("Total revenue:", round(total_revenue, 2))
print("Partition verified: S01 / 2024 / 10")