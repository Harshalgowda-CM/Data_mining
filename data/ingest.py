import pandas as pd
from pathlib import Path
import hashlib
import re

SALES_DIR = Path("sales")
OUTPUT_DIR = Path("object_store")

OUTPUT_DIR.mkdir(exist_ok=True)

all_rows = []

for file in SALES_DIR.glob("SALES_*.csv"):

    name = file.name

    # Extract store and business date from filename
    match = re.match(r"SALES_(S\d+)_(\d{8})", name)

    if not match:
        print(f"Skipping unexpected file: {name}")
        continue

    store_id = match.group(1)

    business_date = pd.to_datetime(
        match.group(2),
        format="%Y%m%d"
    ).date()

    # S01-S05: comma-separated CSV
    if store_id in ["S01", "S02", "S03", "S04", "S05"]:

        df = pd.read_csv(file)

        df = df.rename(columns={
            "product_code": "product_code",
            "qty": "qty",
            "unit_price": "unit_price",
            "line_type": "line_type",
            "ts": "ts"
        })

        df["ts"] = pd.to_datetime(df["ts"])

    # S06-S09: semicolon-separated CSV
    elif store_id in ["S06", "S07", "S08", "S09"]:

        df = pd.read_csv(file, sep=";")

        df = df.rename(columns={
            "item_code": "product_code",
            "quantity": "qty",
            "rate": "unit_price",
            "type": "line_type",
            "txn_time": "ts"
        })

        df["ts"] = pd.to_datetime(
            df["ts"],
            format="%d-%m-%Y %H:%M:%S"
        )

    # S10-S12: comma-separated CSV with epoch timestamp
    elif store_id in ["S10", "S11", "S12"]:

        df = pd.read_csv(file)

        df["ts"] = pd.to_datetime(
            df["ts"],
            unit="s",
            utc=True
        ).dt.tz_localize(None)

    else:
        print(f"Unknown store: {store_id}")
        continue

    # Add standard fields
    df["store_id"] = store_id
    df["business_date"] = pd.Timestamp(business_date)
    df["source_file"] = name

    # Identify resend files
    df["is_resend"] = "__R" in name

    required = [
        "store_id",
        "business_date",
        "bill_no",
        "line_no",
        "product_code",
        "qty",
        "unit_price",
        "line_type",
        "ts",
        "source_file",
        "is_resend"
    ]

    df = df[required]

    all_rows.append(df)

    print(f"Loaded {name}: {len(df):,} rows")


# Combine all files
print("\nCombining files...")

sales = pd.concat(all_rows, ignore_index=True)

print(f"Raw rows: {len(sales):,}")


# Numeric conversion
sales["qty"] = pd.to_numeric(sales["qty"])
sales["unit_price"] = pd.to_numeric(sales["unit_price"])


# Revenue definition
# SALE, RETURN, DISCOUNT and VOID affect revenue.
# TAX and TENDER do not.
sales["revenue"] = 0.0

revenue_types = [
    "SALE",
    "RETURN",
    "DISCOUNT",
    "VOID"
]

sales.loc[
    sales["line_type"].isin(revenue_types),
    "revenue"
] = sales["qty"] * sales["unit_price"]


# Remove duplicate logical lines caused by resends
# Safe key = bill_no + line_no
before = len(sales)

sales = (
    sales
    .sort_values(
        ["bill_no", "line_no", "source_file"]
    )
    .drop_duplicates(
        subset=["bill_no", "line_no"],
        keep="first"
    )
    .reset_index(drop=True)
)

after = len(sales)

print(
    f"Rows removed as duplicate/resend lines: "
    f"{before - after:,}"
)

print(f"Final rows: {after:,}")


# Create partition columns
sales["year"] = sales["business_date"].dt.year
sales["month"] = sales["business_date"].dt.month


# Write partitioned Parquet
print("\nWriting partitioned Parquet files...")

sales.to_parquet(
    OUTPUT_DIR,
    engine="pyarrow",
    partition_cols=[
        "store_id",
        "year",
        "month"
    ],
    index=False
)


# Deterministic checksum
check = sales.sort_values(
    [
        "store_id",
        "business_date",
        "bill_no",
        "line_no"
    ]
)

check_string = check[
    [
        "store_id",
        "business_date",
        "bill_no",
        "line_no",
        "product_code",
        "qty",
        "unit_price",
        "line_type",
        "revenue"
    ]
].to_csv(index=False)

checksum = hashlib.sha256(
    check_string.encode("utf-8")
).hexdigest()


print("\n========================================")
print("INGESTION COMPLETE")
print("========================================")
print(f"Final rows : {len(sales):,}")
print(f"Checksum   : {checksum}")
print(f"Output     : {OUTPUT_DIR.resolve()}")