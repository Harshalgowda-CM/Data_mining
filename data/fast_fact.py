import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text

OBJECT_STORE = Path("object_store")

engine = create_engine(
    "postgresql+psycopg2://annapurna:annapurna@localhost:5432/annapurna"
)

print("Reading Parquet files...")

files = list(OBJECT_STORE.rglob("*.parquet"))

print("Parquet files:", len(files))

frames = []

for i, file in enumerate(files, 1):
    df = pd.read_parquet(file)

    if not df.empty:
        store_id = file.parts[-3].split("=")[1]
        df["store_id"] = store_id
        frames.append(df)

    if i % 100 == 0:
        print("Read:", i, "files")

print("Combining data...")

sales = pd.concat(frames, ignore_index=True)

print("Total source rows:", len(sales))

# Remove duplicate logical lines
sales = sales.drop_duplicates(
    subset=["bill_no", "line_no"]
)

print("After deduplication:", len(sales))

# Connect to PostgreSQL
with engine.begin() as conn:

    print("Loading fact_sales...")

    conn.execute(text("TRUNCATE fact_sales"))

    # Process in chunks
    chunk_size = 10000

    for start in range(0, len(sales), chunk_size):

        chunk = sales.iloc[start:start + chunk_size].copy()

        for _, row in chunk.iterrows():

            result = conn.execute(
                text("""
                    SELECT
                        ds.store_sk,
                        dp.product_sk,
                        dc.category_sk
                    FROM dim_store ds
                    LEFT JOIN dim_product dp
                      ON dp.product_code = :product_code
                     AND :business_date >= dp.valid_from
                     AND (
                         dp.valid_to IS NULL
                         OR :business_date < dp.valid_to
                     )
                    LEFT JOIN dim_category dc
                      ON dc.category_id = dp.category_id
                    WHERE ds.store_id = :store_id
                """),
                {
                    "product_code": row["product_code"],
                    "business_date": pd.to_datetime(
                        row["business_date"]
                    ).date(),
                    "store_id": row["store_id"]
                }
            ).fetchone()

            if result:

                store_sk, product_sk, category_sk = result

                conn.execute(
                    text("""
                        INSERT INTO fact_sales
                        (
                            bill_no,
                            line_no,
                            business_date,
                            store_sk,
                            product_sk,
                            category_sk,
                            qty,
                            unit_price,
                            revenue,
                            line_type
                        )
                        VALUES
                        (
                            :bill_no,
                            :line_no,
                            :business_date,
                            :store_sk,
                            :product_sk,
                            :category_sk,
                            :qty,
                            :unit_price,
                            :revenue,
                            :line_type
                        )
                        ON CONFLICT (bill_no, line_no)
                        DO NOTHING
                    """),
                    {
                        "bill_no": row["bill_no"],
                        "line_no": int(row["line_no"]),
                        "business_date": pd.to_datetime(
                            row["business_date"]
                        ).date(),
                        "store_sk": store_sk,
                        "product_sk": product_sk,
                        "category_sk": category_sk,
                        "qty": row["qty"],
                        "unit_price": row["unit_price"],
                        "revenue": row["revenue"],
                        "line_type": row["line_type"]
                    }
                )

        print(
            "Loaded:",
            min(start + chunk_size, len(sales)),
            "/",
            len(sales)
        )

with engine.connect() as conn:

    count = conn.execute(
        text("SELECT COUNT(*) FROM fact_sales")
    ).scalar()

print()
print("========================================")
print("STAR SCHEMA LOAD COMPLETE")
print("========================================")
print("fact_sales rows:", count)
print("========================================")