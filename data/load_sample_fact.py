import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text

engine = create_engine(
    "postgresql+psycopg2://annapurna:annapurna@localhost:5432/annapurna"
)

files = list(Path("object_store").rglob("*.parquet"))

print("Reading Parquet files...")

frames = []

for f in files:
    df = pd.read_parquet(f)

    if not df.empty:
        # Correct partition extraction:
        # object_store/store_id=S12/year=2024/month=1/file.parquet
        store_id = f.parts[-4].split("=")[1]

        df["store_id"] = store_id
        frames.append(df)

sales = pd.concat(frames, ignore_index=True)

sales = sales.drop_duplicates(
    subset=["bill_no", "line_no"]
)

# Use 10,000 rows for the lab demonstration
sales = sales.head(10000)

print("Sample rows:", len(sales))

with engine.begin() as conn:

    conn.execute(text("TRUNCATE fact_sales"))

    inserted = 0

    for _, r in sales.iterrows():

        business_date = pd.to_datetime(
            r["business_date"]
        ).date()

        result = conn.execute(
            text("""
                SELECT
                    ds.store_sk,
                    dp.product_sk,
                    dc.category_sk
                FROM dim_store ds
                JOIN dim_product dp
                  ON dp.product_code = :product_code
                 AND :business_date >= dp.valid_from
                 AND :business_date <= dp.valid_to
                JOIN dim_category dc
                  ON dc.category_id = dp.category_id
                WHERE ds.store_id = :store_id
            """),
            {
                "product_code": str(r["product_code"]),
                "business_date": business_date,
                "store_id": str(r["store_id"])
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
                    "bill_no": str(r["bill_no"]),
                    "line_no": int(r["line_no"]),
                    "business_date": business_date,
                    "store_sk": store_sk,
                    "product_sk": product_sk,
                    "category_sk": category_sk,
                    "qty": r["qty"],
                    "unit_price": r["unit_price"],
                    "revenue": r["revenue"],
                    "line_type": str(r["line_type"])
                }
            )

            inserted += 1

print()
print("========================================")
print("SAMPLE STAR FACT LOAD COMPLETE")
print("========================================")

with engine.connect() as conn:
    count = conn.execute(
        text("SELECT COUNT(*) FROM fact_sales")
    ).scalar()

print("fact_sales rows:", count)
print("Rows successfully linked:", inserted)
print("========================================")