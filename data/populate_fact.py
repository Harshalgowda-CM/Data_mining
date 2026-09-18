import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text

OBJECT_STORE = Path("object_store")

engine = create_engine(
    "postgresql+psycopg2://annapurna:annapurna@localhost:5432/annapurna"
)

files = list(OBJECT_STORE.rglob("*.parquet"))

print("Parquet files found:", len(files))

total = 0

with engine.begin() as conn:

    for file in files:

        df = pd.read_parquet(file)

        if df.empty:
            continue

        # Get store from partition path
        store_id = file.parts[-3].split("=")[1]

        df["store_id"] = store_id

        # Create business date
        df["business_date"] = pd.to_datetime(
            df["business_date"]
        ).dt.date

        # Insert rows into fact table
        for _, row in df.iterrows():

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
                    "business_date": row["business_date"],
                    "store_id": row["store_id"]
                }
            ).fetchone()

            if result is None:
                continue

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
                    "business_date": row["business_date"],
                    "store_sk": store_sk,
                    "product_sk": product_sk,
                    "category_sk": category_sk,
                    "qty": row["qty"],
                    "unit_price": row["unit_price"],
                    "revenue": row["revenue"],
                    "line_type": row["line_type"]
                }
            )

            total += 1

print()
print("========================================")
print("FACT TABLE LOAD COMPLETE")
print("========================================")
print("Rows processed:", total)

with engine.connect() as conn:
    count = conn.execute(
        text("SELECT COUNT(*) FROM fact_sales")
    ).scalar()

print("fact_sales rows:", count)