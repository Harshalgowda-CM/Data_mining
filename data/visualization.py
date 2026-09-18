import duckdb
import matplotlib.pyplot as plt
import os

# Create visualization folder
os.makedirs("visualizations", exist_ok=True)

con = duckdb.connect()

# ============================================================
# 1. MONTHLY REVENUE
# ============================================================

monthly = con.execute("""
SELECT
    STRFTIME(business_date, '%Y-%m') AS month,
    ROUND(SUM(revenue), 2) AS revenue
FROM read_parquet(
    'object_store/store_id=*/year=*/month=*/*.parquet',
    hive_partitioning=true
)
GROUP BY 1
ORDER BY 1
""").fetchall()

months = [row[0] for row in monthly]
monthly_revenue = [row[1] for row in monthly]

plt.figure(figsize=(12, 6))
plt.plot(months, monthly_revenue, marker="o")
plt.title("Monthly Revenue - 2024")
plt.xlabel("Month")
plt.ylabel("Revenue (INR)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/monthly_revenue.png", dpi=300)
plt.close()

# ============================================================
# 2. REVENUE BY STORE
# ============================================================

store = con.execute("""
SELECT
    store_id,
    ROUND(SUM(revenue), 2) AS revenue
FROM read_parquet(
    'object_store/store_id=*/year=*/month=*/*.parquet',
    hive_partitioning=true
)
GROUP BY store_id
ORDER BY store_id
""").fetchall()

stores = [row[0] for row in store]
store_revenue = [row[1] for row in store]

plt.figure(figsize=(12, 6))
plt.bar(stores, store_revenue)
plt.title("Revenue by Store - 2024")
plt.xlabel("Store")
plt.ylabel("Revenue (INR)")
plt.tight_layout()
plt.savefig("visualizations/revenue_by_store.png", dpi=300)
plt.close()

# ============================================================
# 3. REVENUE BY CATEGORY
# ============================================================

con.execute("INSTALL postgres;")
con.execute("LOAD postgres;")

con.execute("""
ATTACH 'host=localhost port=5432 dbname=annapurna user=annapurna password=annapurna'
AS pg (TYPE POSTGRES, READ_ONLY)
""")

category = con.execute("""
SELECT
    c.category_name,
    ROUND(SUM(s.revenue), 2) AS revenue
FROM read_parquet(
    'object_store/store_id=*/year=*/month=*/*.parquet',
    hive_partitioning=true
) s
JOIN pg.public.products p
    ON s.product_code = p.product_code
   AND s.business_date >= p.valid_from
   AND s.business_date <= p.valid_to
JOIN pg.public.product_categories c
    ON p.category_id = c.category_id
GROUP BY c.category_name
ORDER BY revenue DESC
""").fetchall()

categories = [row[0] for row in category]
category_revenue = [row[1] for row in category]

plt.figure(figsize=(12, 7))
plt.barh(categories, category_revenue)
plt.title("Revenue by Category - 2024")
plt.xlabel("Revenue (INR)")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig("visualizations/revenue_by_category.png", dpi=300)
plt.close()

con.close()

print("=" * 70)
print("VISUALIZATION FILES CREATED")
print("=" * 70)
print("1. visualizations/monthly_revenue.png")
print("2. visualizations/revenue_by_store.png")
print("3. visualizations/revenue_by_category.png")
print("=" * 70)