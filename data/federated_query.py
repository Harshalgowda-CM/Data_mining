import duckdb

con = duckdb.connect()

# Load PostgreSQL extension
con.execute("INSTALL postgres;")
con.execute("LOAD postgres;")

# Attach PostgreSQL database
con.execute("""
ATTACH 'host=localhost port=5432 dbname=annapurna user=annapurna password=annapurna'
AS pg (TYPE POSTGRES, READ_ONLY)
""")

print("=" * 70)
print("FEDERATED QUERY - DUCKDB + POSTGRESQL + PARQUET")
print("=" * 70)

query = """
SELECT
    s.store_id,
    st.store_name,
    st.city,
    c.category_name,
    ROUND(SUM(s.revenue), 2) AS total_revenue
FROM read_parquet(
    'object_store/store_id=*/year=2024/month=10/*.parquet',
    hive_partitioning=true
) s
JOIN pg.public.stores st
    ON s.store_id = st.store_id
JOIN pg.public.products p
    ON s.product_code = p.product_code
   AND s.business_date >= p.valid_from
   AND s.business_date <= p.valid_to
JOIN pg.public.product_categories c
    ON p.category_id = c.category_id
GROUP BY
    s.store_id,
    st.store_name,
    st.city,
    c.category_name
ORDER BY
    s.store_id,
    c.category_name
LIMIT 30
"""

result = con.execute(query).fetchall()

print("\nQUERY RESULT:")
print("-" * 70)

for row in result:
    print(row)

print("\n" + "=" * 70)
print("ENGINE EVIDENCE - QUERY PLAN")
print("=" * 70)

plan = con.execute("EXPLAIN " + query).fetchall()

for row in plan:
    print(row[1])

print("\n" + "=" * 70)
print("Federated query completed successfully.")
print("Sales source : Parquet object store")
print("Master source: PostgreSQL")
print("Query engine : DuckDB")
print("=" * 70)

con.close()