-- Q1(c) STAR SCHEMA

DROP TABLE IF EXISTS fact_sales CASCADE;
DROP TABLE IF EXISTS dim_product CASCADE;
DROP TABLE IF EXISTS dim_category CASCADE;
DROP TABLE IF EXISTS dim_store CASCADE;
DROP TABLE IF EXISTS dim_date CASCADE;

-- Date Dimension
CREATE TABLE dim_date (
    date_sk INTEGER PRIMARY KEY,
    full_date DATE UNIQUE NOT NULL,
    day_of_week INTEGER NOT NULL,
    day_name TEXT NOT NULL,
    month INTEGER NOT NULL,
    month_name TEXT NOT NULL,
    year INTEGER NOT NULL
);

-- Store Dimension
CREATE TABLE dim_store (
    store_sk SERIAL PRIMARY KEY,
    store_id TEXT UNIQUE NOT NULL,
    store_name TEXT NOT NULL,
    city TEXT NOT NULL,
    state TEXT NOT NULL,
    region TEXT NOT NULL,
    floor_area_sqft INTEGER,
    opened_on DATE
);

-- Category Dimension
CREATE TABLE dim_category (
    category_sk SERIAL PRIMARY KEY,
    category_id TEXT UNIQUE NOT NULL,
    category_name TEXT NOT NULL,
    department TEXT NOT NULL,
    gst_rate NUMERIC(4,3)
);

-- Product Dimension
CREATE TABLE dim_product (
    product_sk INTEGER PRIMARY KEY,
    product_code TEXT NOT NULL,
    product_name TEXT NOT NULL,
    category_id TEXT NOT NULL,
    valid_from DATE NOT NULL,
    valid_to DATE
);

-- Fact table
CREATE TABLE fact_sales (
    sales_sk BIGSERIAL PRIMARY KEY,
    bill_no TEXT NOT NULL,
    line_no INTEGER NOT NULL,
    business_date DATE NOT NULL,
    store_sk INTEGER NOT NULL REFERENCES dim_store(store_sk),
    product_sk INTEGER REFERENCES dim_product(product_sk),
    category_sk INTEGER REFERENCES dim_category(category_sk),
    qty NUMERIC,
    unit_price NUMERIC,
    revenue NUMERIC,
    line_type TEXT,
    UNIQUE(bill_no, line_no)
);

-- Populate date dimension
INSERT INTO dim_date
SELECT
    TO_CHAR(d, 'YYYYMMDD')::INTEGER,
    d,
    EXTRACT(ISODOW FROM d)::INTEGER,
    TO_CHAR(d, 'Day'),
    EXTRACT(MONTH FROM d)::INTEGER,
    TO_CHAR(d, 'Month'),
    EXTRACT(YEAR FROM d)::INTEGER
FROM generate_series(
    DATE '2024-01-01',
    DATE '2024-12-31',
    INTERVAL '1 day'
) AS x(d);

-- Populate store dimension
INSERT INTO dim_store
(store_id, store_name, city, state, region, floor_area_sqft, opened_on)
SELECT
    store_id,
    store_name,
    city,
    state,
    region,
    floor_area_sqft,
    opened_on
FROM stores;

-- Populate category dimension
INSERT INTO dim_category
(category_id, category_name, department, gst_rate)
SELECT
    category_id,
    category_name,
    department,
    gst_rate
FROM product_categories;

-- Populate product dimension
INSERT INTO dim_product
(product_sk, product_code, product_name, category_id, valid_from, valid_to)
SELECT
    product_sk,
    product_code,
    product_name,
    category_id,
    valid_from,
    valid_to
FROM products;

-- Display row counts
SELECT 'dim_date' AS table_name, COUNT(*) AS rows FROM dim_date
UNION ALL
SELECT 'dim_store', COUNT(*) FROM dim_store
UNION ALL
SELECT 'dim_category', COUNT(*) FROM dim_category
UNION ALL
SELECT 'dim_product', COUNT(*) FROM dim_product
UNION ALL
SELECT 'fact_sales', COUNT(*) FROM fact_sales;