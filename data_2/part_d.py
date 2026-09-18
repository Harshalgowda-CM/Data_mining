import pandas as pd
import glob
import os
import sqlite3
import time


# ============================================================
# QUESTION 2(d) - PERSISTENT DATABASE AND INDEXING
# ============================================================

DATA_PATH = "."
DB_NAME = "tender_retrieval.db"


print("=" * 70)
print("QUESTION 2(d) - DATABASE STORAGE AND INDEXING")
print("=" * 70)


# ------------------------------------------------------------
# 1. LOAD ALL NOTICE FILES
# ------------------------------------------------------------

notice_files = glob.glob(
    os.path.join(DATA_PATH, "notices", "*.csv")
)

notice_list = []

for file in notice_files:
    notice_list.append(pd.read_csv(file))

notices = pd.concat(
    notice_list,
    ignore_index=True
)

print("\nTotal notices loaded:", len(notices))


# ------------------------------------------------------------
# 2. CREATE SQLITE DATABASE
# ------------------------------------------------------------

if os.path.exists(DB_NAME):
    os.remove(DB_NAME)

conn = sqlite3.connect(DB_NAME)

print("\nSQLite database created:", DB_NAME)


# ------------------------------------------------------------
# 3. STORE NOTICES IN DATABASE
# ------------------------------------------------------------

start_insert = time.perf_counter()

notices[
    [
        "notice_id",
        "portal_id",
        "published_at",
        "title",
        "body",
        "estimated_value",
        "closing_date"
    ]
].to_sql(
    "notices",
    conn,
    if_exists="replace",
    index=False
)

insert_time = time.perf_counter() - start_insert

print(
    "Database insertion time:",
    f"{insert_time:.4f}",
    "seconds"
)


# ------------------------------------------------------------
# 4. CREATE INDEXES
# ------------------------------------------------------------

start_index = time.perf_counter()

cursor = conn.cursor()

cursor.execute("""
CREATE UNIQUE INDEX idx_notice_id
ON notices(notice_id)
""")

cursor.execute("""
CREATE INDEX idx_portal_id
ON notices(portal_id)
""")

cursor.execute("""
CREATE INDEX idx_published_at
ON notices(published_at)
""")

cursor.execute("""
CREATE INDEX idx_closing_date
ON notices(closing_date)
""")

conn.commit()

index_time = time.perf_counter() - start_index

print(
    "Index creation time:",
    f"{index_time:.4f}",
    "seconds"
)


# ------------------------------------------------------------
# 5. VERIFY NUMBER OF ROWS
# ------------------------------------------------------------

cursor.execute(
    "SELECT COUNT(*) FROM notices"
)

row_count = cursor.fetchone()[0]

print(
    "\nRows stored in database:",
    row_count
)


# ------------------------------------------------------------
# 6. TEST INDEXED NOTICE-ID LOOKUP
# ------------------------------------------------------------

test_ids = [
    "N000001",
    "N000009",
    "N000017",
    "N000025",
    "N000033"
]

lookup_times = []

print("\nTesting indexed notice-ID lookups:")

for notice_id in test_ids:

    start = time.perf_counter()

    cursor.execute(
        """
        SELECT *
        FROM notices
        WHERE notice_id = ?
        """,
        (notice_id,)
    )

    result = cursor.fetchone()

    elapsed = time.perf_counter() - start

    lookup_times.append(elapsed)

    print(
        notice_id,
        "->",
        "Found" if result else "Not found",
        "| Time:",
        f"{elapsed * 1000:.4f} ms"
    )


# ------------------------------------------------------------
# 7. AVERAGE LOOKUP TIME
# ------------------------------------------------------------

average_lookup = sum(lookup_times) / len(lookup_times)

print(
    "\nAverage indexed lookup time:",
    f"{average_lookup * 1000:.4f}",
    "ms"
)


# ------------------------------------------------------------
# 8. QUERY BY PORTAL
# ------------------------------------------------------------

start = time.perf_counter()

cursor.execute("""
SELECT COUNT(*)
FROM notices
WHERE portal_id = ?
""", ("P136",))

portal_count = cursor.fetchone()[0]

portal_time = time.perf_counter() - start

print(
    "\nPortal P136 notices:",
    portal_count
)

print(
    "Indexed portal query time:",
    f"{portal_time * 1000:.4f}",
    "ms"
)


# ------------------------------------------------------------
# 9. DATABASE SIZE
# ------------------------------------------------------------

conn.commit()
conn.close()

database_size = os.path.getsize(DB_NAME)

print(
    "\nDatabase size:",
    f"{database_size / (1024 * 1024):.2f}",
    "MB"
)


# ------------------------------------------------------------
# 10. FINAL SUMMARY
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("DATABASE AND INDEX SUMMARY")
print("=" * 70)

print("Database:", DB_NAME)
print("Rows:", row_count)
print("Indexes:", 4)
print(
    "Average notice-ID lookup:",
    f"{average_lookup * 1000:.4f} ms"
)
print(
    "Database size:",
    f"{database_size / (1024 * 1024):.2f} MB"
)

print("\n")
print("=" * 70)
print("PART (D) COMPLETED")
print("=" * 70)