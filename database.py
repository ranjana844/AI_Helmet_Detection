import sqlite3
from datetime import datetime

# Create database
conn = sqlite3.connect("traffic_system.db")

cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    violation TEXT,
    date_time TEXT
)
""")

conn.commit()

# Insert violation function
def insert_violation(violation):

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
    INSERT INTO violations (
        violation,
        date_time
    )
    VALUES (?, ?)
    """, (violation, current_time))

    conn.commit()