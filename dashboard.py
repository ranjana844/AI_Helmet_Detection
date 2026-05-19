import streamlit as st
import sqlite3
import pandas as pd

# =========================
# PAGE SETTINGS
# =========================

st.set_page_config(
    page_title="AI Helmet Traffic Dashboard",
    layout="wide"
)

# =========================
# TITLE
# =========================

st.title("AI Helmet Detection Traffic Dashboard")

st.markdown("---")

# =========================
# DATABASE CONNECTION
# =========================

conn = sqlite3.connect("traffic_system.db")

# Read table
query = "SELECT * FROM violations"

try:

    df = pd.read_sql_query(query, conn)

    # =========================
    # METRICS
    # =========================

    total_violations = len(df)

    col1, col2 = st.columns(2)

    col1.metric(
        "Total Violations",
        total_violations
    )

    col2.metric(
        "Database Status",
        "Connected"
    )

    st.markdown("---")

    # =========================
    # TABLE
    # =========================

    st.subheader("Violation Records")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.markdown("---")

    # =========================
    # CHART
    # =========================

    st.subheader("Violation Analytics")

    violation_count = df["violation"].value_counts()

    st.bar_chart(violation_count)

except:

    st.error("No data found in database.")

# Close DB
conn.close()