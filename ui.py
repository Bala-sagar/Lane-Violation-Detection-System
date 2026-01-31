import streamlit as st
import sqlite3
import os
from PIL import Image

DB_PATH = "violations.db"

st.set_page_config(
    page_title="Lane Violation Monitor",
    layout="wide"
)

st.title("🚦 Lane Violation Monitoring System")

# ---------- DATABASE ----------

def get_violations():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, timestamp, vehicle_type, lane,
               violation_type, plate_number, event_dir
        FROM violations
        ORDER BY id DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


violations = get_violations()

if not violations:
    st.warning("No violations recorded yet.")
    st.stop()

# ---------- SIDEBAR ----------
st.sidebar.header("📋 Violations")

# ---- FILTER ----
filter_option = st.sidebar.radio(
    "Filter by violation type",
    (
        "All",
        "heavy_in_normal_lane",
        "normal_in_heavy_lane"
    )
)

# Apply filter
if filter_option != "All":
    filtered = [v for v in violations if v[4] == filter_option]
else:
    filtered = violations

if not filtered:
    st.sidebar.warning("No violations found.")
    st.stop()

# ---- LIST VIEW ----
labels = [
    f"{v[1]} | {v[4]} | {v[5]}"
    for v in filtered
]

selected_index = st.sidebar.radio(
    "Select an event",
    range(len(labels)),
    format_func=lambda i: labels[i]
)

selected = filtered[selected_index]


(
    vid,
    timestamp,
    vehicle_type,
    lane,
    violation_type,
    plate_number,
    event_dir
) = selected

# ---------- MAIN VIEW ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Full Frame")
    frame_path = os.path.join(event_dir, "frame.jpg")
    if os.path.exists(frame_path):
        st.image(Image.open(frame_path), width=600)

with col2:
    st.subheader("🚗 Vehicle")
    vehicle_path = os.path.join(event_dir, "vehicle.jpg")
    if os.path.exists(vehicle_path):
        st.image(Image.open(vehicle_path), width=600)


st.subheader("🔢 Number Plate")
plate_path = os.path.join(event_dir, "plate.jpg")
if os.path.exists(plate_path):
    st.image(Image.open(plate_path), width=300)

st.subheader("📝 Metadata")
st.markdown(f"""
- **Timestamp:** {timestamp}
- **Vehicle Type:** {vehicle_type}
- **Lane:** {lane}
- **Violation:** {violation_type}
- **Plate Number:** {plate_number}
""")
