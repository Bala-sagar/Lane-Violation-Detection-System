import sqlite3

DB_PATH = "violations.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            vehicle_type TEXT,
            lane TEXT,
            violation_type TEXT,
            plate_number TEXT,
            event_dir TEXT
        )
    """)

    conn.commit()
    conn.close()


def insert_violation(timestamp, vehicle_type, lane, violation_type, plate_number, event_dir):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO violations (
            timestamp, vehicle_type, lane,
            violation_type, plate_number, event_dir
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        timestamp,
        vehicle_type,
        lane,
        violation_type,
        plate_number,
        event_dir
    ))

    conn.commit()
    conn.close()
