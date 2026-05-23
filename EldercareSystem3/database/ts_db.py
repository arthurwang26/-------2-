import sqlite3
import datetime
import json
from pathlib import Path
from config import cfg
from utils.logger import get_logger

logger = get_logger("ts_db")

class TimeSeriesDB:
    def __init__(self, db_path=None):
        if db_path is None:
            db_dir = cfg.output_dir / "database"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "eldercare_ts.db"
            
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Anomaly Scores Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    video_name TEXT,
                    anomaly_score REAL
                )
            ''')
            # Events Table (Emotions, Actions, HOI)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    video_name TEXT,
                    person TEXT,
                    event_type TEXT,
                    event_value TEXT,
                    confidence REAL
                )
            ''')
            # Temporal Risk Score Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS temporal_risk (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    risk_score REAL,
                    trend TEXT
                )
            ''')
            conn.commit()
            logger.info(f"Time-Series Database initialized at {self.db_path}")

    def insert_anomaly(self, video_name, score):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO anomalies (video_name, anomaly_score) VALUES (?, ?)",
                (video_name, float(score))
            )
            conn.commit()

    def insert_events(self, events_by_clip):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for vid_name, events in events_by_clip.items():
                for ev in events:
                    cursor.execute(
                        "INSERT INTO events (video_name, person, event_type, event_value, confidence) VALUES (?, ?, ?, ?, ?)",
                        (
                            vid_name, 
                            ev.get("person", "Unknown"), 
                            ev.get("type", "Unknown"), 
                            str(ev.get("action", ev.get("emotion", ev.get("description", "")))), 
                            float(ev.get("confidence", 1.0))
                        )
                    )
            conn.commit()

    def insert_temporal_risk(self, risk_score, trend):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO temporal_risk (risk_score, trend) VALUES (?, ?)",
                (float(risk_score), str(trend))
            )
            conn.commit()

