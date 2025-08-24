# adk_medical_evaluation/local_storage/sqlite_storage.py
"""
SQLite storage utilities for medical evaluation
Project: mindfulfocus-470008
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

class SQLiteStorage:
    """SQLite database utilities for medical evaluation data"""
    
    def __init__(self, db_path: str = "local_database/medical_evaluation.db"):
        self.db_path = db_path
        self.base_dir = os.path.dirname(db_path) if os.path.dirname(db_path) else "."
        
        # Create directory if needed
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Initialize database
        self.init_extended_database()
        
        print(f"✅ SQLiteStorage initialized: {db_path}")
    
    def init_extended_database(self):
        """Initialize extended database schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Enable foreign keys
                conn.execute("PRAGMA foreign_keys = ON")
                
                # Sessions table (if not exists)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        app_name TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        state TEXT NOT NULL DEFAULT '{}',
                        messages TEXT NOT NULL DEFAULT '[]',
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL,
                        UNIQUE(app_name, user_id, session_id)
                    )
                """)
                
                # Evaluation runs table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS evaluation_runs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_key TEXT NOT NULL,
                        pipeline_data TEXT,
                        results TEXT,
                        metrics TEXT,
                        status TEXT DEFAULT 'running',
                        timestamp TIMESTAMP NOT NULL,
                        duration_seconds INTEGER,
                        frames_processed INTEGER DEFAULT 0,
                        success_rate REAL,
                        average_symmetry REAL,
                        average_ear_difference REAL,
                        overall_score REAL,
                        FOREIGN KEY(session_key) REFERENCES sessions(id)
                    )
                """)
                
                # Frame analyses table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS frame_analyses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        evaluation_run_id INTEGER,
                        frame_id INTEGER NOT NULL,
                        image_path TEXT,
                        landmarks_detected INTEGER,
                        symmetry_score REAL,
                        ear_left REAL,
                        ear_right REAL,
                        ear_difference REAL,
                        mouth_asymmetry_mm REAL,
                        eyebrow_diff_mm REAL,
                        severity_score REAL,
                        analysis_data TEXT,
                        processing_time_ms REAL,
                        timestamp TIMESTAMP NOT NULL,
                        FOREIGN KEY(evaluation_run_id) REFERENCES evaluation_runs(id)
                    )
                """)
                
                # Performance metrics table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        evaluation_run_id INTEGER,
                        metric_name TEXT NOT NULL,
                        metric_value REAL,
                        metric_category TEXT,
                        metric_data TEXT,
                        timestamp TIMESTAMP NOT NULL,
                        FOREIGN KEY(evaluation_run_id) REFERENCES evaluation_runs(id)
                    )
                """)
                
                # Agent interactions table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS agent_interactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_key TEXT NOT NULL,
                        agent_name TEXT NOT NULL,
                        interaction_type TEXT,  -- 'query', 'tool_call', 'response'
                        input_data TEXT,
                        output_data TEXT,
                        processing_time_ms REAL,
                        timestamp TIMESTAMP NOT NULL,
                        FOREIGN KEY(session_key) REFERENCES sessions(id)
                    )
                """)
                
                # Reports table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS reports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        evaluation_run_id INTEGER,
                        report_type TEXT NOT NULL,  -- 'final', 'interim', 'summary'
                        report_data TEXT NOT NULL,
                        file_path TEXT,
                        timestamp TIMESTAMP NOT NULL,
                        FOREIGN KEY(evaluation_run_id) REFERENCES evaluation_runs(id)
                    )
                """)
                
                # Create indexes for better performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_app_user ON sessions(app_name, user_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_runs_session ON evaluation_runs(session_key)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_frame_analyses_run ON frame_analyses(evaluation_run_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_frame_analyses_frame ON frame_analyses(frame_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_performance_metrics_run ON performance_metrics(evaluation_run_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_interactions_session ON agent_interactions(session_key)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_reports_run ON reports(evaluation_run_id)")
                
                conn.commit()
                print("✅ Extended database schema initialized")
                
        except Exception as e:
            print(f"❌ Error initializing extended database: {e}")
            raise
    
    def save_frame_analysis_detailed(self, evaluation_run_id: int, frame_analysis: Dict[str, Any]) -> int:
        """Save detailed frame analysis to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO frame_analyses 
                    (evaluation_run_id, frame_id, image_path, landmarks_detected, 
                     symmetry_score, ear_left, ear_right, ear_difference,
                     mouth_asymmetry_mm, eyebrow_diff_mm, severity_score,
                     analysis_data, processing_time_ms, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    evaluation_run_id,
                    frame_analysis.get('frame_id', 0),
                    frame_analysis.get('image_path', ''),
                    frame_analysis.get('landmarks_detected', 0),
                    frame_analysis.get('symmetry_score', 0.0),
                    frame_analysis.get('ear_left', 0.0),
                    frame_analysis.get('ear_right', 0.0),
                    frame_analysis.get('ear_difference', 0.0),
                    frame_analysis.get('mouth_asymmetry_mm', 0.0),
                    frame_analysis.get('eyebrow_diff_mm', 0.0),
                    frame_analysis.get('severity_score', 0.0),
                    json.dumps(frame_analysis),
                    frame_analysis.get('processing_time_ms', 0.0),
                    datetime.now().isoformat()
                ))
                conn.commit()
                return cursor.lastrowid
                
        except Exception as e:
            print(f"❌ Error saving detailed frame analysis: {e}")
            raise
    
    def save_performance_metrics(self, evaluation_run_id: int, metrics: Dict[str, Any]) -> List[int]:
        """Save performance metrics to database"""
        metric_ids = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                timestamp = datetime.now().isoformat()
                
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        # Determine category
                        category = self._get_metric_category(metric_name)
                        
                        cursor = conn.execute("""
                            INSERT INTO performance_metrics
                            (evaluation_run_id, metric_name, metric_value, metric_category, timestamp)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            evaluation_run_id,
                            metric_name,
                            float(metric_value),
                            category,
                            timestamp
                        ))
                        metric_ids.append(cursor.lastrowid)
                    
                    elif isinstance(metric_value, dict):
                        # Save complex metrics as JSON
                        cursor = conn.execute("""
                            INSERT INTO performance_metrics
                            (evaluation_run_id, metric_name, metric_data, metric_category, timestamp)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            evaluation_run_id,
                            metric_name,
                            json.dumps(metric_value),
                            self._get_metric_category(metric_name),
                            timestamp
                        ))
                        metric_ids.append(cursor.lastrowid)
                
                conn.commit()
                
        except Exception as e:
            print(f"❌ Error saving performance metrics: {e}")
            raise
        
        return metric_ids
    
    def _get_metric_category(self, metric_name: str) -> str:
        """Categorize metric based on name"""
        name_lower = metric_name.lower()
        
        if 'symmetry' in name_lower:
            return 'symmetry'
        elif 'ear' in name_lower or 'eye' in name_lower:
            return 'eye_analysis'
        elif 'mouth' in name_lower:
            return 'mouth_analysis'
        elif 'performance' in name_lower or 'time' in name_lower or 'fps' in name_lower:
            return 'performance'
        elif 'accuracy' in name_lower or 'precision' in name_lower or 'recall' in name_lower:
            return 'accuracy'
        elif 'score' in name_lower:
            return 'scoring'
        else:
            return 'general'
    
    def log_agent_interaction(self, session_key: str, agent_name: str, 
                            interaction_type: str, input_data: Any = None, 
                            output_data: Any = None, processing_time_ms: float = 0.0) -> int:
        """Log agent interaction"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO agent_interactions
                    (session_key, agent_name, interaction_type, input_data, 
                     output_data, processing_time_ms, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_key,
                    agent_name,
                    interaction_type,
                    json.dumps(input_data) if input_data else None,
                    json.dumps(output_data) if output_data else None,
                    processing_time_ms,
                    datetime.now().isoformat()
                ))
                conn.commit()
                return cursor.lastrowid
                
        except Exception as e:
            print(f"❌ Error logging agent interaction: {e}")
            return -1
    
    def save_report(self, evaluation_run_id: int, report_type: str, 
                   report_data: Dict[str, Any], file_path: str = None) -> int:
        """Save evaluation report"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO reports
                    (evaluation_run_id, report_type, report_data, file_path, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    evaluation_run_id,
                    report_type,
                    json.dumps(report_data, default=str),
                    file_path,
                    datetime.now().isoformat()
                ))
                conn.commit()
                return cursor.lastrowid
                
        except Exception as e:
            print(f"❌ Error saving report: {e}")
            raise
    
    def get_evaluation_run_summary(self, evaluation_run_id: int) -> Optional[Dict[str, Any]]:
        """Get summary of evaluation run"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get basic run info
                cursor = conn.execute("""
                    SELECT * FROM evaluation_runs WHERE id = ?
                """, (evaluation_run_id,))
                
                run_data = cursor.fetchone()
                if not run_data:
                    return None
                
                # Get frame analysis stats
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as frame_count,
                        AVG(symmetry_score) as avg_symmetry,
                        AVG(ear_difference) as avg_ear_diff,
                        AVG(severity_score) as avg_severity,
                        AVG(landmarks_detected) as avg_landmarks,
                        AVG(processing_time_ms) as avg_processing_time
                    FROM frame_analyses WHERE evaluation_run_id = ?
                """, (evaluation_run_id,))
                
                stats = cursor.fetchone()
                
                # Get performance metrics
                cursor = conn.execute("""
                    SELECT metric_name, metric_value, metric_category
                    FROM performance_metrics 
                    WHERE evaluation_run_id = ? AND metric_value IS NOT NULL
                """, (evaluation_run_id,))
                
                metrics = {row[0]: row[1] for row in cursor.fetchall()}
                
                summary = {
                    'run_id': evaluation_run_id,
                    'status': run_data[4],  # status column
                    'duration_seconds': run_data[6],  # duration_seconds column
                    'frames_processed': run_data[7] if run_data[7] else stats[0] if stats else 0,
                    'frame_stats': {
                        'total_frames': stats[0] if stats else 0,
                        'avg_symmetry_score': round(stats[1], 4) if stats and stats[1] else 0,
                        'avg_ear_difference': round(stats[2], 4) if stats and stats[2] else 0,
                        'avg_severity_score': round(stats[3], 2) if stats and stats[3] else 0,
                        'avg_landmarks_detected': round(stats[4], 1) if stats and stats[4] else 0,
                        'avg_processing_time_ms': round(stats[5], 2) if stats and stats[5] else 0
                    },
                    'performance_metrics': metrics,
                    'overall_score': run_data[9] if run_data[9] else 0  # overall_score column
                }
                
                return summary
                
        except Exception as e:
            print(f"❌ Error getting evaluation run summary: {e}")
            return None
    
    def get_session_evaluations(self, session_key: str) -> List[Dict[str, Any]]:
        """Get all evaluation runs for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, status, timestamp, duration_seconds, frames_processed, overall_score
                    FROM evaluation_runs 
                    WHERE session_key = ?
                    ORDER BY timestamp DESC
                """, (session_key,))
                
                runs = []
                for row in cursor.fetchall():
                    runs.append({
                        'id': row[0],
                        'status': row[1],
                        'timestamp': row[2],
                        'duration_seconds': row[3],
                        'frames_processed': row[4],
                        'overall_score': row[5]
                    })
                
                return runs
                
        except Exception as e:
            print(f"❌ Error getting session evaluations: {e}")
            return []
    
    def update_evaluation_run_status(self, evaluation_run_id: int, status: str, 
                                   final_metrics: Dict[str, Any] = None) -> bool:
        """Update evaluation run status and final metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if final_metrics:
                    conn.execute("""
                        UPDATE evaluation_runs 
                        SET status = ?, frames_processed = ?, success_rate = ?, 
                            average_symmetry = ?, average_ear_difference = ?, overall_score = ?
                        WHERE id = ?
                    """, (
                        status,
                        final_metrics.get('total_frames', 0),
                        final_metrics.get('success_rate', 0.0),
                        final_metrics.get('average_symmetry_score', 0.0),
                        final_metrics.get('average_ear_difference', 0.0),
                        final_metrics.get('overall_score', 0.0),
                        evaluation_run_id
                    ))
                else:
                    conn.execute("""
                        UPDATE evaluation_runs SET status = ? WHERE id = ?
                    """, (status, evaluation_run_id))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"❌ Error updating evaluation run status: {e}")
            return False
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}
                
                # Table counts
                tables = ['sessions', 'evaluation_runs', 'frame_analyses', 
                         'performance_metrics', 'agent_interactions', 'reports']
                
                for table in tables:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f'{table}_count'] = cursor.fetchone()[0]
                
                # Recent activity (last 24 hours)
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM evaluation_runs 
                    WHERE datetime(timestamp) > datetime('now', '-1 day')
                """)
                stats['recent_evaluations'] = cursor.fetchone()[0]
                
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM frame_analyses 
                    WHERE datetime(timestamp) > datetime('now', '-1 day')
                """)
                stats['recent_frame_analyses'] = cursor.fetchone()[0]
                
                # Performance statistics
                cursor = conn.execute("""
                    SELECT 
                        AVG(overall_score) as avg_overall_score,
                        MAX(overall_score) as best_score,
                        MIN(overall_score) as worst_score
                    FROM evaluation_runs 
                    WHERE overall_score IS NOT NULL
                """)
                
                perf_stats = cursor.fetchone()
                if perf_stats:
                    stats['performance'] = {
                        'average_overall_score': round(perf_stats[0], 3) if perf_stats[0] else 0,
                        'best_score': round(perf_stats[1], 3) if perf_stats[1] else 0,
                        'worst_score': round(perf_stats[2], 3) if perf_stats[2] else 0
                    }
                
                # Database size
                cursor = conn.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                cursor = conn.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                stats['database_size_mb'] = round((page_count * page_size) / (1024*1024), 2)
                
                return stats
                
        except Exception as e:
            print(f"❌ Error getting database statistics: {e}")
            return {}