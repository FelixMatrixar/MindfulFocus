# adk_medical_evaluation/local_storage/session_manager.py
"""
Local session management using SQLite database
Project: mindfulfocus-470008
"""

import sqlite3
import json
import os
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class Session:
    """Local session data structure"""
    app_name: str
    user_id: str
    session_id: str
    state: Dict[str, Any]
    messages: List[Dict[str, Any]] = None
    created_at: str = None
    updated_at: str = None

    def __post_init__(self):
        if self.messages is None:
            self.messages = []
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = datetime.now().isoformat()

class LocalSessionService:
    """Local session service using SQLite for medical evaluation"""

    def __init__(self, db_path: str = "local_database/medical_evaluation.db"):
        self.db_path = db_path
        self.base_dir = os.path.dirname(db_path) if os.path.dirname(db_path) else "."

        # Create directory if it doesn't exist
        os.makedirs(self.base_dir, exist_ok=True)

        # Initialize database
        self.init_database()

        print(f"✅ LocalSessionService initialized with database: {db_path}")

    # --------------------------- Internal helpers ---------------------------

    @staticmethod
    def _json_dump_safe(obj: Any) -> str:
        """Serialize to JSON, stringifying non-JSON-serializable values."""
        return json.dumps(obj, default=str)

    # --------------------------- Schema setup ---------------------------

    def init_database(self):
        """Initialize SQLite database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Sessions table
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
                        frames_processed INTEGER,
                        FOREIGN KEY(session_key) REFERENCES sessions(id)
                    )
                """)

                # Frame analysis table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS frame_analyses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        evaluation_run_id INTEGER,
                        frame_id INTEGER NOT NULL,
                        image_path TEXT,
                        landmarks_detected INTEGER,
                        symmetry_score REAL,
                        ear_difference REAL,
                        severity_score REAL,
                        analysis_data TEXT,
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
                        metric_data TEXT,
                        timestamp TIMESTAMP NOT NULL,
                        FOREIGN KEY(evaluation_run_id) REFERENCES evaluation_runs(id)
                    )
                """)

                conn.commit()
                print("✅ Database tables initialized successfully")

        except Exception as e:
            print(f"❌ Error initializing database: {e}")
            raise

    # --------------------------- Sessions ---------------------------

    async def create_session(self, app_name: str, user_id: str,
                             session_id: str, state: Dict[str, Any] = None) -> Session:
        """Create a new session"""
        if state is None:
            state = {}

        session_key = f"{app_name}_{user_id}_{session_id}"
        current_time = datetime.now().isoformat()

        # Add default state values
        default_state = {
            "session_created": current_time,
            "project_id": "mindfulfocus-470008",
            "model": "gemini-2.5-pro",
            "frames_analyzed": 0,
            "evaluation_status": "initialized",
            "analysis_results": []
        }
        default_state.update(state)

        try:
            # Use thread executor for database operations
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                self._create_session_sync,
                session_key, app_name, user_id, session_id,
                default_state, current_time
            )

            session = Session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                state=default_state,
                created_at=current_time,
                updated_at=current_time
            )

            print(f"✅ Session created: {session_key}")
            return session

        except Exception as e:
            print(f"❌ Error creating session: {e}")
            raise

    def _create_session_sync(self, session_key: str, app_name: str, user_id: str,
                             session_id: str, state: Dict[str, Any], current_time: str):
        """Synchronous session creation for executor"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sessions
                (id, app_name, user_id, session_id, state, messages, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_key, app_name, user_id, session_id,
                self._json_dump_safe(state), self._json_dump_safe([]),
                current_time, current_time
            ))
            conn.commit()

    async def get_session(self, app_name: str, user_id: str,
                          session_id: str) -> Optional[Session]:
        """Get existing session"""
        session_key = f"{app_name}_{user_id}_{session_id}"

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self._get_session_sync, session_key)

            if result:
                res_app_name, res_user_id, res_session_id, state_json, messages_json, created_at, updated_at = result
                return Session(
                    app_name=res_app_name,
                    user_id=res_user_id,
                    session_id=res_session_id,
                    state=json.loads(state_json),
                    messages=json.loads(messages_json),
                    created_at=created_at,
                    updated_at=updated_at
                )
            return None

        except Exception as e:
            print(f"❌ Error getting session: {e}")
            return None

    def _get_session_sync(self, session_key: str):
        """Synchronous session retrieval"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT app_name, user_id, session_id, state, messages, created_at, updated_at
                FROM sessions WHERE id = ?
            """, (session_key,))
            return cursor.fetchone()

    async def update_session_state(self, app_name: str, user_id: str,
                                   session_id: str, state_updates: Dict[str, Any]) -> bool:
        """Update session state"""
        session = await self.get_session(app_name, user_id, session_id)
        if not session:
            return False

        # Update state
        session.state.update(state_updates)
        session.updated_at = datetime.now().isoformat()

        session_key = f"{app_name}_{user_id}_{session_id}"

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                self._update_session_sync,
                session_key, session.state, session.updated_at
            )
            return True

        except Exception as e:
            print(f"❌ Error updating session state: {e}")
            return False

    def _update_session_sync(self, session_key: str, state: Dict[str, Any], updated_at: str):
        """Synchronous state update"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE sessions SET state = ?, updated_at = ? WHERE id = ?
            """, (self._json_dump_safe(state), updated_at, session_key))
            conn.commit()

    # --------------------------- Evaluation runs ---------------------------

    async def create_evaluation_run(self, session_key: str, pipeline_data: Dict[str, Any] = None) -> int:
        """Create a new evaluation run record"""
        try:
            loop = asyncio.get_running_loop()
            run_id = await loop.run_in_executor(
                None,
                self._create_evaluation_run_sync,
                session_key, pipeline_data
            )
            print(f"✅ Evaluation run created with ID: {run_id}")
            return run_id

        except Exception as e:
            print(f"❌ Error creating evaluation run: {e}")
            raise

    def _create_evaluation_run_sync(self, session_key: str, pipeline_data: Optional[Dict[str, Any]] = None) -> int:
        """Synchronous evaluation run creation"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO evaluation_runs
                (session_key, pipeline_data, timestamp, status)
                VALUES (?, ?, ?, ?)
            """, (
                session_key,
                self._json_dump_safe(pipeline_data or {}),
                datetime.now().isoformat(),
                "running"
            ))
            conn.commit()
            return cursor.lastrowid

    async def update_evaluation_run(
        self,
        evaluation_run_id: int,
        *,
        results: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
        duration_seconds: Optional[int] = None,
        frames_processed: Optional[int] = None
    ) -> bool:
        """Update an existing evaluation run record."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._update_evaluation_run_sync,
            evaluation_run_id, results, metrics, status, duration_seconds, frames_processed
        )

    def _update_evaluation_run_sync(
        self,
        evaluation_run_id: int,
        results: Optional[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]],
        status: Optional[str],
        duration_seconds: Optional[int],
        frames_processed: Optional[int]
    ) -> bool:
        fields: List[str] = []
        params: List[Any] = []

        if results is not None:
            fields.append("results = ?")
            params.append(self._json_dump_safe(results))
        if metrics is not None:
            fields.append("metrics = ?")
            params.append(self._json_dump_safe(metrics))
        if status is not None:
            fields.append("status = ?")
            params.append(status)
        if duration_seconds is not None:
            fields.append("duration_seconds = ?")
            params.append(int(duration_seconds))
        if frames_processed is not None:
            fields.append("frames_processed = ?")
            params.append(int(frames_processed))

        if not fields:
            return True  # nothing to update

        sql = f"UPDATE evaluation_runs SET {', '.join(fields)} WHERE id = ?"
        params.append(evaluation_run_id)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(sql, tuple(params))
            conn.commit()
        return True

    # --------------------------- Frame analyses ---------------------------

    async def save_frame_analysis(self, evaluation_run_id: int, frame_data: Dict[str, Any]) -> bool:
        """Save frame analysis results"""
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                self._save_frame_analysis_sync,
                evaluation_run_id, frame_data
            )
            return True

        except Exception as e:
            print(f"❌ Error saving frame analysis: {e}")
            return False

    def _save_frame_analysis_sync(self, evaluation_run_id: int, frame_data: Dict[str, Any]):
        """Synchronous frame analysis save"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO frame_analyses
                (evaluation_run_id, frame_id, image_path, landmarks_detected,
                 symmetry_score, ear_difference, severity_score, analysis_data, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                evaluation_run_id,
                frame_data.get('frame_id', 0),
                frame_data.get('image_path', ''),
                frame_data.get('landmarks_detected', 0),
                frame_data.get('symmetry_score', 0.0),
                frame_data.get('ear_difference', 0.0),
                frame_data.get('severity_score', 0.0),
                self._json_dump_safe(frame_data),
                datetime.now().isoformat()
            ))
            conn.commit()

    # --------------------------- Stats ---------------------------

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats: Dict[str, Any] = {}

                # Count sessions
                cursor = conn.execute("SELECT COUNT(*) FROM sessions")
                stats['total_sessions'] = cursor.fetchone()[0]

                # Count evaluation runs
                cursor = conn.execute("SELECT COUNT(*) FROM evaluation_runs")
                stats['total_evaluation_runs'] = cursor.fetchone()[0]

                # Count frame analyses
                cursor = conn.execute("SELECT COUNT(*) FROM frame_analyses")
                stats['total_frame_analyses'] = cursor.fetchone()[0]

                # Count performance metrics
                cursor = conn.execute("SELECT COUNT(*) FROM performance_metrics")
                stats['total_performance_metrics'] = cursor.fetchone()[0]

                return stats

        except Exception as e:
            print(f"❌ Error getting database stats: {e}")
            return {}
