# adk_medical_evaluation/main_evaluator.py
"""
Main medical pipeline evaluator using ADK with local storage
Project: mindfulfocus-470008
Model: Gemini 2.5 Pro
"""

import asyncio
import os
import json
import time
from copy import deepcopy
from datetime import datetime
from typing import Dict, Any, Optional

# Optional: only needed if you actually capture frames in this file
# import cv2

# ADK imports
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService   # ✅ ADK-compliant session service
from google.genai import types

# Local imports
from local_storage.session_manager import LocalSessionService  # your own SQLite persistence
from local_storage.sqlite_storage import SQLiteStorage         # keep if you still use it elsewhere
from local_storage.file_storage import LocalFileStorage
from agents.medical_evaluator_agent import create_medical_evaluator_agent
from config.agent_config import get_api_config, EVALUATION_CONFIG, STORAGE_CONFIG


APP_NAME = "medical_pipeline_evaluation"
USER_ID = "evaluator"


def _jsonify_config(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Return a JSON-safe shallow copy of cfg: any non-serializable value is stringified.
    """
    cfg = deepcopy(cfg or {})
    def _to_jsonable(x):
        try:
            json.dumps(x)
            return x
        except TypeError:
            return str(x)
    return {k: _to_jsonable(v) for k, v in cfg.items()}


class LocalMedicalPipelineEvaluator:
    """
    Comprehensive local medical pipeline evaluator using ADK
    """

    def __init__(self, model: str = "gemini-2.5-pro", project_id: str = "mindfulfocus-470008"):
        self.project_id = project_id
        self.model = model

        # ✅ Use ADK's session service with Runner (fixes InvocationContext errors)
        self.adk_session_service = InMemorySessionService()

        # Keep your own local persistence separate (never pass these to Runner)
        self.local_db = LocalSessionService(STORAGE_CONFIG["database_path"])
        self.sqlite_storage = SQLiteStorage(STORAGE_CONFIG["database_path"])  # if you still need it
        self.file_storage = LocalFileStorage("evaluation_data")

        # Agent & runner
        self.agent = create_medical_evaluator_agent(model)
        self.runner = Runner(
            agent=self.agent,
            app_name=APP_NAME,
            session_service=self.adk_session_service,  # ✅ ADK-compliant service only
        )

        # Session tracking (ADK)
        self.session_id: Optional[str] = None
        self.session_state_snapshot: Dict[str, Any] = {}

        # Evaluation state
        self.evaluation_active = False
        self.current_evaluation_run_id: Optional[int] = None

        print(f"🚀 LocalMedicalPipelineEvaluator initialized")
        print(f"📊 Project: {project_id}")
        print(f"🤖 Model: {model}")

    async def initialize(self, session_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the evaluation system with a new ADK session and a local evaluation run.
        """
        print("🔧 Initializing evaluation session...")

        try:
            # Setup API configuration
            api_config = get_api_config()
            os.environ["GOOGLE_API_KEY"] = api_config.get("api_key", "")
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = str(api_config.get("use_vertex_ai", "False"))

            if not os.environ.get("GOOGLE_API_KEY"):
                print("❌ GOOGLE_API_KEY not set! Please set your Gemini API key.")
                return False

            # Prepare a JSON-safe initial state (DO NOT store live objects)
            initial_state = {
                "evaluation_started": datetime.now().isoformat(),
                "project_id": self.project_id,
                "model": self.model,
                "frames_analyzed": 0,
                "storage_info": {
                    "session_db_path": getattr(self.local_db, "db_path", STORAGE_CONFIG.get("database_path")),
                    "sqlite_db_path": getattr(self.sqlite_storage, "db_path", STORAGE_CONFIG.get("database_path", "")) if hasattr(self.sqlite_storage, "db_path") else STORAGE_CONFIG.get("database_path", ""),
                    "file_root": getattr(self.file_storage, "base_dir", "evaluation_data"),
                    "backend": "local_sqlite"
                },
                "evaluation_config": _jsonify_config(EVALUATION_CONFIG),
            }

            # Merge optional session config (stringify anything odd)
            if session_config:
                for k, v in session_config.items():
                    try:
                        json.dumps(v)
                        initial_state[k] = v
                    except TypeError:
                        initial_state[k] = str(v)

            # ✅ Create the ADK session (returns an ADK Session object)
            adk_session = await self.adk_session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                state=initial_state
            )
            self.session_id = adk_session.id
            self.session_state_snapshot = initial_state

            # Create a local evaluation run record in YOUR DB (separate from ADK)
            session_key = f"{APP_NAME}_{USER_ID}_{self.session_id}"
            self.current_evaluation_run_id = await self.local_db.create_evaluation_run(
                session_key=session_key,
                pipeline_data={"project_id": self.project_id, "model": self.model}
            )

            print(f"✅ Session initialized: {self.session_id}")
            print(f"📋 Evaluation run ID: {self.current_evaluation_run_id}")
            return True

        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False

    async def evaluate_from_camera(self,
                                   duration_seconds: int = 300,
                                   analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Evaluate pipeline using live camera feed (agent tools must implement the heavy lifting).
        """
        if not self.session_id:
            print("❌ System not initialized. Call initialize() first.")
            return {"error": "System not initialized"}

        print(f"📹 Starting camera evaluation for {duration_seconds} seconds")
        print(f"🎯 Analysis type: {analysis_type}")

        self.evaluation_active = True
        start_time = time.time()

        try:
            # Step 1: Process camera stream
            print("📹 Step 1: Processing camera stream...")
            stream_config = {
                "duration_seconds": duration_seconds,
                "camera_index": 0,
                "fps_target": 10,
                "width": 1280,
                "height": 720
            }
            stream_query = f"""Please process the camera stream for medical evaluation.

Stream configuration: {json.dumps(stream_config, indent=2)}

Use the 'process_camera_stream' tool to:
1. Capture frames from the camera for {duration_seconds} seconds
2. Save frames locally with metadata
3. Prepare frames for batch analysis
4. Provide processing statistics

Focus on maintaining consistent quality and frame rate."""
            stream_result = await self._call_agent(stream_query)
            print(f"📹 Camera stream processing: {stream_result[:200]}...")

            # Step 2: Batch process frames
            print("📦 Step 2: Batch processing frames...")
            batch_config = {"batch_size": 10, "max_workers": 2}
            batch_query = f"""Please process the captured frames in batches for detailed medical analysis.

Batch configuration: {json.dumps(batch_config, indent=2)}

Use the 'batch_process_frames' tool to:
1. Process frames from the camera stream queue
2. Analyze each frame for facial landmarks and medical metrics
3. Calculate symmetry, EAR, and severity scores
4. Handle any processing errors gracefully
5. Provide batch processing statistics

Ensure thorough analysis of each frame while maintaining processing efficiency."""
            batch_result = await self._call_agent(batch_query)
            print(f"📦 Batch processing: {batch_result[:200]}...")

            # Step 3: Calculate performance metrics
            print("📊 Step 3: Calculating performance metrics...")
            metrics_query = f"""Please calculate comprehensive performance metrics from the batch processing results.

Use the 'calculate_performance_metrics' tool to:
1. Analyze success rates and processing performance
2. Calculate medical accuracy metrics
3. Assess consistency and reliability
4. Determine deployment readiness
5. Generate detailed performance report

Focus on both technical performance and medical accuracy assessments."""
            metrics_result = await self._call_agent(metrics_query)
            print(f"📊 Performance metrics: {metrics_result[:200]}...")

            # Step 4: Medical assessment
            print("🏥 Step 4: Generating medical assessment...")
            medical_query = f"""Please generate a comprehensive medical assessment based on the analysis results.

Use the 'generate_medical_assessment' tool to:
1. Analyze patterns in facial asymmetry across all frames
2. Assess clinical significance of findings
3. Generate severity classifications
4. Provide medical recommendations and next steps
5. Include appropriate medical disclaimers

Focus on clinical relevance and actionable medical insights."""
            medical_result = await self._call_agent(medical_query)
            print(f"🏥 Medical assessment: {medical_result[:200]}...")

            # Step 5: Aggregate & export
            print("🔄 Step 5: Aggregating and exporting results...")
            export_config = {"formats": ["json", "csv", "txt", "medical_report"]}
            export_query = f"""Please aggregate all results and export in multiple formats.

Export configuration: {json.dumps(export_config, indent=2)}

Use the 'aggregate_results' and 'export_results' tools to:
1. Compile comprehensive statistical analysis
2. Export results in JSON, CSV, and text formats
3. Generate clinical report for medical review
4. Create summary assessment with recommendations
5. Organize all outputs for easy access

Ensure exports are comprehensive and suitable for both technical and clinical audiences."""
            export_result = await self._call_agent(export_query)
            print(f"📤 Export results: {export_result[:200]}...")

            # Step 6: Final report
            print("📋 Step 6: Generating final evaluation report...")
            report_query = f"""Please generate and save a comprehensive final evaluation report.

Use the 'save_evaluation_report' tool to:
1. Compile all analysis results into a comprehensive report
2. Include executive summary with key findings
3. Provide technical performance assessment
4. Include medical evaluation and recommendations
5. Add deployment readiness assessment
6. Save report with complete metadata

This should be the definitive evaluation document for this session."""
            final_result = await self._call_agent(report_query)
            print(f"📋 Final report: {final_result[:200]}...")

            # Wrap up
            total_time = time.time() - start_time

            # ✅ Update your local DB (not ADK) with final status
            if self.current_evaluation_run_id:
                await self.local_db.update_evaluation_run(
                    self.current_evaluation_run_id,
                    status="completed",
                    metrics={"total_duration_seconds": total_time}
                )

            evaluation_summary = {
                "status": "completed",
                "session_id": self.session_id,
                "evaluation_run_id": self.current_evaluation_run_id,
                "total_duration_seconds": round(total_time, 2),
                "analysis_type": analysis_type,
                "steps_completed": 6,
                "final_report_generated": True,
                "database_updated": True,
                "evaluation_timestamp": datetime.now().isoformat()
            }

            print(f"✅ Evaluation complete!")
            print(f"📊 Duration: {total_time:.1f} seconds")
            print(f"🎯 Session: {self.session_id}")
            print(f"📋 Run ID: {self.current_evaluation_run_id}")

            return evaluation_summary

        except Exception as e:
            error_msg = f"Evaluation failed: {str(e)}"
            print(f"❌ {error_msg}")

            # ✅ Update your local DB on failure
            if self.current_evaluation_run_id:
                await self.local_db.update_evaluation_run(
                    self.current_evaluation_run_id,
                    status="failed"
                )

            return {
                "status": "failed",
                "error": error_msg,
                "session_id": self.session_id or "unknown"
            }

        finally:
            self.evaluation_active = False

    async def evaluate_from_images(self, image_directory: str) -> Dict[str, Any]:
        """
        Evaluate pipeline using pre-captured images.
        (Left mostly unchanged aside from using ADK session_id).
        """
        if not self.session_id:
            print("❌ System not initialized. Call initialize() first.")
            return {"error": "System not initialized"}

        if not os.path.exists(image_directory):
            return {"error": f"Image directory not found: {image_directory}"}

        print(f"🖼️ Starting image directory evaluation: {image_directory}")

        try:
            image_files = [f for f in os.listdir(image_directory)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if not image_files:
                return {"error": "No image files found in directory"}

            print(f"📸 Found {len(image_files)} images to process")

            # Process images in batches via agent tools
            for i, image_file in enumerate(image_files):
                image_path = os.path.join(image_directory, image_file)
                frame_data = {"frame_id": i + 1, "image_path": image_path, "timestamp": time.time()}

                query = f"""Please analyze this medical image frame.

Frame data: {json.dumps(frame_data, indent=2)}

Use the 'analyze_pipeline_frame' tool to:
1. Extract facial landmarks from the image
2. Calculate comprehensive medical metrics
3. Assess facial symmetry and asymmetries
4. Generate severity scores and clinical indicators
5. Save detailed analysis results

Provide thorough medical analysis for this frame."""
                _ = await self._call_agent(query, frame_data)

                if (i + 1) % 10 == 0:
                    print(f"📊 Processed {i + 1}/{len(image_files)} images")

            # Final combined analysis
            analysis_query = f"""Please generate comprehensive performance metrics and medical assessment for {len(image_files)} analyzed images.

Use the following tools in sequence:
1. 'calculate_performance_metrics' - analyze overall performance
2. 'generate_medical_assessment' - create clinical evaluation
3. 'save_evaluation_report' - generate final report

Focus on statistical analysis and clinical insights across all images."""
            final_analysis = await self._call_agent(analysis_query)

            return {
                "status": "completed",
                "session_id": self.session_id,
                "images_processed": len(image_files),
                "image_directory": image_directory,
                "final_analysis": final_analysis
            }

        except Exception as e:
            return {"status": "failed", "error": str(e), "session_id": self.session_id}

    async def _call_agent(self, query: str, additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Call the agent with a query and optional context.
        Uses ADK Runner with the ADK session service + session_id.
        """
        try:
            enhanced_query = (
                f"{query}\n\nAdditional Context:\n" +
                json.dumps(additional_context or {}, indent=2,
                           default=lambda o: f"<non-serializable:{type(o).__name__}>")
            )

            content = types.Content(role='user', parts=[types.Part(text=enhanced_query)])

            final_response = "No response received from agent"

            # ✅ The runner now uses the ADK session service + session_id (valid types)
            async for event in self.runner.run_async(
                user_id=USER_ID,
                session_id=self.session_id,
                new_message=content
            ):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        final_response = event.content.parts[0].text
                    break

            return final_response

        except Exception as e:
            error_msg = f"Agent call failed: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

    def get_session_summary(self) -> Dict[str, Any]:
        """Return a simple snapshot about the current session (local view)."""
        if not self.session_id:
            return {"error": "No active session"}

        # We keep a local snapshot from initialize(); agent tools may update state internally,
        # but for a quick summary this is sufficient without querying ADK again.
        state = self.session_state_snapshot or {}
        return {
            "session_id": self.session_id,
            "project_id": self.project_id,
            "model": self.model,
            "evaluation_run_id": self.current_evaluation_run_id,
            "evaluation_active": self.evaluation_active,
            "session_state": state
        }


# ------------------------------ Interactive CLI ------------------------------

async def run_interactive_evaluation():
    """Interactive launcher for medical evaluation"""
    print("🏥 Medical Pipeline Evaluator - Interactive Mode")
    print("=" * 70)

    evaluator = LocalMedicalPipelineEvaluator()

    if not await evaluator.initialize():
        print("❌ Failed to initialize evaluator")
        return

    while True:
        print("\n🎯 Evaluation Options:")
        print("1. 📹 Camera Evaluation (5 minutes)")
        print("2. 📹 Camera Evaluation (Custom duration)")
        print("3. 🖼️  Image Directory Evaluation")
        print("4. ⚙️  Quick Performance Test (1 minute)")
        print("5. 📊 Session Summary")
        print("6. 🧹 Cleanup Session Files")
        print("7. 📋 View Database Statistics")
        print("8. ❌ Exit")

        try:
            choice = input("\nSelect option (1-8): ").strip()

            if choice == "1":
                print("📹 Starting 5-minute camera evaluation...")
                result = await evaluator.evaluate_from_camera(duration_seconds=300)
                display_evaluation_results(result)

            elif choice == "2":
                try:
                    duration = int(input("Enter duration in seconds: "))
                    if duration < 10 or duration > 3600:
                        print("❌ Duration must be between 10 and 3600 seconds")
                        continue
                    print(f"📹 Starting {duration}-second camera evaluation...")
                    result = await evaluator.evaluate_from_camera(duration_seconds=duration)
                    display_evaluation_results(result)
                except ValueError:
                    print("❌ Invalid duration. Please enter a number.")
                    continue

            elif choice == "3":
                image_dir = input("Enter image directory path: ").strip()
                if not os.path.exists(image_dir):
                    print(f"❌ Directory not found: {image_dir}")
                    continue
                print(f"🖼️ Starting image directory evaluation...")
                result = await evaluator.evaluate_from_images(image_dir)
                display_evaluation_results(result)

            elif choice == "4":
                print("⚙️ Starting quick performance test...")
                result = await evaluator.evaluate_from_camera(duration_seconds=60)
                display_evaluation_results(result)

            elif choice == "5":
                summary = evaluator.get_session_summary()
                display_session_summary(summary)

            elif choice == "6":
                keep_reports = input("Keep report files? (y/n): ").strip().lower() == 'y'
                # If you wire a cleanup tool, you can call it here via _call_agent(...)
                print(f"🧹 Cleanup (placeholder). Keep reports: {keep_reports}")

            elif choice == "7":
                # If you want to keep using sqlite_storage stats view:
                stats = evaluator.sqlite_storage.get_database_statistics()
                display_database_stats(stats)

            elif choice == "8":
                print("👋 Exiting evaluation system...")
                break

            else:
                print("❌ Invalid choice. Please select 1-8.")

        except KeyboardInterrupt:
            print("\n⏸️ Operation cancelled by user")
            continue
        except Exception as e:
            print(f"❌ Error: {e}")
            continue


def display_evaluation_results(results: Dict[str, Any]):
    print("\n" + "🏆 EVALUATION RESULTS 🏆")
    print("=" * 50)
    if results.get("status") == "completed":
        print(f"✅ Status: {results['status'].upper()}")
        print(f"📊 Session: {results.get('session_id', 'Unknown')}")
        print(f"⏱️ Duration: {results.get('total_duration_seconds', 0):.1f} seconds")
        if 'evaluation_run_id' in results:
            print(f"📋 Run ID: {results['evaluation_run_id']}")
        print(f"🎯 Analysis Type: {results.get('analysis_type', 'comprehensive')}")
        print(f"✅ Steps Completed: {results.get('steps_completed', 'Unknown')}")
    elif results.get("status") == "failed":
        print(f"❌ Status: FAILED")
        print(f"🚨 Error: {results.get('error', 'Unknown error')}")
    else:
        print(f"⚠️ Status: {results.get('status', 'Unknown')}")
    print("=" * 50)


def display_session_summary(summary: Dict[str, Any]):
    print("\n" + "📊 SESSION SUMMARY 📊")
    print("=" * 40)
    if "error" in summary:
        print(f"❌ {summary['error']}")
        return
    print(f"🆔 Session ID: {summary.get('session_id', 'Unknown')}")
    print(f"🏷️ Project: {summary.get('project_id', 'Unknown')}")
    print(f"🤖 Model: {summary.get('model', 'Unknown')}")
    print(f"📋 Run ID: {summary.get('evaluation_run_id', 'None')}")
    print(f"🔄 Active: {'Yes' if summary.get('evaluation_active', False) else 'No'}")
    state = summary.get('session_state', {})
    if state:
        print(f"📊 Frames Analyzed: {state.get('frames_analyzed', 0)}")
        print(f"🕐 Started: {state.get('evaluation_started', 'Unknown')}")
    print("=" * 40)


def display_database_stats(stats: Dict[str, Any]):
    print("\n" + "🗄️ DATABASE STATISTICS 🗄️")
    print("=" * 45)
    if not stats:
        print("❌ No statistics available")
        return
    print(f"📊 Sessions: {stats.get('sessions_count', 0)}")
    print(f"🔬 Evaluation Runs: {stats.get('evaluation_runs_count', 0)}")
    print(f"🖼️ Frame Analyses: {stats.get('frame_analyses_count', 0)}")
    print(f"📈 Performance Metrics: {stats.get('performance_metrics_count', 0)}")
    print(f"🤖 Agent Interactions: {stats.get('agent_interactions_count', 0)}")
    print(f"📋 Reports: {stats.get('reports_count', 0)}")
    if 'recent_evaluations' in stats:
        print(f"🕐 Recent Evaluations (24h): {stats['recent_evaluations']}")
    if 'performance' in stats:
        perf = stats['performance']
        print(f"📊 Avg Score: {perf.get('average_overall_score', 0):.3f}")
        print(f"🏆 Best Score: {perf.get('best_score', 0):.3f}")
    if 'database_size_mb' in stats:
        print(f"💾 Database Size: {stats['database_size_mb']} MB")
    print("=" * 45)


# ------------------------------ Main entrypoint ------------------------------

async def main():
    print("🏥 Medical Pipeline Evaluator with ADK")
    print("Project: mindfulfocus-470008")
    print("Model: Gemini 2.5 Pro")
    print("=" * 60)

    if not os.getenv("GOOGLE_API_KEY"):
        api_key = input("🔑 Enter your Gemini API key: ").strip()
        if not api_key:
            print("❌ API key required. Exiting.")
            return
        os.environ["GOOGLE_API_KEY"] = api_key

    try:
        await run_interactive_evaluation()
    except KeyboardInterrupt:
        print("\n👋 Evaluation system shutdown complete")
    except Exception as e:
        print(f"❌ System error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
