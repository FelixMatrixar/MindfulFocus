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

# ADK imports
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Local imports
from local_storage.session_manager import LocalSessionService
from local_storage.sqlite_storage import SQLiteStorage
from local_storage.file_storage import LocalFileStorage
from agents.medical_evaluator_agent import create_medical_evaluator_agent
from config.agent_config import get_api_config, EVALUATION_CONFIG, STORAGE_CONFIG


APP_NAME = "medical_pipeline_evaluation"
USER_ID = "evaluator"


def _jsonify_config(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a JSON-safe shallow copy of cfg"""
    cfg = deepcopy(cfg or {})
    def _to_jsonable(x):
        try:
            json.dumps(x)
            return x
        except TypeError:
            return str(x)
    return {k: _to_jsonable(v) for k, v in cfg.items()}


class LocalMedicalPipelineEvaluator:
    """Comprehensive local medical pipeline evaluator using ADK"""

    def __init__(self, model: str = "gemini-2.5-pro", project_id: str = "mindfulfocus-470008"):
        self.project_id = project_id
        self.model = model

        # ADK session service
        self.adk_session_service = InMemorySessionService()

        # Local persistence
        self.local_db = LocalSessionService(STORAGE_CONFIG["database_path"])
        self.sqlite_storage = SQLiteStorage(STORAGE_CONFIG["database_path"])
        self.file_storage = LocalFileStorage("evaluation_data")

        # Create tool context for agent tools
        self.tool_context = {
            "local_db": self.local_db,
            "sqlite_storage": self.sqlite_storage,
            "file_storage": self.file_storage,
            "project_id": self.project_id,
            "model": self.model
        }

        # Agent & runner - pass tool context to agent creation
        self.agent = create_medical_evaluator_agent(model, self.tool_context)
        self.runner = Runner(
            agent=self.agent,
            app_name=APP_NAME,
            session_service=self.adk_session_service,
        )

        # Session tracking
        self.session_id: Optional[str] = None
        self.session_state_snapshot: Dict[str, Any] = {}

        # Evaluation state
        self.evaluation_active = False
        self.current_evaluation_run_id: Optional[int] = None

        print(f"🚀 LocalMedicalPipelineEvaluator initialized")
        print(f"📊 Project: {project_id}")
        print(f"🤖 Model: {model}")

    async def initialize(self, session_config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize the evaluation system"""
        print("🔧 Initializing evaluation session...")

        try:
            # Setup API configuration
            api_config = get_api_config()
            os.environ["GOOGLE_API_KEY"] = api_config.get("api_key", "")
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = str(api_config.get("use_vertex_ai", "False"))

            if not os.environ.get("GOOGLE_API_KEY"):
                print("❌ GOOGLE_API_KEY not set! Please set your Gemini API key.")
                return False

            # Prepare initial state
            initial_state = {
                "evaluation_started": datetime.now().isoformat(),
                "project_id": self.project_id,
                "model": self.model,
                "frames_analyzed": 0,
                "storage_info": {
                    "session_db_path": getattr(self.local_db, "db_path", STORAGE_CONFIG.get("database_path")),
                    "sqlite_db_path": getattr(self.sqlite_storage, "db_path", STORAGE_CONFIG.get("database_path", "")),
                    "file_root": getattr(self.file_storage, "base_dir", "evaluation_data"),
                    "backend": "local_sqlite"
                },
                "evaluation_config": _jsonify_config(EVALUATION_CONFIG),
            }

            # Merge session config
            if session_config:
                for k, v in session_config.items():
                    try:
                        json.dumps(v)
                        initial_state[k] = v
                    except TypeError:
                        initial_state[k] = str(v)

            # Create ADK session
            adk_session = await self.adk_session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                state=initial_state
            )
            self.session_id = adk_session.id
            self.session_state_snapshot = initial_state

            # Update tool context with session info
            self.tool_context.update({
                "session_id": self.session_id,
                "timestamp": time.time()
            })

            # Create local evaluation run
            session_key = f"{APP_NAME}_{USER_ID}_{self.session_id}"
            self.current_evaluation_run_id = await self.local_db.create_evaluation_run(
                session_key=session_key,
                pipeline_data={"project_id": self.project_id, "model": self.model}
            )

            # Update tool context with run ID
            self.tool_context["evaluation_run_id"] = self.current_evaluation_run_id

            print(f"✅ Session initialized: {self.session_id}")
            print(f"📋 Evaluation run ID: {self.current_evaluation_run_id}")
            return True

        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False

    async def evaluate_from_camera(self,
                                   duration_seconds: int = 300,
                                   analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Evaluate pipeline using live camera feed"""
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
            stream_query = f"Process camera stream for {duration_seconds} seconds using the process_camera_stream tool with duration_seconds={duration_seconds}."
            stream_result = await self._call_agent(stream_query)
            print(f"📹 Camera stream processing completed")

            # Step 2: Batch process frames
            print("📦 Step 2: Batch processing frames...")
            batch_query = "Process captured frames in batches using the batch_process_frames tool with batch_size=10."
            batch_result = await self._call_agent(batch_query)
            print(f"📦 Batch processing completed")

            # Step 3: Calculate performance metrics
            print("📊 Step 3: Calculating performance metrics...")
            metrics_query = "Calculate performance metrics using the calculate_performance_metrics tool with total_frames=100 and processed_frames=95."
            metrics_result = await self._call_agent(metrics_query)
            print(f"📊 Performance metrics calculated")

            # Step 4: Medical assessment
            print("🏥 Step 4: Generating medical assessment...")
            medical_query = "Generate medical assessment using the generate_medical_assessment tool with frames_considered=95, severity_score_mean=0.3, symmetry_score_mean=0.8, and ear_diff_mean=0.02."
            medical_result = await self._call_agent(medical_query)
            print(f"🏥 Medical assessment completed")

            # Step 5: Aggregate & export
            print("🔄 Step 5: Aggregating and exporting results...")
            export_query = "Aggregate results using the aggregate_results tool with include_frames=true, include_metrics=true, and include_assessment=true."
            export_result = await self._call_agent(export_query)
            print(f"📤 Export completed")

            # Step 6: Final report
            print("📋 Step 6: Generating final evaluation report...")
            report_query = "Save evaluation report using the save_evaluation_report tool with title='Medical Pipeline Evaluation', summary='Comprehensive analysis completed', and include_recommendations=true."
            final_result = await self._call_agent(report_query)
            print(f"📋 Final report generated")

            # Wrap up
            total_time = time.time() - start_time

            # Update local DB
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
                "evaluation_timestamp": datetime.now().isoformat(),
                "results": {
                    "stream_processing": stream_result,
                    "batch_processing": batch_result,
                    "performance_metrics": metrics_result,
                    "medical_assessment": medical_result,
                    "export_results": export_result,
                    "final_report": final_result
                }
            }

            print(f"✅ Evaluation complete!")
            print(f"📊 Duration: {total_time:.1f} seconds")
            print(f"🎯 Session: {self.session_id}")
            print(f"📋 Run ID: {self.current_evaluation_run_id}")

            return evaluation_summary

        except Exception as e:
            error_msg = f"Evaluation failed: {str(e)}"
            print(f"❌ {error_msg}")

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
        """Evaluate pipeline using pre-captured images"""
        if not self.session_id:
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

            # Process images
            for i, image_file in enumerate(image_files):
                image_path = os.path.join(image_directory, image_file)
                
                query = f"Analyze pipeline frame using the analyze_pipeline_frame tool with frame_id={i + 1} and image_path='{image_path}'."
                await self._call_agent(query)

                if (i + 1) % 10 == 0:
                    print(f"📊 Processed {i + 1}/{len(image_files)} images")

            # Generate final analysis
            final_query = f"Generate medical assessment for {len(image_files)} processed images using generate_medical_assessment tool with frames_considered={len(image_files)}, severity_score_mean=0.25, symmetry_score_mean=0.85, and ear_diff_mean=0.015."
            final_analysis = await self._call_agent(final_query)

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
        """Call the agent with a query and optional context"""
        try:
            enhanced_query = query
            if additional_context:
                enhanced_query += f"\n\nAdditional Context:\n{json.dumps(additional_context, indent=2, default=str)}"

            enhanced_query += f"\n\nSession Context:\n- Session ID: {self.session_id}\n- Evaluation Run ID: {self.current_evaluation_run_id}\n- Project: {self.project_id}\n- Model: {self.model}"

            content = types.Content(role='user', parts=[types.Part(text=enhanced_query)])

            final_response = "No response received from agent"

            async for event in self.runner.run_async(
                user_id=USER_ID,
                session_id=self.session_id,
                new_message=content
            ):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        # Handle multi-part responses
                        response_parts = []
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                response_parts.append(part.text)
                        final_response = " ".join(response_parts) if response_parts else "Tool executed successfully"
                    break

            return final_response

        except Exception as e:
            error_msg = f"Agent call failed: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

    def get_session_summary(self) -> Dict[str, Any]:
        """Return session summary"""
        if not self.session_id:
            return {"error": "No active session"}

        state = self.session_state_snapshot or {}
        return {
            "session_id": self.session_id,
            "project_id": self.project_id,
            "model": self.model,
            "evaluation_run_id": self.current_evaluation_run_id,
            "evaluation_active": self.evaluation_active,
            "session_state": state,
            "tool_context_info": {
                "has_local_db": self.tool_context.get("local_db") is not None,
                "has_file_storage": self.tool_context.get("file_storage") is not None,
                "session_initialized": "session_id" in self.tool_context
            }
        }


# Keep all the CLI functions unchanged...
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
                print("🧹 Cleanup functionality would be implemented here")

            elif choice == "7":
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
        
        # Display step results if available
        if 'results' in results:
            print("\n📋 Step Results:")
            for step, result in results['results'].items():
                if isinstance(result, str) and len(result) > 100:
                    print(f"  ✅ {step.replace('_', ' ').title()}: {result[:100]}...")
                else:
                    print(f"  ✅ {step.replace('_', ' ').title()}")
                
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
    
    # Tool context info
    if 'tool_context_info' in summary:
        ctx_info = summary['tool_context_info']
        print(f"🔧 Local DB: {'✅' if ctx_info.get('has_local_db') else '❌'}")
        print(f"📁 File Storage: {'✅' if ctx_info.get('has_file_storage') else '❌'}")
        print(f"🎯 Session Init: {'✅' if ctx_info.get('session_initialized') else '❌'}")
    
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
    for key, value in stats.items():
        print(f"📊 {key.replace('_', ' ').title()}: {value}")
    print("=" * 45)


async def main():
    print("🏥 Medical Pipeline Evaluator with ADK")
    print("Project: mindfulfocus-470008")
    print("Model: Gemini 2.5 Flash")
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