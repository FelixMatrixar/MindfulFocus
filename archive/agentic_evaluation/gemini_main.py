"""
Main execution script for Gemini-powered medical pipeline evaluation
"""

import asyncio
import json
import os
from datetime import datetime
import numpy as np
from gemini_pipeline_integration import GeminiPipelineEvaluationOrchestrator

# Configuration
PROJECT_ID = "your-gcp-project-id"  # Replace with your GCP project
LOCATION = "us-central1"

def display_gemini_results(evaluation_report):
    """Display Gemini evaluation results"""
    if not evaluation_report:
        print("❌ No results to display")
        return
    
    print("\n" + "🧠 GEMINI-POWERED EVALUATION RESULTS 🧠")
    print("=" * 70)
    
    summary = evaluation_report.get('evaluation_summary', {})
    print(f"📋 Report ID: {evaluation_report.get('report_id', 'N/A')}")
    print(f"🤖 Model Used: {evaluation_report.get('gemini_model', 'gemini-1.5-pro')}")
    print(f"📊 Overall Score: {summary.get('overall_score', 0):.3f}/1.0")
    print(f"🎯 Deployment Ready: {'✅ Yes' if summary.get('deployment_ready', False) else '❌ No'}")
    print(f"🧠 Gemini Confidence: {summary.get('gemini_confidence', 0):.3f}")
    print(f"⚠️ Critical Issues: {len(summary.get('critical_issues', []))}")
    
    # Show Gemini insights
    gemini_content = evaluation_report.get('gemini_report_content', '')
    if gemini_content:
        print(f"\n🧠 Gemini Analysis Preview:")
        print("─" * 50)
        print(gemini_content[:500] + "..." if len(gemini_content) > 500 else gemini_content)
    
    # Show actionable recommendations
    recommendations = evaluation_report.get('actionable_recommendations', [])
    if recommendations:
        print(f"\n🎯 Gemini Recommendations:")
        print("─" * 40)
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"  {i}. {rec}")
    
    # Show technical improvements
    improvements = evaluation_report.get('technical_improvements', [])
    if improvements:
        print(f"\n🔧 Technical Improvements:")
        print("─" * 40)
        for i, imp in enumerate(improvements[:3], 1):
            print(f"  {i}. {imp}")
    
    # Show deployment plan
    deployment_plan = evaluation_report.get('clinical_deployment_plan', {})
    if deployment_plan:
        print(f"\n🚀 Deployment Plan:")
        print("─" * 30)
        immediate = deployment_plan.get('immediate_actions', [])
        if immediate:
            print(f"  Immediate: {immediate[0] if immediate else 'None'}")
        
        short_term = deployment_plan.get('short_term_goals', [])
        if short_term:
            print(f"  Short-term: {short_term[0] if short_term else 'None'}")

async def main():
    """Main execution function"""
    print("🧠 Gemini-Powered Medical Pipeline Evaluation System")
    print("=" * 70)
    print(f"📍 Project: {PROJECT_ID}")
    print(f"🌍 Location: {LOCATION}")
    
    # Verify GCP setup
    if not PROJECT_ID or PROJECT_ID == "your-gcp-project-id":
        print("❌ Please set your GCP PROJECT_ID in the script")
        return
    
    try:
        orchestrator = GeminiPipelineEvaluationOrchestrator(PROJECT_ID, LOCATION)
        
        while True:
            print("\n🎯 Gemini Evaluation Options:")
            print("1. ⚡ Quick Evaluation (60s, minimal Gemini)")
            print("2. 🧠 Standard Evaluation (5min, moderate Gemini)")
            print("3. 🔬 Deep Evaluation (5min, extensive Gemini)")
            print("4. 📊 Check Gemini Status")
            print("5. ❌ Exit")
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == "1":
                print("⚡ Starting Quick Gemini Evaluation...")
                start_time = datetime.now()
                report = await orchestrator.run_quick_gemini_evaluation(60)
                
                if report:
                    print(f"⚡ Quick evaluation completed in {datetime.now() - start_time}")
                    display_gemini_results(report)
                    save_report(report, "quick")
                
            elif choice == "2":
                print("🧠 Starting Standard Gemini Evaluation...")
                start_time = datetime.now()
                report = await orchestrator.run_comprehensive_gemini_evaluation(300, use_detailed_analysis=False)
                
                if report:
                    print(f"🧠 Standard evaluation completed in {datetime.now() - start_time}")
                    display_gemini_results(report)
                    save_report(report, "standard")
                
            elif choice == "3":
                print("🔬 Starting Deep Gemini Evaluation...")
                start_time = datetime.now()
                report = await orchestrator.run_deep_gemini_evaluation(300)
                
                if report:
                    print(f"🔬 Deep evaluation completed in {datetime.now() - start_time}")
                    display_gemini_results(report)
                    save_report(report, "deep")
                
            elif choice == "4":
                await check_gemini_status(PROJECT_ID, LOCATION)
                
            elif choice == "5":
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice, please try again.")
            
            if choice in ["1", "2", "3"]:
                input("\nPress Enter to continue...")
    
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    except Exception as e:
        print(f"❌ Error: {e}")

def save_report(report, evaluation_type):
    """Save evaluation report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gemini_evaluation_{evaluation_type}_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"💾 Report saved: {filename}")
    except Exception as e:
        print(f"⚠️ Could not save report: {e}")

async def check_gemini_status(project_id, location):
    """Check Gemini/Vertex AI status"""
    print("🔍 Checking Gemini/Vertex AI status...")
    
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        
        # Test Gemini model
        model = GenerativeModel("gemini-1.5-pro")
        test_response = model.generate_content("Test connection - respond with 'OK'")
        
        print("✅ Gemini connection successful!")
        print(f"🤖 Model: gemini-1.5-pro")
        print(f"📍 Project: {project_id}")
        print(f"🌍 Location: {location}")
        print(f"💬 Test Response: {test_response.text[:50]}")
        
    except Exception as e:
        print(f"❌ Gemini connection failed: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Check GCP project ID and permissions")
        print("2. Enable Vertex AI API in your project")
        print("3. Set up authentication: gcloud auth application-default login")
        print("4. Install required packages: pip install google-cloud-aiplatform")

if __name__ == "__main__":
    asyncio.run(main())