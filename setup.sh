#!/bin/bash
# setup.sh - Complete setup for Local Medical Pipeline Evaluator with ADK

echo "🏥 Medical Pipeline Evaluator with ADK - Complete Setup"
echo "Project ID: mindfulfocus-470008"
echo "Model: Gemini 2.5 Pro"
echo "=" * 70

# Set project variables
PROJECT_ID="mindfulfocus-470008"
MODEL_NAME="gemini-1.5-pro"  # Using 1.5-pro as 2.5-pro might not be available yet

# Navigate to project directory
cd E:/MoleProject/MindfulFocus/

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p adk_medical_evaluation/agents
mkdir -p adk_medical_evaluation/tools
mkdir -p adk_medical_evaluation/local_storage
mkdir -p adk_medical_evaluation/config
mkdir -p evaluation_data/sessions
mkdir -p evaluation_data/images
mkdir -p evaluation_data/reports
mkdir -p evaluation_data/metrics
mkdir -p evaluation_data/logs
mkdir -p local_database

# Create __init__.py files
touch adk_medical_evaluation/__init__.py
touch adk_medical_evaluation/agents/__init__.py
touch adk_medical_evaluation/tools/__init__.py
touch adk_medical_evaluation/local_storage/__init__.py
touch adk_medical_evaluation/config/__init__.py

echo "✅ Setup script complete!"
echo "📝 Next steps:"
echo "1. Add your Gemini API key to the main_evaluator.py file"
echo "2. Run: python adk_medical_evaluation/main_evaluator.py"
echo "3. Or use the interactive launcher: python launcher.py"