"""
Updated Medical Pipeline Evaluator using Gemini from Vertex AI
"""

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import asyncio
import vertexai
from vertexai.generative_models import GenerativeModel

# Agent Garden imports (conceptual - adjust based on actual SDK)
from agent_garden import Agent, Tool, SearchTool, DatabaseTool, FunctionTool

@dataclass
class EvaluationMetrics:
    accuracy_score: float
    precision: float
    recall: float
    latency_ms: float
    landmarks_detected: int
    false_positive_rate: float
    clinical_relevance_score: float
    gemini_confidence_avg: float

class GeminiMedicalPipelineEvaluatorAgent(Agent):
    """Medical pipeline evaluator using Gemini for AI analysis"""
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        super().__init__()
        self.project_id = project_id
        self.location = location
        
        # Initialize Vertex AI and Gemini
        vertexai.init(project=project_id, location=location)
        self.gemini_model = GenerativeModel("gemini-1.5-pro")
        
        self.setup_tools()
        self.evaluation_history = []
        self.performance_baseline = None
        
    def setup_tools(self):
        """Initialize Agent Garden tools"""
        # Search tool for medical validation
        self.search_tool = SearchTool(
            name="medical_search",
            description="Search for medical literature and validation data"
        )
        
        # Database tool for storing evaluations
        self.db_tool = DatabaseTool(
            name="evaluation_db",
            connection_type="cloud_sql_postgresql",
            description="Store and query evaluation results"
        )
        
        # Function calling for custom analysis
        self.function_tool = FunctionTool(
            name="custom_analysis",
            description="Execute custom evaluation functions"
        )
        
        # Add tools to agent
        self.add_tool(self.search_tool)
        self.add_tool(self.db_tool)
        self.add_tool(self.function_tool)
    
    async def evaluate_pipeline(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main evaluation function using Gemini for analysis"""
        print("🤖 Starting Gemini-powered pipeline evaluation...")
        
        # 1. Performance Analysis with Gemini insights
        performance_metrics = await self.analyze_performance_with_gemini(pipeline_data)
        
        # 2. Medical Accuracy Validation using Gemini
        medical_validation = await self.validate_medical_accuracy_gemini(pipeline_data)
        
        # 3. Real-time Monitoring
        monitoring_results = await self.monitor_realtime_performance(pipeline_data)
        
        # 4. Gemini-based Component Analysis
        component_analysis = await self.analyze_components_with_gemini(pipeline_data)
        
        # 5. Clinical Relevance Assessment with Gemini
        clinical_assessment = await self.assess_clinical_relevance_gemini(pipeline_data)
        
        # 6. Generate Comprehensive Report using Gemini
        evaluation_report = await self.generate_gemini_evaluation_report({
            'performance': performance_metrics,
            'medical_validation': medical_validation,
            'monitoring': monitoring_results,
            'components': component_analysis,
            'clinical': clinical_assessment
        })
        
        # Store results
        await self.store_evaluation_results(evaluation_report)
        
        return evaluation_report

    async def analyze_performance_with_gemini(self, pipeline_data: Dict[str, Any]) -> EvaluationMetrics:
        """Analyze performance metrics with Gemini insights"""
        print("📊 Analyzing performance with Gemini...")
        
        # Basic performance calculation
        landmark_metrics = pipeline_data.get('metrics', {})
        processing_times = pipeline_data.get('processing_times', [])
        frame_results = pipeline_data.get('frame_results', [])
        
        # Calculate basic metrics
        total_frames = len(frame_results)
        successful_detections = sum(1 for frame in frame_results if frame.get('landmarks_detected', 0) > 400)
        accuracy = successful_detections / total_frames if total_frames > 0 else 0
        
        symmetry_scores = [frame.get('symmetry_score', 0) for frame in frame_results]
        precision = max(0, 1 - np.std(symmetry_scores)) if symmetry_scores else 0
        
        recall = np.mean([min(1.0, frame.get('landmarks_detected', 0) / 468) for frame in frame_results])
        avg_latency = np.mean(processing_times) if processing_times else 0
        
        # Calculate Gemini confidence average
        gemini_confidences = [frame.get('gemini_confidence', 0) for frame in frame_results if 'gemini_confidence' in frame]
        gemini_confidence_avg = np.mean(gemini_confidences) if gemini_confidences else 0.0
        
        # Use Gemini for advanced performance analysis
        performance_analysis_prompt = f"""
        Analyze this medical face analysis pipeline performance data:
        
        Total Frames: {total_frames}
        Successful Detections: {successful_detections}
        Accuracy Rate: {accuracy:.3f}
        Average Processing Time: {avg_latency:.3f}s
        Symmetry Score Variance: {np.std(symmetry_scores) if symmetry_scores else 0:.4f}
        Average Gemini Confidence: {gemini_confidence_avg:.3f}
        
        Frame Results Sample: {json.dumps(frame_results[:5], indent=2)}
        
        Provide insights on:
        1. Overall performance assessment
        2. Bottlenecks and optimization opportunities
        3. Reliability and consistency evaluation
        4. Recommendations for improvement
        
        Focus on technical performance aspects.
        """
        
        try:
            gemini_analysis = self.gemini_model.generate_content(performance_analysis_prompt)
            print(f"🧠 Gemini Performance Analysis: {gemini_analysis.text[:200]}...")
        except Exception as e:
            print(f"⚠️ Gemini analysis failed: {e}")
            gemini_analysis = None
        
        return EvaluationMetrics(
            accuracy_score=accuracy,
            precision=precision,
            recall=recall,
            latency_ms=avg_latency * 1000,
            landmarks_detected=np.mean([frame.get('landmarks_detected', 0) for frame in frame_results]),
            false_positive_rate=self._calculate_false_positive_rate(frame_results),
            clinical_relevance_score=accuracy * precision * recall,
            gemini_confidence_avg=gemini_confidence_avg
        )

    async def validate_medical_accuracy_gemini(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate medical accuracy using Gemini and literature search"""
        print("🏥 Validating medical accuracy with Gemini...")
        
        # Search for medical validation standards
        search_results = await self.search_tool.search(
            "facial asymmetry detection medical accuracy sensitivity specificity validation"
        )
        
        # Use Gemini for comprehensive medical validation
        medical_validation_prompt = f"""
        As a medical AI expert, validate this facial analysis pipeline against clinical standards:
        
        Pipeline Metrics: {json.dumps(pipeline_data.get('metrics', {}), indent=2)}
        
        Medical Literature Context: {search_results}
        
        Sample Results: {json.dumps(pipeline_data.get('frame_results', [])[:3], indent=2)}
        
        Evaluate:
        1. Clinical sensitivity and specificity of the landmark detection
        2. Appropriateness of asymmetry thresholds for medical diagnosis
        3. Reliability for detecting conditions like Bell's palsy
        4. Comparison with established medical imaging standards
        5. False positive/negative risk assessment
        6. Regulatory and clinical deployment considerations
        
        Provide detailed medical validation assessment with specific recommendations.
        """
        
        try:
            medical_assessment = self.gemini_model.generate_content(medical_validation_prompt)
            validation_text = medical_assessment.text
        except Exception as e:
            validation_text = f"Gemini medical validation failed: {e}"
        
        return {
            'medical_literature_search': search_results,
            'gemini_clinical_validation': validation_text,
            'validation_score': self._extract_validation_score(validation_text),
            'clinical_recommendations': self._extract_clinical_recommendations(validation_text),
            'regulatory_notes': self._extract_regulatory_considerations(validation_text)
        }

    async def analyze_components_with_gemini(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pipeline components with Gemini insights"""
        print("🔧 Analyzing components with Gemini...")
        
        component_analysis_prompt = f"""
        Analyze the individual components of this medical face analysis pipeline:
        
        Pipeline Data: {json.dumps(pipeline_data, indent=2, default=str)[:3000]}...
        
        Component Analysis Required:
        1. **MediaPipe Landmark Analyzer**:
           - Accuracy and reliability
           - Processing speed
           - Edge case handling
        
        2. **Facial Metrics Calculator**:
           - Mathematical accuracy of symmetry calculations
           - EAR (Eye Aspect Ratio) reliability
           - Clinical relevance of metrics
        
        3. **Medical Assessment Logic**:
           - Severity scoring appropriateness
           - Threshold validation
           - Clinical correlation
        
        4. **Integration Architecture**:
           - System reliability
           - Error handling
           - Performance optimization opportunities
        
        Provide specific technical recommendations for each component.
        """
        
        try:
            component_analysis = self.gemini_model.generate_content(component_analysis_prompt)
            analysis_text = component_analysis.text
        except Exception as e:
            analysis_text = f"Component analysis failed: {e}"
        
        return {
            'gemini_component_analysis': analysis_text,
            'landmark_analyzer_score': self._extract_component_score(analysis_text, 'landmark'),
            'metrics_calculator_score': self._extract_component_score(analysis_text, 'metrics'),
            'medical_logic_score': self._extract_component_score(analysis_text, 'medical'),
            'integration_score': self._extract_component_score(analysis_text, 'integration'),
            'improvement_recommendations': self._extract_improvement_recommendations(analysis_text)
        }

    async def assess_clinical_relevance_gemini(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess clinical relevance using Gemini"""
        print("🏥 Assessing clinical relevance with Gemini...")
        
        clinical_search = await self.search_tool.search(
            "facial asymmetry clinical diagnosis Bell's palsy stroke detection medical imaging"
        )
        
        clinical_assessment_prompt = f"""
        Assess the clinical relevance and medical utility of this facial analysis system:
        
        System Metrics: {json.dumps(pipeline_data.get('metrics', {}), indent=2)}
        Clinical Literature: {clinical_search}
        Performance Data: {json.dumps(pipeline_data.get('evaluation_summary', {}), indent=2)}
        
        Clinical Assessment Required:
        1. **Diagnostic Utility**: How useful is this system for actual medical diagnosis?
        2. **Clinical Workflow Integration**: How would this fit into medical practice?
        3. **Sensitivity/Specificity**: What are the expected clinical performance metrics?
        4. **Use Case Validation**: What medical conditions can this reliably detect?
        5. **Limitations and Contraindications**: What are the system's medical limitations?
        6. **Training Requirements**: What medical training would users need?
        7. **Regulatory Pathway**: What would be required for medical device approval?
        
        Provide detailed clinical assessment with deployment recommendations.
        """
        
        try:
            clinical_assessment = self.gemini_model.generate_content(clinical_assessment_prompt)
            assessment_text = clinical_assessment.text
        except Exception as e:
            assessment_text = f"Clinical assessment failed: {e}"
        
        return {
            'clinical_literature': clinical_search,
            'gemini_clinical_assessment': assessment_text,
            'diagnostic_utility_score': self._extract_diagnostic_score(assessment_text),
            'clinical_workflow_score': self._extract_workflow_score(assessment_text),
            'regulatory_readiness': self._extract_regulatory_readiness(assessment_text),
            'recommended_use_cases': self._extract_use_cases(assessment_text),
            'medical_limitations': self._extract_limitations(assessment_text)
        }

    async def generate_gemini_evaluation_report(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report using Gemini"""
        print("📝 Generating Gemini evaluation report...")
        
        report_generation_prompt = f"""
        Generate a comprehensive technical and clinical evaluation report for this medical face analysis pipeline:
        
        {json.dumps(evaluation_data, indent=2, default=str)[:4000]}...
        
        Report Structure Required:
        
        # Executive Summary
        - Overall assessment and recommendation
        - Key findings and scores
        - Deployment readiness
        
        # Technical Performance Analysis
        - Accuracy, precision, recall metrics
        - Processing performance
        - Reliability assessment
        
        # Medical Validation
        - Clinical accuracy assessment
        - Comparison with medical standards
        - Sensitivity/specificity analysis
        
        # Component Analysis
        - Individual component
        # Component Analysis
        - Individual component performance
        - Integration effectiveness
        - Optimization opportunities

        # Clinical Relevance Assessment
        - Medical utility evaluation
        - Use case recommendations
        - Clinical workflow integration

        # Risk Assessment
        - Medical risks and limitations
        - False positive/negative implications
        - Safety considerations

        # Regulatory Considerations
        - Medical device classification
        - Approval pathway requirements
        - Compliance recommendations

        # Recommendations and Next Steps
        - Immediate improvements needed
        - Long-term development roadmap
        - Deployment strategy

        Provide specific, actionable recommendations with technical details.
        """

        try:
            report_response = self.gemini_model.generate_content(report_generation_prompt)
            report_content = report_response.text
        except Exception as e:
            report_content = f"Report generation failed: {e}"

        # Create structured report
        evaluation_report = {
            'report_id': f"gemini_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'pipeline_version': evaluation_data.get('version', '1.0'),
            'gemini_model': "gemini-1.5-pro",
            'evaluation_summary': {
                'overall_score': self.calculate_overall_score(evaluation_data),
                'deployment_ready': self.assess_deployment_readiness(evaluation_data),
                'critical_issues': self.extract_critical_issues(evaluation_data),
                'gemini_confidence': self._calculate_gemini_confidence(evaluation_data)
            },
            'detailed_analysis': evaluation_data,
            'gemini_report_content': report_content,
            'actionable_recommendations': self._extract_actionable_recommendations(report_content),
            'technical_improvements': self._extract_technical_improvements(report_content),
            'clinical_deployment_plan': self._extract_deployment_plan(report_content)
        }

        return evaluation_report

    # Helper methods for Gemini response parsing
    def _extract_validation_score(self, text: str) -> float:
        """Extract validation score from Gemini response"""
        # Look for score indicators in text
        score_patterns = ['score:', 'rating:', '/10', '%']
        for line in text.split('\n'):
            line_lower = line.lower()
            for pattern in score_patterns:
                if pattern in line_lower:
                    # Extract numerical score
                    import re
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        score = float(numbers[0])
                        return min(1.0, score / 10 if score > 1 else score)
        return 0.7  # Default moderate score

    def _extract_clinical_recommendations(self, text: str) -> List[str]:
        """Extract clinical recommendations from Gemini response"""
        recommendations = []
        lines = text.split('\n')
        in_recommendations = False
        
        for line in lines:
            line = line.strip()
            if any(word in line.lower() for word in ['recommend', 'suggest', 'advise']):
                in_recommendations = True
            if in_recommendations and line and len(line) > 20:
                if line.startswith('-') or line.startswith('•') or line.startswith('1.'):
                    recommendations.append(line.lstrip('-•1234567890. '))
                elif not any(word in line.lower() for word in ['conclusion', 'summary']):
                    recommendations.append(line)
                
        return recommendations[:8]  # Limit to top 8

    def _extract_component_score(self, text: str, component: str) -> float:
        """Extract component-specific score from analysis"""
        component_section = ""
        lines = text.split('\n')
        in_component = False
        
        for line in lines:
            if component.lower() in line.lower():
                in_component = True
            elif in_component and line.strip() == "":
                break
            elif in_component:
                component_section += line + " "
        
        # Look for score indicators
        if any(word in component_section.lower() for word in ['excellent', 'outstanding']):
            return 0.9
        elif any(word in component_section.lower() for word in ['good', 'solid', 'reliable']):
            return 0.8
        elif any(word in component_section.lower() for word in ['adequate', 'acceptable']):
            return 0.7
        elif any(word in component_section.lower() for word in ['poor', 'inadequate', 'concerning']):
            return 0.4
        else:
            return 0.6  # Default moderate

    def _calculate_gemini_confidence(self, evaluation_data: Dict[str, Any]) -> float:
        """Calculate overall Gemini confidence from evaluation"""
        confidences = []
        
        # Extract confidence from performance analysis
        performance = evaluation_data.get('performance', {})
        if isinstance(performance, dict) and hasattr(performance, 'gemini_confidence_avg'):
            confidences.append(performance.gemini_confidence_avg)
        
        # Extract from medical validation
        medical = evaluation_data.get('medical_validation', {})
        if isinstance(medical, dict):
            validation_score = medical.get('validation_score', 0)
            confidences.append(validation_score)
        
        return np.mean(confidences) if confidences else 0.6

    def _extract_actionable_recommendations(self, report_content: str) -> List[str]:
        """Extract actionable recommendations from Gemini report"""
        recommendations = []
        lines = report_content.split('\n')
        
        # Look for recommendation sections
        in_recommendations = False
        for line in lines:
            line = line.strip()
            
            if any(header in line.lower() for header in ['recommendations', 'next steps', 'improvements']):
                in_recommendations = True
                continue
            
            if in_recommendations:
                if line.startswith('#') and 'recommend' not in line.lower():
                    in_recommendations = False
                elif line and len(line) > 15:
                    if line.startswith(('-', '•', '*')) or line[0].isdigit():
                        clean_rec = line.lstrip('-•*1234567890. ').strip()
                        if clean_rec and len(clean_rec) > 10:
                            recommendations.append(clean_rec)
        
        return recommendations[:10]  # Top 10 recommendations

    def _extract_technical_improvements(self, report_content: str) -> List[str]:
        """Extract technical improvements from report"""
        improvements = []
        lines = report_content.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(word in line.lower() for word in ['optimize', 'improve', 'enhance', 'upgrade']):
                if len(line) > 20 and not line.startswith('#'):
                    improvements.append(line.lstrip('-•*1234567890. '))
        
        return improvements[:8]

    def _extract_deployment_plan(self, report_content: str) -> Dict[str, Any]:
        """Extract deployment plan from report"""
        plan = {
            'immediate_actions': [],
            'short_term_goals': [],
            'long_term_objectives': [],
            'success_metrics': []
        }
        
        lines = report_content.split('\n')
        current_category = None
        
        for line in lines:
            line = line.strip()
            
            if 'immediate' in line.lower() or 'urgent' in line.lower():
                current_category = 'immediate_actions'
            elif 'short' in line.lower() and 'term' in line.lower():
                current_category = 'short_term_goals'
            elif 'long' in line.lower() and 'term' in line.lower():
                current_category = 'long_term_objectives'
            elif 'metric' in line.lower() or 'measure' in line.lower():
                current_category = 'success_metrics'
            elif current_category and line and len(line) > 10:
                if line.startswith(('-', '•', '*')) or line[0].isdigit():
                    clean_item = line.lstrip('-•*1234567890. ').strip()
                    if clean_item:
                        plan[current_category].append(clean_item)
        
        return plan

    def _calculate_false_positive_rate(self, frame_results: List[Dict]) -> float:
        """Calculate false positive rate from frame results"""
        if not frame_results:
            return 0.0
        
        false_positives = 0
        total_frames = len(frame_results)
        
        for frame in frame_results:
            # Consider it a false positive if high severity but good symmetry
            severity = frame.get('severity_score', 0)
            symmetry = frame.get('symmetry_score', 1.0)
            
            if severity > 6 and symmetry > 0.85:
                false_positives += 1
        
        return false_positives / total_frames if total_frames > 0 else 0.0