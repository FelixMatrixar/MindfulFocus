import { GoogleGenerativeAI } from '@google/generative-ai';

class GeminiWellnessAnalyzer {
  constructor() {
    this.genAI = null;
    this.model = null;
    this.frameHistory = [];
    this.maxHistoryLength = 25; // Reduced from 60 (now ~5 seconds of data)
    this.lastAnalysisTime = 0;
    this.analysisInterval = 5000; // Every 5 seconds instead of 3
  }

  async initialize(apiKey) {
    try {
      this.genAI = new GoogleGenerativeAI(apiKey);
      this.model = this.genAI.getGenerativeModel({ 
        model: "gemini-2.5-pro",
        generationConfig: {
          temperature: 0.7,
          topK: 40,
          topP: 0.95,
          maxOutputTokens: 500,
        }
      });
      return true;
    } catch (error) {
      console.error('Gemini initialization failed:', error);
      return false;
    }
  }

  addFrameData(wellnessMetrics) {
    const frameData = {
      ...wellnessMetrics,
      timestamp: Date.now()
    };
    
    this.frameHistory.push(frameData);
    
    // Keep only recent frames
    if (this.frameHistory.length > this.maxHistoryLength) {
      this.frameHistory = this.frameHistory.slice(-this.maxHistoryLength);
    }
  }

  async analyzeWellnessTrends() {
    if (!this.model || this.frameHistory.length < 5) { // Reduced threshold
      return this.getDefaultRecommendation();
    }

    const now = Date.now();
    if (now - this.lastAnalysisTime < this.analysisInterval) {
      return null;
    }

    this.lastAnalysisTime = now;

    try {
      const trends = this.calculateTrends();
      const currentMetrics = this.frameHistory[this.frameHistory.length - 1];
      
      const prompt = this.buildComprehensivePrompt(currentMetrics, trends);
      
      const result = await this.model.generateContent(prompt);
      const response = await result.response;
      
      return this.parseAIResponse(response.text());
    } catch (error) {
      console.error('Gemini analysis error:', error);
      return this.getDefaultRecommendation();
    }
  }

  calculateTrends() {
    if (this.frameHistory.length < 10) return null; // Reduced threshold

    const recent = this.frameHistory.slice(-5); // Last 5 frames (~1 second)
    const earlier = this.frameHistory.slice(-15, -10); // Earlier 5 frames

    const calculateAverage = (frames, metric) => 
      frames.reduce((sum, frame) => sum + frame[metric], 0) / frames.length;

    const recentAvg = {
      faceAsymmetry: calculateAverage(recent, 'faceAsymmetry'),
      fatigue: calculateAverage(recent, 'fatigue'),
      eyeStrain: calculateAverage(recent, 'eyeStrain'),
      worsenedPosture: calculateAverage(recent, 'worsenedPosture'),
      stressedEmotion: calculateAverage(recent, 'stressedEmotion')
    };

    const earlierAvg = {
      faceAsymmetry: calculateAverage(earlier, 'faceAsymmetry'),
      fatigue: calculateAverage(earlier, 'fatigue'),
      eyeStrain: calculateAverage(earlier, 'eyeStrain'),
      worsenedPosture: calculateAverage(earlier, 'worsenedPosture'),
      stressedEmotion: calculateAverage(earlier, 'stressedEmotion')
    };

    return {
      faceAsymmetryTrend: this.getTrendDirection(earlierAvg.faceAsymmetry, recentAvg.faceAsymmetry),
      fatigueTrend: this.getTrendDirection(earlierAvg.fatigue, recentAvg.fatigue),
      eyeStrainTrend: this.getTrendDirection(earlierAvg.eyeStrain, recentAvg.eyeStrain),
      postureTrend: this.getTrendDirection(earlierAvg.worsenedPosture, recentAvg.worsenedPosture),
      stressTrend: this.getTrendDirection(earlierAvg.stressedEmotion, recentAvg.stressedEmotion),
      sessionDuration: (recent[recent.length - 1].timestamp - this.frameHistory[0].timestamp) / 60000 // minutes
    };
  }

  getTrendDirection(earlier, recent) {
    const change = recent - earlier;
    if (Math.abs(change) < 0.5) return 'stable';
    return change > 0 ? 'worsening' : 'improving';
  }

  buildComprehensivePrompt(currentMetrics, trends) {
    const trendInfo = trends ? `
TEMPORAL ANALYSIS (${Math.round(trends.sessionDuration)} min session):
- Face Asymmetry: ${trends.faceAsymmetryTrend}
- Fatigue Level: ${trends.fatigueTrend} 
- Eye Strain: ${trends.eyeStrainTrend}
- Posture Quality: ${trends.postureTrend}
- Stress Level: ${trends.stressTrend}
` : 'TEMPORAL ANALYSIS: Insufficient data for trend analysis';

    return `You are an expert AI wellness coach specializing in real-time biometric analysis for computer users. Analyze the current wellness state and provide actionable micro-interventions.

CURRENT BIOMETRIC READINGS (0-10 scale, 10=severe issue):
- Face Asymmetry: ${currentMetrics.faceAsymmetry.toFixed(1)}/10
- Fatigue Level: ${currentMetrics.fatigue.toFixed(1)}/10
- Eye Strain: ${currentMetrics.eyeStrain.toFixed(1)}/10  
- Posture Issues: ${currentMetrics.worsenedPosture.toFixed(1)}/10
- Stress/Tension: ${currentMetrics.stressedEmotion.toFixed(1)}/10

${trendInfo}

ANALYSIS FRAMEWORK:
1. Identify the PRIMARY concern (highest score + trend consideration)
2. Provide ONE specific, immediate action (max 12 words)
3. Calculate wellness score (0-100, considering both current state and trends)
4. Determine focus quality based on combined metrics
5. Award tokens based on improvement potential and effort required

RESPONSE REQUIREMENTS:
- Prioritize interventions by impact and feasibility
- Focus on evidence-based micro-interventions (30s-2min)
- Consider cumulative strain patterns
- Encourage sustainable habits

Return ONLY valid JSON:
{
  "recommendation": "Blink 10 times slowly, look 20 feet away",
  "wellnessScore": 72,
  "focusQuality": "Good",
  "tokenReward": 2,
  "primaryConcern": "Eye Strain",
  "urgency": "Medium",
  "reasoning": "Eye strain trending upward, immediate relief needed"
}`;
  }

  parseAIResponse(responseText) {
    try {
      // Extract JSON from response
      const jsonMatch = responseText.match(/\{[\s\S]*\}/);
      if (!jsonMatch) throw new Error('No JSON found in response');
      
      const parsed = JSON.parse(jsonMatch[0]);
      
      // Validate and sanitize the response
      return {
        recommendation: this.sanitizeRecommendation(parsed.recommendation || "Take a deep breath and relax"),
        wellnessScore: Math.max(0, Math.min(100, parsed.wellnessScore || 50)),
        focusQuality: this.validateFocusQuality(parsed.focusQuality || "Fair"),
        tokenReward: Math.max(0, Math.min(5, parsed.tokenReward || 1)),
        primaryConcern: parsed.primaryConcern || "General Wellness",
        urgency: this.validateUrgency(parsed.urgency || "Low"),
        reasoning: parsed.reasoning || "AI analysis completed"
      };
    } catch (error) {
      console.error('Failed to parse AI response:', error);
      return this.getDefaultRecommendation();
    }
  }

  sanitizeRecommendation(rec) {
    if (typeof rec !== 'string' || rec.length > 100) {
      return "Take a 30-second break and breathe deeply";
    }
    return rec.trim();
  }

  validateFocusQuality(quality) {
    const validQualities = ['Poor', 'Fair', 'Good', 'Excellent'];
    return validQualities.includes(quality) ? quality : 'Fair';
  }

  validateUrgency(urgency) {
    const validUrgencies = ['Low', 'Medium', 'High'];
    return validUrgencies.includes(urgency) ? urgency : 'Low';
  }

  getDefaultRecommendation() {
    return {
      recommendation: "Maintain good posture and blink regularly",
      wellnessScore: 75,
      focusQuality: "Fair",
      tokenReward: 1,
      primaryConcern: "General Wellness",
      urgency: "Low",
      reasoning: "Default wellness guidance"
    };
  }
}

export const geminiWellnessAnalyzer = new GeminiWellnessAnalyzer();