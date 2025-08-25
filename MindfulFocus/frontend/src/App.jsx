import { useEffect, useState, useRef } from "react";
import { Radar } from "react-chartjs-2";
import Typed from "typed.js";
import "chart.js/auto";
import "./index.css";
import { useMediaPipeAnalysis } from "./hooks/useMediaPipeAnalysis";
import { analyzeWellnessMetrics } from "./utils/poseAnalysis";
import { geminiWellnessAnalyzer } from "./utils/GeminiAI";

// Force the API key prompt every reload
const ALWAYS_PROMPT_API = true;

function App() {
  const [balance, setBalance] = useState(0);
  const [chartData, setChartData] = useState([0, 0, 0, 0, 0]);
  const [recommendation, setRecommendation] = useState("Starting wellness analysis...");
  const [wellnessScore, setWellnessScore] = useState(0);
  const [focusQuality, setFocusQuality] = useState("Unknown");
  const [primaryConcern, setPrimaryConcern] = useState("Initializing");
  const [apiKey, setApiKey] = useState("");
  const [showApiSetup, setShowApiSetup] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentAnalysis, setCurrentAnalysis] = useState(null);

  const typedRef = useRef(null);
  const frameCountRef = useRef(0);
  const lastUpdateRef = useRef(0);
  const lastChartUpdateRef = useRef(0);

  const AI_ANALYSIS_INTERVAL = 5000; // 5s
  const CHART_UPDATE_INTERVAL = 1000; // 1s

  // Helper: detect Electron bridge
  const hasElectronBridge = () =>
    typeof window !== "undefined" &&
    window.api &&
    typeof window.api.getApiKey === "function" &&
    typeof window.api.setApiKey === "function" &&
    typeof window.api.getBalance === "function" &&
    typeof window.api.addTokens === "function" &&
    typeof window.api.clearApiKey === "function";

  // On reload/close: wipe any stored API key (Electron + browser)
  useEffect(() => {
    const onBeforeUnload = async () => {
      try { await window.api?.clearApiKey?.(); } catch {}
      try { localStorage.removeItem("gemini_api_key"); } catch {}
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, []);

  // Initialization
  useEffect(() => {
    const initializeApp = async () => {
      try {
        const currentBalance = hasElectronBridge()
          ? await window.api.getBalance()
          : Number(localStorage.getItem("mf_balance") || 0);
        setBalance(currentBalance);

        if (ALWAYS_PROMPT_API) {
          setShowApiSetup(true); // force modal every load
          return;
        }

        const savedApiKey = hasElectronBridge()
          ? await window.api.getApiKey()
          : localStorage.getItem("gemini_api_key");

        if (savedApiKey) {
          setApiKey(savedApiKey);
          const ok = await geminiWellnessAnalyzer.initialize(savedApiKey);
          if (!ok) setShowApiSetup(true);
        } else {
          setShowApiSetup(true);
        }
      } catch (e) {
        console.error("init failed:", e);
        setShowApiSetup(true);
      }
    };

    const waitForAPI = (retries = 50) => {
      if (hasElectronBridge()) return initializeApp();
      if (retries > 0) setTimeout(() => waitForAPI(retries - 1), 100);
      else initializeApp(); // browser fallback
    };

    waitForAPI();
  }, []);

  // MediaPipe results handler
  const handleMediaPipeResults = (results) => {
    try {
      const { pose, face } = results || {};
      const now = Date.now();
      if (!pose || !face) return;

      if (now - lastChartUpdateRef.current >= CHART_UPDATE_INTERVAL) {
        const metrics = analyzeWellnessMetrics(pose, face);
        const newChartData = [
          metrics.faceAsymmetry,
          metrics.fatigue,
          metrics.eyeStrain,
          metrics.worsenedPosture,
          metrics.stressedEmotion,
        ];
        setChartData(newChartData);
        geminiWellnessAnalyzer.addFrameData(metrics);
        lastChartUpdateRef.current = now;
        frameCountRef.current++;
      }

      const elapsed = now - lastUpdateRef.current;
      if (elapsed >= AI_ANALYSIS_INTERVAL && !isAnalyzing) {
        performAIAnalysis();
        lastUpdateRef.current = now;
      }
    } catch (e) {
      console.error("mediapipe processing error:", e);
    }
  };

  const { videoRef, canvasRef, isInitialized, error } = useMediaPipeAnalysis(
    handleMediaPipeResults
  );

  const performAIAnalysis = async () => {
    if (isAnalyzing) return;
    setIsAnalyzing(true);
    try {
      const analysis = await geminiWellnessAnalyzer.analyzeWellnessTrends();
      if (analysis) {
        setCurrentAnalysis(analysis);
        setRecommendation(analysis.recommendation);
        setWellnessScore(analysis.wellnessScore);
        setFocusQuality(analysis.focusQuality);
        setPrimaryConcern(analysis.primaryConcern);

        if (typedRef.current) try { typedRef.current.destroy(); } catch {}
        try {
          typedRef.current = new Typed("#recommendation", {
            strings: [analysis.recommendation],
            typeSpeed: 50,
            showCursor: false,
            onComplete: () => {
              const el = document.getElementById("wellness-indicator");
              if (el) {
                el.classList.add("animate-pulse");
                setTimeout(() => el.classList.remove("animate-pulse"), 2000);
              }
            },
          });
        } catch {}
      }
    } catch (e) {
      console.error("AI analysis failed:", e);
      setRecommendation("Keep maintaining good posture and taking breaks");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleApiKeySubmit = async () => {
    if (!apiKey.trim()) return;
    try {
      // Electron: set via IPC (may be ephemeral); Browser: DO NOT persist
      if (hasElectronBridge()) {
        await window.api.setApiKey(apiKey);
      } else {
        // Keep only in memory for this session
        // (do NOT write localStorage)
      }

      const ok = await geminiWellnessAnalyzer.initialize(apiKey);
      if (ok) {
        setShowApiSetup(false);
        setRecommendation("AI wellness analysis activated! 🧠");
      } else {
        alert("Failed to initialize Gemini AI. Please check your API key.");
      }
    } catch (e) {
      console.error("API key setup failed:", e);
      alert("Failed to save API key. Please try again.");
    }
  };

  const handleRewardClaim = async () => {
    if (!currentAnalysis?.tokenReward) return;
    const delta = Number(currentAnalysis.tokenReward) || 0;
    try {
      let newBalance = 0;
      if (hasElectronBridge()) {
        newBalance = await window.api.addTokens(delta);
      } else {
        const local = Number(localStorage.getItem("mf_balance") || 0) + delta;
        localStorage.setItem("mf_balance", String(local));
        newBalance = local;
      }
      setBalance(newBalance);
      const btn = document.getElementById("reward-button");
      if (btn) {
        const orig = btn.textContent;
        btn.textContent = `+${delta} Tokens! ✨`;
        btn.disabled = true;
        btn.classList.add("bg-green-500");
        setTimeout(() => {
          btn.textContent = orig;
          btn.disabled = false;
          btn.classList.remove("bg-green-500");
        }, 3000);
      }
      setCurrentAnalysis((p) => ({ ...p, tokenReward: 0 }));
    } catch (e) {
      console.error("claim reward failed:", e);
    }
  };

  const getWellnessColor = (s) => (s >= 80 ? "text-green-600" : s >= 60 ? "text-yellow-600" : "text-red-600");
  const getUrgencyColor = (u) => ({ High: "bg-red-100 text-red-800", Medium: "bg-yellow-100 text-yellow-800", default: "bg-green-100 text-green-800" }[u] || "bg-green-100 text-green-800");

  const labels = ["Face Asymmetry", "Fatigue", "Eye Strain", "Worsened Posture", "Stressed Emotion"];

  if (showApiSetup) {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100">
        <div className="bg-white p-8 rounded-2xl shadow-2xl max-w-md w-full mx-4 border border-gray-200">
          <div className="text-center mb-6">
            <div className="w-16 h-16 bg-brandBlue rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl text-white">🧠</span>
            </div>
            <h2 className="text-2xl font-bold text-brandBlue">Setup AI Wellness Coach</h2>
            <p className="text-gray-600 mt-2">Enter your Google Gemini API key to enable intelligent wellness analysis</p>
          </div>

          <div className="space-y-4">
            <input
              type="password"
              placeholder="Enter Gemini API Key"
              className="w-full p-4 border border-gray-300 rounded-xl focus:ring-2 focus:ring-brandBlue focus:border-transparent"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleApiKeySubmit()}
            />

            <button
              onClick={handleApiKeySubmit}
              disabled={!apiKey.trim()}
              className="w-full bg-brandBlue text-white py-4 rounded-xl font-semibold hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Initialize AI Coach
            </button>
          </div>

          <div className="mt-6 p-4 bg-blue-50 rounded-xl">
            <p className="text-xs text-blue-800 font-medium mb-2">Get your free API key:</p>
            <a href="https://makersuite.google.com/app/apikey" target="_blank" rel="noopener noreferrer" className="text-xs text-blue-600 underline break-all">
              https://makersuite.google.com/app/apikey
            </a>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen w-screen flex flex-col bg-gray-50">
      <div className="flex justify-between items-center border-b-2 border-brandBlue p-4 bg-white shadow-sm">
        <div className="flex items-center space-x-4">
          <img src="/Assets/Logo.png" alt="Logo" className="h-12" />
          <div>
            <h1 className="text-2xl font-bold text-brandBlue">Mindful Focus</h1>
            <div className="flex items-center space-x-2 mt-1">
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${isInitialized ? "bg-green-100 text-green-800" : "bg-yellow-100 text-yellow-800"}`}>
                {isInitialized ? "🟢 AI Active" : "🟡 Initializing..."}
              </span>
              {error && (
                <span className="px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">❌ Camera Error</span>
              )}
              {/* Change API Key button */}
              <button
                onClick={async () => {
                  try { await window.api?.clearApiKey?.(); } catch {}
                  try { localStorage.removeItem("gemini_api_key"); } catch {}
                  setApiKey("");
                  setShowApiSetup(true);
                }}
                className="ml-3 text-xs px-3 py-1 rounded-lg border border-gray-300 hover:bg-gray-100"
              >
                Change API Key
              </button>
            </div>
          </div>
        </div>

        <div className="text-right">
          <p className="text-sm font-semibold text-brandBlue">Rewards Balance</p>
          <p className="text-3xl font-bold text-brandBlue flex items-center justify-end gap-2">
            {balance} <span className="text-yellow-500">⭐</span>
          </p>
          <div className="text-xs text-gray-600 space-y-1">
            <div className="flex justify-between items-center"><span>Focus Quality:</span><span className="font-semibold">{focusQuality}</span></div>
            <div className="flex justify-between items-center" id="wellness-indicator">
              <span>Wellness:</span>
              <span className={`font-semibold ${getWellnessColor(wellnessScore)}`}>{wellnessScore}/100</span>
            </div>
            <div className="flex justify-between items-center"><span>Primary:</span><span className="font-semibold text-xs">{primaryConcern}</span></div>
          </div>
        </div>
      </div>

      <div className="flex flex-1 p-6 space-x-6">
        <div className="flex-1 relative">
          <div className="rounded-2xl overflow-hidden border-4 border-brandBlue shadow-lg h-full">
            <video ref={videoRef} autoPlay muted playsInline className="w-full h-full object-cover" />
            <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full pointer-events-none" />

            {isAnalyzing && (
              <div className="absolute top-4 right-4 bg-brandBlue text-white px-4 py-2 rounded-full text-sm font-medium flex items-center space-x-2">
                <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full"></div>
                <span>Analyzing...</span>
              </div>
            )}

            {!isInitialized && !error && (
              <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center">
                <div className="text-white text-center">
                  <div className="animate-spin w-8 h-8 border-4 border-white border-t-transparent rounded-full mx-auto mb-4"></div>
                  <p className="text-lg font-semibold">Initializing MediaPipe...</p>
                  <p className="text-sm opacity-75">Setting up pose and face detection</p>
                </div>
              </div>
            )}

            {error && (
              <div className="absolute inset-0 bg-red-900 bg-opacity-90 flex items-center justify-center">
                <div className="text-white text-center p-6">
                  <p className="text-xl font-bold mb-2">⚠️ Camera Access Required</p>
                  <p className="text-sm">{error}</p>
                  <button onClick={() => window.location.reload()} className="mt-4 bg-white text-red-900 px-4 py-2 rounded-lg font-semibold">Retry</button>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="w-1/3 space-y-6">
          <div className="bg-white rounded-2xl p-6 shadow-lg border border-gray-200">
            <h3 className="text-lg font-bold text-brandBlue mb-4">Wellness Metrics</h3>
            <div className="h-64">
              <Radar
                data={{
                  labels,
                  datasets: [{
                    label: "Current Level",
                    data: chartData,
                    backgroundColor: "rgba(19, 88, 213, 0.1)",
                    borderColor: "#1358D5",
                    borderWidth: 2,
                    pointBackgroundColor: "#1358D5",
                    pointBorderColor: "#fff",
                    pointBorderWidth: 2,
                    pointRadius: 4,
                  }],
                }}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  scales: { r: { min: 0, max: 10, ticks: { stepSize: 2, font: { size: 10 } }, grid: { color: "rgba(0,0,0,0.1)" } } },
                  plugins: { legend: { display: false } },
                  elements: { line: { tension: 0.2 } },
                }}
              />
            </div>
          </div>

          <div className="bg-white rounded-2xl p-6 shadow-lg border border-gray-200">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-brandBlue">AI Recommendations</h3>
              {currentAnalysis?.urgency && (
                <span className={`px-3 py-1 rounded-full text-xs font-medium ${getUrgencyColor(currentAnalysis.urgency)}`}>
                  {currentAnalysis.urgency} Priority
                </span>
              )}
            </div>
            <div className="space-y-4">
              <div className="bg-blue-50 rounded-xl p-4">
                <p id="recommendation" className="text-lg font-medium text-brandBlue leading-relaxed">{recommendation}</p>
                <p className="text-xs text-gray-500 mt-2 flex items-center"><span className="mr-2">🤖</span>AI Wellness Coach</p>
              </div>
              {currentAnalysis?.reasoning && (
                <div className="text-sm text-gray-600 bg-gray-50 rounded-xl p-3">
                  <p className="font-medium mb-1">Analysis:</p>
                  <p>{currentAnalysis.reasoning}</p>
                </div>
              )}
              <div className="bg-yellow-50 rounded-xl p-4">
                <p className="text-sm text-gray-700 mb-3">
                  Complete the suggested action to earn <span className="font-bold text-brandBlue">{currentAnalysis?.tokenReward || 1} tokens</span>
                </p>
                <button
                  id="reward-button"
                  onClick={handleRewardClaim}
                  disabled={!currentAnalysis?.tokenReward}
                  className="w-full bg-brandBlue text-white py-3 rounded-xl font-semibold hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                >
                  <span>✅ I completed it!</span>
                  {currentAnalysis?.tokenReward > 0 && (
                    <span className="bg-yellow-500 text-brandBlue px-2 py-1 rounded-full text-xs font-bold">
                      +{currentAnalysis.tokenReward}
                    </span>
                  )}
                </button>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-2xl p-6 shadow-lg border border-gray-200">
            <h3 className="text-lg font-bold text-brandBlue mb-4">Session Stats</h3>
            <div className="grid grid-cols-2 gap-4 text-center">
              <div className="bg-blue-50 rounded-xl p-3">
                <p className="text-2xl font-bold text-brandBlue">{frameCountRef.current}</p>
                <p className="text-xs text-gray-600">Analyses Completed</p>
              </div>
              <div className="bg-green-50 rounded-xl p-3">
                <p className="text-2xl font-bold text-green-600">{Math.floor(frameCountRef.current / 5)}</p>
                <p className="text-xs text-gray-600">Seconds Active</p>
              </div>
              <div className="bg-purple-50 rounded-xl p-3 col-span-2">
                <p className="text-lg font-bold text-purple-600">5 FPS</p>
                <p className="text-xs text-gray-600">Analysis Rate</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;