// MediaPipe Pose landmarks (33 points)
const POSE_LANDMARKS = {
  NOSE: 0,
  LEFT_EYE_INNER: 1,
  LEFT_EYE: 2,
  LEFT_EYE_OUTER: 3,
  RIGHT_EYE_INNER: 4,
  RIGHT_EYE: 5,
  RIGHT_EYE_OUTER: 6,
  LEFT_EAR: 7,
  RIGHT_EAR: 8,
  MOUTH_LEFT: 9,
  MOUTH_RIGHT: 10,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_PINKY: 17,
  RIGHT_PINKY: 18,
  LEFT_INDEX: 19,
  RIGHT_INDEX: 20,
  LEFT_THUMB: 21,
  RIGHT_THUMB: 22,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
  LEFT_HEEL: 29,
  RIGHT_HEEL: 30,
  LEFT_FOOT_INDEX: 31,
  RIGHT_FOOT_INDEX: 32
};

// MediaPipe Face landmarks (468 points) - key regions
const FACE_REGIONS = {
  // Left eye landmarks
  LEFT_EYE: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
  // Right eye landmarks  
  RIGHT_EYE: [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
  // Mouth landmarks
  MOUTH: [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
  // Eyebrow landmarks
  LEFT_EYEBROW: [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
  RIGHT_EYEBROW: [296, 334, 293, 300, 276, 283, 282, 295, 285, 336],
  // Face oval landmarks
  FACE_OVAL: [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
};

export const analyzeWellnessMetrics = (poseResults, faceResults) => {
  console.log("🔬 Starting wellness metrics analysis...");
  console.log("🔬 Input data:", { 
    poseResults, 
    faceResults,
    hasPoseLandmarks: !!poseResults?.poseLandmarks,
    hasFaceLandmarks: !!faceResults?.multiFaceLandmarks?.[0],
    poseLandmarksCount: poseResults?.poseLandmarks?.length || 0,
    faceLandmarksCount: faceResults?.multiFaceLandmarks?.[0]?.length || 0
  });
  
  // Check if we have valid data
  const hasPoseData = poseResults?.poseLandmarks && poseResults.poseLandmarks.length >= 33;
  const hasFaceData = faceResults?.multiFaceLandmarks?.[0] && faceResults.multiFaceLandmarks[0].length >= 468;
  
  console.log("🔬 Data validation:", { hasPoseData, hasFaceData });
  
  if (!hasPoseData && !hasFaceData) {
    console.warn("⚠️ Insufficient data for analysis");
    return {
      faceAsymmetry: 0,
      fatigue: 0,
      eyeStrain: 0,
      worsenedPosture: 0,
      stressedEmotion: 0
    };
  }

  const poseLandmarks = hasPoseData ? poseResults.poseLandmarks : null;
  const faceLandmarks = hasFaceData ? faceResults.multiFaceLandmarks[0] : null;

  console.log("🔬 Processing landmarks...");

  const metrics = {
    faceAsymmetry: faceLandmarks ? analyzeFaceAsymmetry(faceLandmarks) : 0,
    fatigue: faceLandmarks ? analyzeFatigue(faceLandmarks) : 0,
    eyeStrain: faceLandmarks ? analyzeEyeStrain(faceLandmarks) : 0,
    worsenedPosture: poseLandmarks ? analyzePosture(poseLandmarks) : 0,
    stressedEmotion: analyzeStress(faceLandmarks, poseLandmarks)
  };

  console.log("🔬 Computed wellness metrics:", metrics);
  return metrics;
};

const calculateDistance = (point1, point2) => {
  if (!point1 || !point2) return 0;
  return Math.sqrt(
    Math.pow(point1.x - point2.x, 2) + 
    Math.pow(point1.y - point2.y, 2) + 
    Math.pow((point1.z || 0) - (point2.z || 0), 2)
  );
};

const analyzeFaceAsymmetry = (faceLandmarks) => {
  try {
    console.log("😐 Analyzing face asymmetry...");
    
    // Get nose tip and eye centers
    const noseCenter = faceLandmarks[1]; // Nose tip
    const leftEyeCenter = getAveragePoint(faceLandmarks, FACE_REGIONS.LEFT_EYE.slice(0, 6));
    const rightEyeCenter = getAveragePoint(faceLandmarks, FACE_REGIONS.RIGHT_EYE.slice(0, 6));
    
    if (!noseCenter || !leftEyeCenter || !rightEyeCenter) {
      console.warn("⚠️ Missing key facial landmarks for asymmetry");
      return 0;
    }
    
    const leftDistance = calculateDistance(noseCenter, leftEyeCenter);
    const rightDistance = calculateDistance(noseCenter, rightEyeCenter);
    
    if (leftDistance === 0 || rightDistance === 0) return 0;
    
    const asymmetry = Math.abs(leftDistance - rightDistance) / Math.max(leftDistance, rightDistance);
    const score = Math.min(10, asymmetry * 25);
    
    console.log("😐 Asymmetry analysis:", { leftDistance, rightDistance, asymmetry, score });
    return score;
  } catch (error) {
    console.error("❌ Face asymmetry analysis failed:", error);
    return 0;
  }
};

const analyzeFatigue = (faceLandmarks) => {
  try {
    console.log("😴 Analyzing fatigue...");
    
    // Calculate eye aspect ratio for both eyes
    const leftEAR = calculateEyeAspectRatio(faceLandmarks, FACE_REGIONS.LEFT_EYE);
    const rightEAR = calculateEyeAspectRatio(faceLandmarks, FACE_REGIONS.RIGHT_EYE);
    
    if (leftEAR === 0 || rightEAR === 0) {
      console.warn("⚠️ Could not calculate eye aspect ratio");
      return 0;
    }
    
    const avgEAR = (leftEAR + rightEAR) / 2;
    
    // Normal EAR is around 0.3, fatigue shows lower values
    // EAR below 0.25 indicates fatigue, below 0.2 indicates high fatigue
    const fatigueScore = Math.max(0, Math.min(10, (0.3 - avgEAR) * 40));
    
    console.log("😴 Fatigue analysis:", { leftEAR, rightEAR, avgEAR, fatigueScore });
    return fatigueScore;
  } catch (error) {
    console.error("❌ Fatigue analysis failed:", error);
    return 0;
  }
};

const analyzeEyeStrain = (faceLandmarks) => {
  try {
    console.log("👁️ Analyzing eye strain...");
    
    // Calculate eye opening for both eyes
    const leftEyeOpening = calculateEyeOpening(faceLandmarks, FACE_REGIONS.LEFT_EYE);
    const rightEyeOpening = calculateEyeOpening(faceLandmarks, FACE_REGIONS.RIGHT_EYE);
    
    if (leftEyeOpening === 0 || rightEyeOpening === 0) {
      console.warn("⚠️ Could not calculate eye opening");
      return 0;
    }
    
    const avgEyeOpening = (leftEyeOpening + rightEyeOpening) / 2;
    
    // Lower eye opening indicates squinting/strain
    // Normal eye opening is around 0.02-0.03, strain shows lower values
    const strainScore = Math.max(0, Math.min(10, (0.025 - avgEyeOpening) * 200));
    
    console.log("👁️ Eye strain analysis:", { leftEyeOpening, rightEyeOpening, avgEyeOpening, strainScore });
    return strainScore;
  } catch (error) {
    console.error("❌ Eye strain analysis failed:", error);
    return 0;
  }
};

const analyzePosture = (poseLandmarks) => {
  try {
    console.log("🚶 Analyzing posture...");
    
    const nose = poseLandmarks[POSE_LANDMARKS.NOSE];
    const leftShoulder = poseLandmarks[POSE_LANDMARKS.LEFT_SHOULDER];
    const rightShoulder = poseLandmarks[POSE_LANDMARKS.RIGHT_SHOULDER];
    const leftHip = poseLandmarks[POSE_LANDMARKS.LEFT_HIP];
    const rightHip = poseLandmarks[POSE_LANDMARKS.RIGHT_HIP];

    if (!nose || !leftShoulder || !rightShoulder) {
      console.warn("⚠️ Missing key pose landmarks");
      return 0;
    }

    // Calculate shoulder center
    const shoulderCenter = {
      x: (leftShoulder.x + rightShoulder.x) / 2,
      y: (leftShoulder.y + rightShoulder.y) / 2,
      z: (leftShoulder.z + rightShoulder.z) / 2
    };

    // Forward head posture: head should be above shoulders
    const headForwardness = Math.abs((nose.z || 0) - shoulderCenter.z);
    
    // Shoulder asymmetry: shoulders should be level
    const shoulderAsymmetry = Math.abs(leftShoulder.y - rightShoulder.y);
    
    // Body alignment: check if upper body is aligned
    let bodyAlignment = 0;
    if (leftHip && rightHip) {
      const hipCenter = {
        x: (leftHip.x + rightHip.x) / 2,
        y: (leftHip.y + rightHip.y) / 2
      };
      bodyAlignment = Math.abs(shoulderCenter.x - hipCenter.x);
    }
    
    const postureScore = Math.min(10, (
      headForwardness * 15 + 
      shoulderAsymmetry * 25 + 
      bodyAlignment * 20
    ));
    
    console.log("🚶 Posture analysis:", { 
      headForwardness, 
      shoulderAsymmetry, 
      bodyAlignment, 
      postureScore 
    });
    
    return postureScore;
  } catch (error) {
    console.error("❌ Posture analysis failed:", error);
    return 0;
  }
};

const analyzeStress = (faceLandmarks, poseLandmarks) => {
  try {
    console.log("😰 Analyzing stress...");
    
    let facialStress = 0;
    let posturalStress = 0;
    
    // Facial stress indicators
    if (faceLandmarks) {
      // Eyebrow position (lowered eyebrows indicate stress/concentration)
      const leftBrowCenter = getAveragePoint(faceLandmarks, FACE_REGIONS.LEFT_EYEBROW.slice(0, 5));
      const rightBrowCenter = getAveragePoint(faceLandmarks, FACE_REGIONS.RIGHT_EYEBROW.slice(0, 5));
      const leftEyeCenter = getAveragePoint(faceLandmarks, FACE_REGIONS.LEFT_EYE.slice(0, 6));
      const rightEyeCenter = getAveragePoint(faceLandmarks, FACE_REGIONS.RIGHT_EYE.slice(0, 6));
      
      if (leftBrowCenter && rightBrowCenter && leftEyeCenter && rightEyeCenter) {
        const browEyeDistance = (
          (leftEyeCenter.y - leftBrowCenter.y) + 
          (rightEyeCenter.y - rightBrowCenter.y)
        ) / 2;
        
        // Closer eyebrows to eyes indicate stress
        facialStress = Math.max(0, browEyeDistance * 50);
      }
      
      // Mouth tension
      if (faceLandmarks.length > 314) {
        const mouthLeft = faceLandmarks[61];
        const mouthRight = faceLandmarks[291];
        const mouthCenter = faceLandmarks[13];
        
        if (mouthLeft && mouthRight && mouthCenter) {
          const mouthCurve = mouthCenter.y - (mouthLeft.y + mouthRight.y) / 2;
          // Negative curve (downward) indicates stress
          facialStress += Math.max(0, -mouthCurve * 30);
        }
      }
    }
    
    // Postural stress indicators
    if (poseLandmarks) {
      const leftShoulder = poseLandmarks[POSE_LANDMARKS.LEFT_SHOULDER];
      const rightShoulder = poseLandmarks[POSE_LANDMARKS.RIGHT_SHOULDER];
      
      if (leftShoulder && rightShoulder) {
        // Elevated shoulders indicate tension
        const shoulderElevation = Math.min(leftShoulder.y, rightShoulder.y); // Lower y = higher position
        posturalStress = Math.max(0, (0.3 - shoulderElevation) * 20); // Adjust threshold as needed
      }
    }
    
    const combinedStress = Math.min(10, facialStress * 0.7 + posturalStress * 0.3);
    
    console.log("😰 Stress analysis:", { facialStress, posturalStress, combinedStress });
    return combinedStress;
  } catch (error) {
    console.error("❌ Stress analysis failed:", error);
    return 0;
  }
};

// Helper functions
const getAveragePoint = (landmarks, indices) => {
  try {
    const validIndices = indices.filter(i => i < landmarks.length && landmarks[i]);
    if (validIndices.length === 0) return null;
    
    const sum = validIndices.reduce(
      (acc, i) => {
        const point = landmarks[i];
        return {
          x: acc.x + (point.x || 0),
          y: acc.y + (point.y || 0),
          z: acc.z + (point.z || 0)
        };
      },
      { x: 0, y: 0, z: 0 }
    );
    
    return {
      x: sum.x / validIndices.length,
      y: sum.y / validIndices.length,
      z: sum.z / validIndices.length
    };
  } catch (error) {
    console.error("❌ Error calculating average point:", error);
    return null;
  }
};

const calculateEyeAspectRatio = (landmarks, eyeIndices) => {
  try {
    // Use first 6 eye landmarks for EAR calculation
    const validIndices = eyeIndices.slice(0, 6).filter(i => i < landmarks.length && landmarks[i]);
    if (validIndices.length < 6) return 0;
    
    const eyePoints = validIndices.map(i => landmarks[i]);
    
    // Vertical eye distances (top/bottom pairs)
    const verticalDist1 = calculateDistance(eyePoints[1], eyePoints[5]);
    const verticalDist2 = calculateDistance(eyePoints[2], eyePoints[4]);
    const avgVerticalDist = (verticalDist1 + verticalDist2) / 2;
    
    // Horizontal eye distance
    const horizontalDist = calculateDistance(eyePoints[0], eyePoints[3]);
    
    if (horizontalDist === 0) return 0;
    
    return avgVerticalDist / horizontalDist;
  } catch (error) {
    console.error("❌ Error calculating eye aspect ratio:", error);
    return 0;
  }
};

const calculateEyeOpening = (landmarks, eyeIndices) => {
  try {
    const validIndices = eyeIndices.filter(i => i < landmarks.length && landmarks[i]);
    if (validIndices.length === 0) return 0;
    
    const eyePoints = validIndices.map(i => landmarks[i]);
    const yCoordinates = eyePoints.map(p => p.y);
    
    // Calculate the height of the eye (max Y - min Y)
    return Math.max(...yCoordinates) - Math.min(...yCoordinates);
  } catch (error) {
    console.error("❌ Error calculating eye opening:", error);
    return 0;
  }
};