import { useEffect, useRef, useState } from 'react';

export const useMediaPipeAnalysis = (onResults) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [error, setError] = useState(null);
  
  const faceLandmarkerRef = useRef(null);
  const poseLandmarkerRef = useRef(null);
  const lastProcessTimeRef = useRef(0);
  const animationFrameRef = useRef(null);
  
  const TARGET_FPS = 5;
  const FRAME_INTERVAL = 1000 / TARGET_FPS;

  useEffect(() => {
    console.log("🎥 Starting MediaPipe Tasks Vision initialization...");
    
    let isComponentMounted = true;
    
    const initializeMediaPipe = async () => {
      try {
        if (!videoRef.current) {
          console.log("⏳ Video element not ready, waiting...");
          setTimeout(initializeMediaPipe, 100);
          return;
        }

        console.log("📦 Loading MediaPipe Tasks Vision...");
        
        // Import MediaPipe Tasks Vision
        const vision = await import('@mediapipe/tasks-vision');
        console.log("📦 MediaPipe imported successfully:", Object.keys(vision));
        
        const { FaceLandmarker, PoseLandmarker, FilesetResolver } = vision;
        
        console.log("📦 Creating FilesetResolver...");
        const visionInstance = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm"
        );
        
        console.log("🦾 Creating PoseLandmarker...");
        const poseLandmarker = await PoseLandmarker.createFromOptions(visionInstance, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5
        });
        
        console.log("😀 Creating FaceLandmarker...");
        const faceLandmarker = await FaceLandmarker.createFromOptions(visionInstance, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numFaces: 1,
          minFaceDetectionConfidence: 0.5,
          minFacePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
          outputFaceBlendshapes: false,
          outputFacialTransformationMatrixes: false
        });
        
        if (!isComponentMounted) {
          console.log("🛑 Component unmounted, cleaning up...");
          faceLandmarker.close();
          poseLandmarker.close();
          return;
        }
        
        faceLandmarkerRef.current = faceLandmarker;
        poseLandmarkerRef.current = poseLandmarker;
        
        console.log("🎥 Setting up camera stream...");
        
        // Get camera stream
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            frameRate: { ideal: 15, max: 15 }
          },
          audio: false
        });
        
        if (!isComponentMounted) {
          console.log("🛑 Component unmounted, stopping stream...");
          stream.getTracks().forEach(track => track.stop());
          return;
        }
        
        videoRef.current.srcObject = stream;
        
        // Wait for video to be ready
        await new Promise((resolve) => {
          videoRef.current.onloadedmetadata = () => {
            console.log("📹 Video metadata loaded");
            resolve();
          };
        });
        
        console.log("✅ MediaPipe Tasks Vision initialization completed successfully");
        setIsInitialized(true);
        
        // Start processing loop
        processFrame();
        
      } catch (err) {
        console.error("❌ MediaPipe initialization failed:", err);
        console.error("❌ Error details:", {
          name: err.name,
          message: err.message,
          stack: err.stack
        });
        setError(err.message);
      }
    };

    const processFrame = async () => {
      if (!isComponentMounted || !faceLandmarkerRef.current || !poseLandmarkerRef.current || !videoRef.current) {
        if (isComponentMounted) {
          animationFrameRef.current = requestAnimationFrame(processFrame);
        }
        return;
      }

      const now = performance.now();
      if (now - lastProcessTimeRef.current < FRAME_INTERVAL) {
        animationFrameRef.current = requestAnimationFrame(processFrame);
        return;
      }

      try {
        const video = videoRef.current;
        if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
          const timestamp = performance.now();
          
          console.log("🔍 Processing frame at timestamp:", timestamp);
          
          // Process pose
          const poseResults = await poseLandmarkerRef.current.detectForVideo(video, timestamp);
          console.log("🦾 Pose results:", {
            landmarksFound: poseResults.landmarks?.length || 0,
            worldLandmarksFound: poseResults.worldLandmarks?.length || 0
          });
          
          // Process face
          const faceResults = await faceLandmarkerRef.current.detectForVideo(video, timestamp);
          console.log("😀 Face results:", {
            landmarksFound: faceResults.faceLandmarks?.length || 0
          });
          
          // Prepare combined results in expected format
          const combinedResults = {
            pose: {
              poseLandmarks: poseResults.landmarks?.[0] || null,
              worldLandmarks: poseResults.worldLandmarks?.[0] || null
            },
            face: {
              multiFaceLandmarks: faceResults.faceLandmarks || []
            }
          };
          
          // Call results callback
          if (onResults && (poseResults.landmarks?.length > 0 || faceResults.faceLandmarks?.length > 0)) {
            console.log("📊 Sending results to callback:", {
              hasPose: !!combinedResults.pose.poseLandmarks,
              hasFace: combinedResults.face.multiFaceLandmarks.length > 0
            });
            
            onResults(combinedResults);
          }
          
          lastProcessTimeRef.current = now;
        }
      } catch (processError) {
        console.error("❌ Frame processing error:", processError);
        console.error("❌ Processing error details:", {
          name: processError.name,
          message: processError.message,
          stack: processError.stack
        });
      }
      
      if (isComponentMounted) {
        animationFrameRef.current = requestAnimationFrame(processFrame);
      }
    };

    initializeMediaPipe();

    return () => {
      console.log("🧹 Cleaning up MediaPipe Tasks Vision...");
      isComponentMounted = false;
      
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach(track => {
          console.log("🛑 Stopping track:", track.kind);
          track.stop();
        });
      }
      
      if (faceLandmarkerRef.current) {
        console.log("🧹 Closing face landmarker...");
        try {
          faceLandmarkerRef.current.close();
        } catch (err) {
          console.warn("⚠️ Error closing face landmarker:", err);
        }
      }
      
      if (poseLandmarkerRef.current) {
        console.log("🧹 Closing pose landmarker...");
        try {
          poseLandmarkerRef.current.close();
        } catch (err) {
          console.warn("⚠️ Error closing pose landmarker:", err);
        }
      }
    };
  }, [onResults]);

  return { 
    videoRef, 
    canvasRef, 
    isInitialized, 
    error 
  };
};