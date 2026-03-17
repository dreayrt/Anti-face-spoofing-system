import React, { useRef, useState, useEffect } from 'react';
import * as faceapi from '@vladmandic/face-api';
import { apiService } from '../../services/api';

const CameraPage = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [verificationResult, setVerificationResult] = useState(null);
  const [error, setError] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [facesDetected, setFacesDetected] = useState(0);
  const [faceBox, setFaceBox] = useState(null); // {x, y, w, h}
  const isProcessingRef = useRef(false);

  // Keep the ref in sync with state
  useEffect(() => {
    isProcessingRef.current = isProcessing;
  }, [isProcessing]);

  // Load Face API models once
  useEffect(() => {
    const loadModels = async () => {
      try {
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri('/models')
        ]);
        console.log('[FaceDetect] Models loaded successfully');
        setModelsLoaded(true);
      } catch (err) {
        console.error("[FaceDetect] Error loading models:", err);
        setError("Failed to load face detection models.");
      }
    };
    loadModels();
  }, []);

  // Start webcam automatically when models are loaded
  useEffect(() => {
    if (modelsLoaded) {
      startCamera();
    }
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, [modelsLoaded]);

  // Face detection loop — runs when streaming and models are loaded
  useEffect(() => {
    if (!isStreaming || !modelsLoaded) return;

    console.log('[FaceDetect] Starting face detection loop');
    
    const timer = setInterval(async () => {
      const video = videoRef.current;
      if (!video || video.readyState !== 4 || isProcessingRef.current) return;
      if (video.videoWidth === 0) return;

      try {
        const detections = await faceapi.detectAllFaces(
          video, 
          new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.3 })
        );
        
        setFacesDetected(detections.length);
        
        if (detections.length === 1) {
          const box = detections[0].box;
          setFaceBox({
            x: box.x,
            y: box.y,
            w: box.width,
            h: box.height
          });
        } else {
          setFaceBox(null);
        }
      } catch (err) {
        console.error('[FaceDetect] Detection error:', err);
      }
    }, 150);

    return () => {
      console.log('[FaceDetect] Clearing face detection interval');
      clearInterval(timer);
    };
  }, [isStreaming, modelsLoaded]);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsStreaming(true);
        setError('');
      }
    } catch (err) {
      setError('Camera access denied. Please grant permissions to use the Kiosk.');
    }
  };

  const captureAndVerify = async () => {
    if (!videoRef.current || !isStreaming || isProcessing || facesDetected !== 1 || !faceBox) return;
    setIsProcessing(true);
    setVerificationResult(null);
    setError('');
    
    // We create a temporary canvas to extract just the image data, we don't need a visible canvas anymore
    const captureCanvas = document.createElement('canvas');
    const context = captureCanvas.getContext('2d');
    captureCanvas.width = videoRef.current.videoWidth;
    captureCanvas.height = videoRef.current.videoHeight;
    
    // Draw current frame to canvas, handle mirroring
    context.translate(captureCanvas.width, 0);
    context.scale(-1, 1);
    context.drawImage(videoRef.current, 0, 0, captureCanvas.width, captureCanvas.height);
    
    // Convert to base64
    const imageData = captureCanvas.toDataURL('image/jpeg', 0.8);
    
    try {
      // Send both image and bounding box
      const result = await apiService.verifyFace(imageData, faceBox);
      setVerificationResult(result);
      
      // Clear result after 5 seconds to reset kiosk
      setTimeout(() => {
        setVerificationResult(null);
      }, 5000);
      
    } catch (err) {
      setError('Verification failed to connect to server.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-[80vh] w-full">
      <div className="text-center mb-8">
        <h2 className="text-4xl font-black bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600 mb-2">Live Authentication</h2>
        <p className="text-slate-500">Hãy đặt khuôn mặt của bạn sao cho rõ nét trong khung hình để ghi nhận sự có mặt.</p>
      </div>
      
      {error && (
        <div className="mb-6 bg-red-50 border border-red-200 text-red-600 px-6 py-4 rounded-xl flex items-center shadow-sm animate-shake">
          <svg className="w-5 h-5 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
          {error}
        </div>
      )}

      <div className="relative group">
        {/* Soft elegant shadow behind camera */}
        <div className={`absolute -inset-1 rounded-3xl blur-xl opacity-40 transition duration-1000 ${isProcessing ? 'bg-indigo-300 animate-pulse' : 'bg-slate-200'}`}></div>
        
        <div className="relative bg-white rounded-[1.5rem] p-3 border border-slate-100 shadow-xl overflow-hidden">
          <div className="relative rounded-2xl overflow-hidden aspect-video w-full max-w-3xl flex items-center justify-center bg-slate-900">
            
            {!isStreaming && !error && (
              <div className="absolute inset-0 flex flex-col items-center justify-center z-10 bg-slate-800">
                <div className="w-12 h-12 border-4 border-slate-600 border-t-white rounded-full animate-spin mb-4"></div>
                <p className="text-white/80 font-medium tracking-wide">
                  {!modelsLoaded ? "Loading AI Models..." : "Initializing Camera..."}
                </p>
              </div>
            )}

            {/* Bounding Box Overlay */}
            {isStreaming && !isProcessing && faceBox && (
              <div 
                className="absolute z-10 border-2 shadow-[0_0_15px_rgba(34,197,94,0.5)] transition-all duration-100 ease-out"
                style={{
                  // Because video is scale-x-[-1] (mirrored), we must invert the X coordinate relative to videoWidth
                  right: `${(faceBox.x / videoRef.current.videoWidth) * 100}%`, 
                  top: `${(faceBox.y / videoRef.current.videoHeight) * 100}%`,
                  width: `${(faceBox.w / videoRef.current.videoWidth) * 100}%`,
                  height: `${(faceBox.h / videoRef.current.videoHeight) * 100}%`,
                  borderColor: facesDetected === 1 ? '#22c55e' : '#ef4444', // Green if 1 face, Red if multiple
                  borderRadius: '10px'
                }}
              >
                <div className="absolute -top-6 left-0 bg-green-500 text-white text-xs font-bold px-2 py-0.5 rounded shadow">
                  Face Detected
                </div>
              </div>
            )}

            {/* Face Status Warnings */}
            {isStreaming && !isProcessing && (
              <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 transition-opacity">
                {facesDetected === 0 && (
                  <span className="bg-amber-100 text-amber-800 border border-amber-300 text-sm px-4 py-1.5 rounded-full font-medium shadow-sm">
                    No face detected
                  </span>
                )}
                {facesDetected > 1 && (
                  <span className="bg-red-100 text-red-800 border border-red-300 text-sm px-4 py-1.5 rounded-full font-medium shadow-sm flex items-center">
                    <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
                    Multiple faces detected
                  </span>
                )}
              </div>
            )}

            {/* Subtle processing overlay (no scan line) */}
            {isProcessing && (
              <div className="absolute inset-0 z-20 pointer-events-none flex items-center justify-center bg-slate-900/30 backdrop-blur-[2px]">
                 <div className="bg-white/90 px-6 py-3 rounded-full shadow-lg flex items-center">
                    <div className="w-5 h-5 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin mr-3"></div>
                    <span className="text-indigo-700 font-semibold tracking-wide">Processing Biometrics...</span>
                 </div>
              </div>
            )}

            <video 
              ref={videoRef} 
              autoPlay 
              playsInline 
              muted
              className={`w-full h-full object-cover transform scale-x-[-1] transition-opacity duration-300 ${isStreaming ? 'opacity-100' : 'opacity-0'}`}
            />
            
            {/* We no longer need the visible canvasRef, it was for the old capture method */}
          </div>
        </div>

        {/* Verification Result Toast Overlay */}
        <div className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-sm z-30 transition-all duration-500 transform ${verificationResult ? 'scale-100 opacity-100 translate-y-0' : 'scale-95 opacity-0 pointer-events-none translate-y-4'}`}>
          {verificationResult && (
            <div className={`p-6 rounded-2xl shadow-2xl backdrop-blur-xl border ${
              verificationResult.success 
                ? 'bg-emerald-900/80 border-emerald-500/50 shadow-[0_0_30px_rgba(16,185,129,0.3)]' 
                : 'bg-rose-900/80 border-rose-500/50 shadow-[0_0_30px_rgba(244,63,94,0.3)]'
            }`}>
              <div className="flex flex-col items-center text-center">
                <div className={`w-16 h-16 rounded-full flex items-center justify-center mb-4 ${verificationResult.success ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'}`}>
                  {verificationResult.success ? (
                    <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M5 13l4 4L19 7"></path></svg>
                  ) : (
                    <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M6 18L18 6M6 6l12 12"></path></svg>
                  )}
                </div>
                
                <h3 className={`text-xl font-bold mb-1 ${verificationResult.success ? 'text-emerald-300' : 'text-rose-300'}`}>
                  {verificationResult.success ? 'Access Granted' : 'Access Denied'}
                </h3>
                
                {verificationResult.user ? (
                  <div className="mt-3 w-full bg-black/30 rounded-xl p-3 border border-white/5">
                    <p className="text-lg font-semibold text-white">{verificationResult.user.name}</p>
                    <p className="text-sm border border-fuchsia-500/30 bg-fuchsia-500/10 text-fuchsia-300 inline-block px-2 py-0.5 rounded-md mt-1 mb-2 font-mono">ID: {verificationResult.user.id}</p>
                    <div className="w-full bg-white/10 rounded-full h-1.5 mt-2 overflow-hidden">
                       <div className="bg-gradient-to-r from-emerald-400 to-cyan-400 h-1.5 rounded-full" style={{width: `${Math.min(100, (verificationResult.liveness_score || 0) * 100)}%`}}></div>
                    </div>
                    <p className="text-[10px] text-white/40 mt-1 uppercase tracking-wider text-left">Liveness Score: {((verificationResult.liveness_score || 0) * 100).toFixed(1)}%</p>
                  </div>
                ) : (
                  <p className="text-white/80 mt-2 text-sm max-w-[250px]">{verificationResult.message || 'Spoof detected or face not matched.'}</p>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="mt-8">
        <button
          onClick={captureAndVerify}
          disabled={!isStreaming || isProcessing || facesDetected !== 1}
          className={`font-bold text-lg py-4 px-12 rounded-2xl transition-all duration-300 ${
            !isStreaming || isProcessing || facesDetected !== 1
              ? 'bg-slate-100 text-slate-400 cursor-not-allowed border border-slate-200 shadow-inner' 
              : 'bg-indigo-600 text-white hover:bg-indigo-700 shadow-md hover:shadow-lg hover:-translate-y-1 focus:ring-4 focus:ring-indigo-100'
          }`}
        >
          <span className="flex items-center justify-center">
            {isProcessing ? (
               <>
                 <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white/70" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                 Processing...
               </>
            ) : (
               <>
                 <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"></path></svg>
                 Scan Face
               </>
            )}
          </span>
        </button>
      </div>
    </div>
  );
};

export default CameraPage;
