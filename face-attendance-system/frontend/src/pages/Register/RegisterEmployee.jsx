import React, { useRef, useState, useEffect } from 'react';
import * as faceapi from '@vladmandic/face-api';
import { apiService } from '../../services/api';

const RegisterEmployee = () => {
  const videoRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [employeeName, setEmployeeName] = useState('');
  const [employeeId, setEmployeeId] = useState('');
  const [registrationResult, setRegistrationResult] = useState(null);
  const [error, setError] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [facesDetected, setFacesDetected] = useState(0);
  const [faceBox, setFaceBox] = useState(null);
  const isProcessingRef = useRef(false);

  // Initialize employee ID
  useEffect(() => {
    setEmployeeId(`EMP-${Date.now()}`);
  }, []);

  // Sync ref
  useEffect(() => {
    isProcessingRef.current = isProcessing;
  }, [isProcessing]);

  // Load models
  useEffect(() => {
    const loadModels = async () => {
      try {
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
          faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
          faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
          faceapi.nets.faceRecognitionNet.loadFromUri('/models')
        ]);
        setModelsLoaded(true);
      } catch (err) {
        console.error("Error loading models:", err);
        setError("Failed to load face detection models.");
      }
    };
    loadModels();
  }, []);

  // Start camera
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

  // Face detection loop
  useEffect(() => {
    if (!isStreaming || !modelsLoaded) return;
    
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
          setFaceBox({ x: box.x, y: box.y, w: box.width, h: box.height });
        } else {
          setFaceBox(null);
        }
      } catch (err) {
        console.error('Detection error:', err);
      }
    }, 150);

    return () => clearInterval(timer);
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
      setError('Camera access denied. Please grant permissions.');
    }
  };

  const captureAndRegister = async () => {
    if (!employeeName.trim()) {
      setError('Please enter the employee name.');
      return;
    }
    if (!videoRef.current || !isStreaming || isProcessing || facesDetected !== 1 || !faceBox) {
      setError('Valid face not detected or camera is not ready.');
      return;
    }
    
    setIsProcessing(true);
    setRegistrationResult(null);
    setError('');
    
    const captureCanvas = document.createElement('canvas');
    const context = captureCanvas.getContext('2d');
    captureCanvas.width = videoRef.current.videoWidth;
    captureCanvas.height = videoRef.current.videoHeight;
    
    context.translate(captureCanvas.width, 0);
    context.scale(-1, 1);
    context.drawImage(videoRef.current, 0, 0, captureCanvas.width, captureCanvas.height);
    
    const imageData = captureCanvas.toDataURL('image/jpeg', 0.8);
    
    try {
      // Extract face descriptor (128D embedding) for face matching
      const detection = await faceapi
        .detectSingleFace(videoRef.current, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.7 }))
        .withFaceLandmarks()
        .withFaceDescriptor();
      
      if (!detection) {
        setError('Could not extract face features. Please try again.');
        setIsProcessing(false);
        return;
      }

      const descriptor = Array.from(detection.descriptor); // Float32Array -> regular array

      const result = await apiService.registerEmployee({
        id: employeeId,
        name: employeeName.trim(),
        image: imageData,
        descriptor: descriptor
      });
      setRegistrationResult(result);
      // Reset form on success
      if (result.success) {
        setEmployeeName('');
        setEmployeeId(`EMP-${Date.now()}`);
      }
      setTimeout(() => setRegistrationResult(null), 5000);
    } catch (err) {
      setError(err.response?.data?.detail || 'Registration failed to connect to server.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-[80vh] w-full">
      <div className="text-center mb-8">
        <h2 className="text-4xl font-black bg-clip-text text-transparent bg-gradient-to-r from-emerald-600 to-teal-600 mb-2">Employee Registration</h2>
        <p className="text-slate-500">Thêm nhân viên mới và đăng ký dữ liệu sinh trắc học khuôn mặt của họ.</p>
      </div>
      
      {error && (
        <div className="mb-6 bg-red-50 border border-red-200 text-red-600 px-6 py-4 rounded-xl flex items-center shadow-sm">
          <svg className="w-5 h-5 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
          {error}
        </div>
      )}

      {/* Form Controls */}
      <div className="w-full max-w-3xl mb-8 flex flex-col md:flex-row gap-4 bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
        <div className="flex-1">
          <label className="block text-sm font-semibold text-slate-700 mb-2">Employee Name</label>
          <input 
            type="text" 
            value={employeeName}
            onChange={(e) => setEmployeeName(e.target.value)}
            className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all"
            placeholder="e.g. Nguyen Van A"
          />
        </div>
        <div className="md:w-1/3">
          <label className="block text-sm font-semibold text-slate-700 mb-2">Employee ID (Auto)</label>
          <input 
            type="text" 
            value={employeeId}
            readOnly
            className="w-full px-4 py-3 rounded-xl bg-slate-50 border border-slate-200 text-slate-500 cursor-not-allowed"
          />
        </div>
      </div>

      <div className="relative group w-full max-w-3xl">
        <div className={`absolute -inset-1 rounded-3xl blur-xl opacity-40 transition duration-1000 ${isProcessing ? 'bg-teal-300 animate-pulse' : 'bg-slate-200'}`}></div>
        
        <div className="relative bg-white rounded-[1.5rem] p-3 border border-slate-100 shadow-xl overflow-hidden w-full">
          <div className="relative rounded-2xl overflow-hidden aspect-video flex items-center justify-center bg-slate-900 w-full">
            
            {!isStreaming && !error && (
              <div className="absolute inset-0 flex flex-col items-center justify-center z-10 bg-slate-800">
                <div className="w-12 h-12 border-4 border-slate-600 border-t-white rounded-full animate-spin mb-4"></div>
                <p className="text-white/80 font-medium tracking-wide">
                  {!modelsLoaded ? "Loading AI Models..." : "Initializing Camera..."}
                </p>
              </div>
            )}

            {isStreaming && !isProcessing && faceBox && (
              <div 
                className="absolute z-10 border-2 shadow-[0_0_15px_rgba(34,197,94,0.5)] transition-all duration-100 ease-out"
                style={{
                  right: `${(faceBox.x / videoRef.current.videoWidth) * 100}%`, 
                  top: `${(faceBox.y / videoRef.current.videoHeight) * 100}%`,
                  width: `${(faceBox.w / videoRef.current.videoWidth) * 100}%`,
                  height: `${(faceBox.h / videoRef.current.videoHeight) * 100}%`,
                  borderColor: facesDetected === 1 ? '#22c55e' : '#ef4444', 
                  borderRadius: '10px'
                }}
              >
                <div className="absolute -top-6 left-0 bg-green-500 text-white text-xs font-bold px-2 py-0.5 rounded shadow">
                  Face Detected
                </div>
              </div>
            )}

            {isStreaming && !isProcessing && (
              <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 transition-opacity whitespace-nowrap">
                {facesDetected === 0 && (
                  <span className="bg-amber-100 text-amber-800 border border-amber-300 text-sm px-4 py-1.5 rounded-full font-medium shadow-sm">
                    No face detected
                  </span>
                )}
                {facesDetected > 1 && (
                  <span className="bg-red-100 text-red-800 border border-red-300 text-sm px-4 py-1.5 rounded-full font-medium shadow-sm flex items-center">
                    Multiple faces detected
                  </span>
                )}
              </div>
            )}

            {isProcessing && (
              <div className="absolute inset-0 z-20 pointer-events-none flex items-center justify-center bg-slate-900/40 backdrop-blur-sm">
                 <div className="bg-white/95 px-6 py-3 rounded-full shadow-xl flex items-center">
                    <div className="w-5 h-5 border-2 border-teal-600 border-t-transparent rounded-full animate-spin mr-3"></div>
                    <span className="text-teal-700 font-bold tracking-wide">Registering Data...</span>
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
          </div>
        </div>

        {/* Success Toast */}
        <div className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[90%] max-w-sm z-30 transition-all duration-500 transform ${registrationResult ? 'scale-100 opacity-100 translate-y-0' : 'scale-95 opacity-0 pointer-events-none translate-y-4'}`}>
          {registrationResult && (
            <div className={`p-6 rounded-2xl shadow-2xl backdrop-blur-xl border ${
              registrationResult.success 
                ? 'bg-emerald-900/90 border-emerald-500/50' 
                : 'bg-rose-900/90 border-rose-500/50'
            }`}>
              <div className="flex flex-col items-center text-center">
                <div className="w-16 h-16 rounded-full flex items-center justify-center mb-4 bg-white/10 text-white">
                  {registrationResult.success ? (
                    <svg className="w-8 h-8 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M5 13l4 4L19 7"></path></svg>
                  ) : (
                    <svg className="w-8 h-8 text-rose-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M6 18L18 6M6 6l12 12"></path></svg>
                  )}
                </div>
                <h3 className="text-xl font-bold mb-2 text-white">
                  {registrationResult.success ? 'Registration Successful!' : 'Registration Failed'}
                </h3>
                <p className="text-white/80">{registrationResult.message}</p>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="mt-8">
        <button
          onClick={captureAndRegister}
          disabled={!isStreaming || isProcessing || facesDetected !== 1 || !employeeName.trim()}
          className={`font-bold text-lg py-4 px-12 rounded-2xl transition-all duration-300 ${
            !isStreaming || isProcessing || facesDetected !== 1 || !employeeName.trim()
              ? 'bg-slate-100 text-slate-400 cursor-not-allowed border border-slate-200' 
              : 'bg-emerald-600 text-white hover:bg-emerald-700 shadow-[0_10px_20px_rgba(16,185,129,0.3)] hover:-translate-y-1 focus:ring-4 focus:ring-emerald-200'
          }`}
        >
          <span className="flex items-center justify-center">
            {isProcessing ? 'Saving to Database...' : 'Register Employee'}
          </span>
        </button>
      </div>
    </div>
  );
};

export default RegisterEmployee;
