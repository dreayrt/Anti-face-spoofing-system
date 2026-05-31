import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as faceapi from '@vladmandic/face-api';
import { recognizeFace } from '../../services/api';
import { evaluateFaceQualityFromCanvas } from '../../utils/faceQuality';

const probabilityConfig = [
  {
    key: 'realPerson',
    label: 'Người thật',
    color: 'from-emerald-400 to-teal-500',
  },
  {
    key: 'spoofAttack',
    label: 'Giả mạo',
    color: 'from-rose-500 to-red-500',
  },
];

const MODEL_PATH = '/models';
const DETECTION_INTERVAL_MS = 180;
const TARGET_VALID_FRAMES = 5;
const MAX_FRAME_ATTEMPTS = 10;
const FRAME_CAPTURE_DELAY_MS = 120;

const DETECTOR_OPTIONS = new faceapi.TinyFaceDetectorOptions({
  inputSize: 416,
  scoreThreshold: 0.3, // Giảm từ 0.5 xuống 0.3 để nhận diện được khuôn mặt trong video tối
});

const SSD_OPTIONS = new faceapi.SsdMobilenetv1Options({
  minConfidence: 0.3,
});

function clampPercent(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, Math.round(value)));
}

function toPercent(value) {
  if (!Number.isFinite(value)) return 0;
  return clampPercent(value * 100);
}

function toFixedNumber(value, digits = 2) {
  if (!Number.isFinite(value)) return 0;
  return Number(value.toFixed(digits));
}

function formatBox(box) {
  if (!box) return null;

  return {
    x: toFixedNumber(box.x),
    y: toFixedNumber(box.y),
    width: toFixedNumber(box.width),
    height: toFixedNumber(box.height),
  };
}

function toDescriptorArray(descriptor) {
  if (!descriptor) return null;
  return Array.from(descriptor, (value) => Number(value));
}

function ProbabilityBar({ label, value, color }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <p className="text-sm font-medium text-slate-200">{label}</p>
        <span className="text-lg font-semibold text-white">{value}%</span>
      </div>
      <div className="h-3 overflow-hidden rounded-full bg-slate-800">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${color} transition-all duration-500`}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}

function sleep(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

export default function CameraPage() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const startLockRef = useRef(false);
  const modelsLoadedRef = useRef(false);
  const detectLoopRef = useRef(null);
  const fileInputRef = useRef(null);
  const uploadVideoRef = useRef(null);
  const uploadCanvasRef = useRef(null);

  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState('');
  const [cameraState, setCameraState] = useState('idle');
  const [isScanning, setIsScanning] = useState(false);
  const [capturedImage, setCapturedImage] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadedImagePreview, setUploadedImagePreview] = useState(null);
  const [modelsReady, setModelsReady] = useState(false);
  const [modelsError, setModelsError] = useState('');
  const [trackingStatus, setTrackingStatus] = useState({
    hasFace: false,
    faceCount: 0,
    ready: false,
    message: 'Đang tải mô hình nhận diện...',
    box: null,
    descriptor: null,
  });

  const [result, setResult] = useState({
    message: 'Chưa có dữ liệu nhận dạng.',
    reason: 'idle',
    liveness_score: null,
    similarity_score: null,
    best_distance: null,
    match_threshold: null,
    success: false,
    user: null,
    frames_used: null,
    matched_votes: null,
    vote_threshold: null,
  });

  const [scanSummary, setScanSummary] = useState({
    validFrames: 0,
    rejectedFrames: 0,
    rejectedReasons: [],
  });

  const probabilities = useMemo(() => {
    const realPerson = toPercent(result.liveness_score ?? 0);
    const spoofAttack = clampPercent(100 - realPerson);

    return {
      realPerson,
      spoofAttack,
    };
  }, [result.liveness_score]);

  const loginStatus = useMemo(() => {
    if (result.reason === 'idle') {
      return {
        label: 'Chưa scan đăng nhập.',
        color: 'text-slate-300',
        card: 'border-white/10 bg-black/20',
      };
    }

    if (result.success && result.user?.name) {
      return {
        label: `Xin chào, ${result.user.name}`,
        color: 'text-emerald-300',
        card: 'border-emerald-500/30 bg-emerald-500/10',
      };
    }

    return {
      label: 'Truy cập bị từ chối',
      color: 'text-rose-300',
      card: 'border-rose-500/30 bg-rose-500/10',
    };
  }, [result.reason, result.success, result.user]);

  const resetTrackingStatus = useCallback((message = 'Đưa khuôn mặt vào camera để bắt đầu.') => {
    setTrackingStatus({
      hasFace: false,
      faceCount: 0,
      ready: false,
      message,
      box: null,
      descriptor: null,
    });
  }, []);

  const stopDetectionLoop = useCallback(() => {
    if (detectLoopRef.current) {
      window.clearTimeout(detectLoopRef.current);
      detectLoopRef.current = null;
    }
  }, []);

  const stopCamera = useCallback(() => {
    stopDetectionLoop();

    const stream = streamRef.current;
    streamRef.current = null;

    if (stream) {
      stream.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch {
          // ignore track stop errors
        }
      });
    }

    if (videoRef.current) {
      try {
        videoRef.current.pause();
        videoRef.current.srcObject = null;
        videoRef.current.removeAttribute('src');
        videoRef.current.load();
      } catch {
        // ignore video cleanup errors
      }
    }

    startLockRef.current = false;
    setCameraReady(false);
    setCameraState('stopped');
    resetTrackingStatus(modelsLoadedRef.current ? 'Camera đã dừng.' : 'Đang tải mô hình nhận diện...');
  }, [resetTrackingStatus, stopDetectionLoop]);

  const loadModels = useCallback(async () => {
    if (modelsLoadedRef.current) return;

    try {
      setModelsError('');
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_PATH),
        faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_PATH),
        faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_PATH),
        faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_PATH),
      ]);
      modelsLoadedRef.current = true;
      setModelsReady(true);
    } catch (error) {
      console.error('Failed to load face detection models:', error);
      setModelsReady(false);
      setModelsError('Không thể tải model nhận diện khuôn mặt từ frontend/public/models.');
      throw error;
    }
  }, []);

  const captureFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas || !video.videoWidth || !video.videoHeight) {
      throw new Error('Camera chưa sẵn sàng để chụp frame.');
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Không thể khởi tạo canvas để chụp frame.');

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    return {
      image: canvas.toDataURL('image/jpeg', 0.98),
      ctx,
      frameWidth: canvas.width,
      frameHeight: canvas.height,
    };
  }, []);

  const runDetection = useCallback(async () => {
    const video = videoRef.current;

    if (
      !modelsLoadedRef.current ||
      !video ||
      video.readyState < 2 ||
      !video.videoWidth ||
      !video.videoHeight
    ) {
      detectLoopRef.current = window.setTimeout(runDetection, DETECTION_INTERVAL_MS);
      return;
    }

    try {
      const detections = await faceapi
        .detectAllFaces(video, SSD_OPTIONS)
        .withFaceLandmarks(true)
        .withFaceDescriptors();

      if (!detections.length) {
        resetTrackingStatus('Không phát hiện khuôn mặt. Hãy nhìn vào camera.');
        detectLoopRef.current = window.setTimeout(runDetection, DETECTION_INTERVAL_MS);
        return;
      }

      if (detections.length > 1) {
        setTrackingStatus({
          hasFace: false,
          faceCount: detections.length,
          ready: false,
          message: 'Phát hiện nhiều khuôn mặt. Chỉ giữ một người trước camera.',
          box: null,
          descriptor: null,
        });
        detectLoopRef.current = window.setTimeout(runDetection, DETECTION_INTERVAL_MS);
        return;
      }

      const detection = detections[0];

      setTrackingStatus({
        hasFace: true,
        faceCount: 1,
        ready: true,
        message: 'Đã phát hiện khuôn mặt.',
        box: formatBox(detection.detection.box),
        descriptor: toDescriptorArray(detection.descriptor),
      });
    } catch (error) {
      console.error('Face tracking failed:', error);
      resetTrackingStatus('Không thể theo dõi khuôn mặt. Hãy thử mở lại camera.');
    }

    detectLoopRef.current = window.setTimeout(runDetection, DETECTION_INTERVAL_MS);
  }, [resetTrackingStatus]);

  const startCamera = useCallback(async () => {
    if (startLockRef.current) return;

    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraState('error');
      setCameraError('Trình duyệt không hỗ trợ getUserMedia.');
      return;
    }

    startLockRef.current = true;
    setUploadedImagePreview(null);

    try {
      setCameraState('requesting');
      setCameraError('');

      await loadModels();
      stopCamera();
      await sleep(250);

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setCameraReady(true);
      setCameraState('ready');
      resetTrackingStatus('Đang dò khuôn mặt...');
      runDetection();
    } catch (error) {
      console.error('Camera start failed:', error);
      setCameraReady(false);
      setCameraState('error');

      if (modelsError) {
        setCameraError(modelsError);
      } else if (error?.name === 'NotAllowedError') {
        setCameraError('Bạn chưa cấp quyền camera cho trình duyệt.');
      } else if (error?.name === 'NotFoundError') {
        setCameraError('Không tìm thấy camera trên thiết bị.');
      } else if (error?.name === 'NotReadableError') {
        setCameraError('Camera đang bị app hoặc tab khác sử dụng. Hãy đóng app khác, bấm "Tắt camera", rồi mở lại.');
      } else {
        setCameraError('Không thể mở camera. Hãy kiểm tra thiết bị hoặc ứng dụng khác đang dùng webcam.');
      }
    } finally {
      startLockRef.current = false;
    }
  }, [loadModels, modelsError, resetTrackingStatus, runDetection, stopCamera]);

  const handleScan = async () => {
    if ((!cameraReady && !uploadedImagePreview) || isScanning || !trackingStatus.ready) return;

    try {
      setIsScanning(true);
      setCameraError('');

      if (uploadedImagePreview && trackingStatus.image) {
        setResult({
          message: 'Đang trích xuất và phân tích khuôn mặt...',
          reason: 'processing',
          liveness_score: null,
          similarity_score: null,
          best_distance: null,
          match_threshold: null,
          success: false,
          user: null,
          frames_used: null,
          matched_votes: null,
          vote_threshold: null,
        });

        try {
          const frameData = await detectFaceOnImage(trackingStatus.image);
          
          setTrackingStatus(prev => ({
            ...prev,
            box: frameData.box,
            descriptor: frameData.descriptor
          }));

          const response = await recognizeFace({
            image: frameData.image,
            box: frameData.box,
            descriptor: frameData.descriptor,
            frames: [frameData],
            vote_min_match: 1,
          });

          setResult({
            message: response.message || 'Đã nhận phản hồi từ hệ thống.',
            reason: response.reason || 'unknown',
            liveness_score: response.liveness_score ?? null,
            similarity_score: response.similarity_score ?? null,
            best_distance: response.best_distance ?? null,
            match_threshold: response.match_threshold ?? null,
            success: !!response.success,
            user: response.user || null,
            frames_used: response.frames_used ?? 1,
            matched_votes: response.matched_votes ?? null,
            vote_threshold: response.vote_threshold ?? 1,
          });
        } catch (err) {
          throw new Error(err.message || 'Lỗi xử lý ảnh tải lên.');
        }
        setIsScanning(false);
        return;
      }

      setResult({
        message: 'Đang thu thập nhiều frame để đăng nhập...',
        reason: 'processing',
        liveness_score: null,
        similarity_score: null,
        best_distance: null,
        match_threshold: null,
        success: false,
        user: null,
        frames_used: null,
        matched_votes: null,
        vote_threshold: null,
      });

      const acceptedFrames = [];
      const rejectedReasons = [];

      for (let attempt = 0; attempt < MAX_FRAME_ATTEMPTS; attempt += 1) {
        if (!trackingStatus.ready || !trackingStatus.descriptor || !trackingStatus.box) {
          await sleep(FRAME_CAPTURE_DELAY_MS);
          continue;
        }

        const frame = captureFrame();
        const quality = evaluateFaceQualityFromCanvas(frame.ctx, frame.frameWidth, frame.frameHeight, trackingStatus.box, {
          minBrightness: 72,
          maxBrightness: 205,
          minContrast: 32,
          minSharpness: 8,
          minFaceAreaRatio: 0.06,
        });

        if (!quality.ok) {
          rejectedReasons.push(...quality.reasons);
          await sleep(FRAME_CAPTURE_DELAY_MS);
          continue;
        }

        acceptedFrames.push({
          image: frame.image,
          box: trackingStatus.box,
          descriptor: trackingStatus.descriptor,
          quality_metrics: quality.metrics,
        });

        setCapturedImage(frame.image);

        if (acceptedFrames.length >= TARGET_VALID_FRAMES) break;

        await sleep(FRAME_CAPTURE_DELAY_MS);
      }

      setScanSummary({
        validFrames: acceptedFrames.length,
        rejectedFrames: Math.max(0, MAX_FRAME_ATTEMPTS - acceptedFrames.length),
        rejectedReasons,
      });

      if (acceptedFrames.length < 3) {
        const reasonText = rejectedReasons.slice(0, 2).join(' ');
        setResult({
          message: `Không đủ frame hợp lệ để đăng nhập (chỉ có ${acceptedFrames.length}). ${reasonText}`.trim(),
          reason: 'poor_quality',
          liveness_score: null,
          similarity_score: null,
          best_distance: null,
          match_threshold: null,
          success: false,
          user: null,
          frames_used: acceptedFrames.length,
          matched_votes: 0,
          vote_threshold: 0,
        });
        return;
      }

      const voteThreshold = Math.max(3, Math.ceil(acceptedFrames.length / 2));

      const response = await recognizeFace({
        image: acceptedFrames[0].image,
        box: acceptedFrames[0].box,
        descriptor: acceptedFrames[0].descriptor,
        frames: acceptedFrames,
        vote_min_match: voteThreshold,
      });

      setResult({
        message: response.message || 'Đã nhận phản hồi từ hệ thống.',
        reason: response.reason || 'unknown',
        liveness_score: response.liveness_score ?? null,
        similarity_score: response.similarity_score ?? null,
        best_distance: response.best_distance ?? null,
        match_threshold: response.match_threshold ?? null,
        success: !!response.success,
        user: response.user || null,
        frames_used: response.frames_used ?? acceptedFrames.length,
        matched_votes: response.matched_votes ?? null,
        vote_threshold: response.vote_threshold ?? voteThreshold,
      });
    } catch (error) {
      const detail = error?.response?.data?.detail;
      setResult({
        message: typeof detail === 'string' ? detail : 'Không thể nhận dữ liệu từ backend/AI.',
        reason: 'request_failed',
        liveness_score: null,
        similarity_score: null,
        best_distance: null,
        match_threshold: null,
        success: false,
        user: null,
        frames_used: null,
        matched_votes: null,
        vote_threshold: null,
      });
    } finally {
      setIsScanning(false);
    }
  };

  const handleUploadClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
      fileInputRef.current.click();
    }
  };

  const processImageFile = async (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  const detectFaceOnImage = async (imageDataUrl) => {
    const img = new Image();
    img.src = imageDataUrl;
    await new Promise((resolve) => { img.onload = resolve; });

    const canvas = uploadCanvasRef.current;
    if (!canvas) throw new Error('Canvas không khả dụng.');

    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    const detections = await faceapi
      .detectAllFaces(canvas, SSD_OPTIONS)
      .withFaceLandmarks(true)
      .withFaceDescriptors();

    if (!detections.length) {
      throw new Error('Không phát hiện khuôn mặt trong ảnh/video.');
    }
    if (detections.length > 1) {
      throw new Error('Phát hiện nhiều khuôn mặt. Chỉ giữ một khuôn mặt.');
    }

    const detection = detections[0];
    return {
      image: canvas.toDataURL('image/jpeg', 0.98),
      box: formatBox(detection.detection.box),
      descriptor: toDescriptorArray(detection.descriptor),
      quality_metrics: null,
    };
  };

  const extractVideoFrames = async (file, maxFrames = 8) => {
    const url = URL.createObjectURL(file);
    const video = uploadVideoRef.current;
    if (!video) throw new Error('Video element không khả dụng.');

    video.src = url;
    video.muted = true;
    await new Promise((resolve, reject) => {
      video.onloadeddata = resolve;
      video.onerror = reject;
    });

    const duration = video.duration;
    if (!duration || duration < 0.5) throw new Error('Video quá ngắn.');

    const interval = duration / (maxFrames + 1);
    const frames = [];

    for (let i = 1; i <= maxFrames; i++) {
      video.currentTime = interval * i;
      await new Promise((resolve) => { video.onseeked = resolve; });
      await sleep(50);

      const canvas = uploadCanvasRef.current;
      if (!canvas) continue;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);

      frames.push(canvas.toDataURL('image/jpeg', 0.98));
    }

    URL.revokeObjectURL(url);
    return frames;
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const isImage = file.type.startsWith('image/');
    const isVideo = file.type.startsWith('video/');

    if (!isImage && !isVideo) {
      setResult({
        message: 'Chỉ hỗ trợ file ảnh hoặc video.',
        reason: 'invalid_file',
        liveness_score: null,
        similarity_score: null,
        best_distance: null,
        match_threshold: null,
        success: false,
        user: null,
        frames_used: null,
        matched_votes: null,
        vote_threshold: null,
      });
      return;
    }

    // Stop current camera if it's running
    stopCamera();
    setUploadedImagePreview(null);

    try {
      setIsUploading(true);
      setResult({
        message: isImage ? 'Đang xử lý ảnh upload...' : 'Đang xử lý video upload...',
        reason: 'processing',
        liveness_score: null,
        similarity_score: null,
        best_distance: null,
        match_threshold: null,
        success: false,
        user: null,
        frames_used: null,
        matched_votes: null,
        vote_threshold: null,
      });

      await loadModels();

      if (isImage) {
        const imageDataUrl = await processImageFile(file);
        setUploadedImagePreview(imageDataUrl);
        setTrackingStatus({
          hasFace: false,
          faceCount: 0,
          ready: true,
          message: 'Ảnh đã sẵn sàng. Hãy nhấn "Scan Face (Vote)".',
          box: null,
          descriptor: null,
          image: imageDataUrl,
        });

        setCapturedImage(imageDataUrl);

        setResult({
          message: 'Ảnh đã sẵn sàng. Hãy nhấn "Scan Face (Vote)" để bắt đầu đăng nhập!',
          reason: 'idle',
          liveness_score: null,
          similarity_score: null,
          best_distance: null,
          match_threshold: null,
          success: false,
          user: null,
          frames_used: null,
          matched_votes: null,
          vote_threshold: null,
        });
      } else {
        // Load video into main viewer for preview and manual scan
        const url = URL.createObjectURL(file);
        if (videoRef.current) {
          videoRef.current.src = url;
          videoRef.current.loop = true;
          try {
            await videoRef.current.play();
            setCameraReady(true);
            setCameraState('ready');
            resetTrackingStatus('Đang dò khuôn mặt trong video...');
            runDetection();
            
            setResult({
              message: 'Video đã được tải lên. Hãy nhấn "Scan Face (Vote)" để kiểm tra Liveness!',
              reason: 'idle',
              liveness_score: null,
              similarity_score: null,
              best_distance: null,
              match_threshold: null,
              success: false,
              user: null,
              frames_used: null,
              matched_votes: null,
              vote_threshold: null,
            });
          } catch (err) {
            throw new Error('Trình duyệt chặn không cho tự động phát video hoặc lỗi định dạng video.');
          }
        }
      }
    } catch (error) {
      const detail = error?.response?.data?.detail;
      setResult({
        message: typeof detail === 'string' ? detail : (error.message || 'Không thể xử lý file upload.'),
        reason: 'upload_failed',
        liveness_score: null,
        similarity_score: null,
        best_distance: null,
        match_threshold: null,
        success: false,
        user: null,
        frames_used: null,
        matched_votes: null,
        vote_threshold: null,
      });
    } finally {
      setIsUploading(false);
    }
  };

  useEffect(() => {
    startCamera();

    return () => {
      stopCamera();
    };
  }, [startCamera, stopCamera]);

  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'hidden') {
        stopCamera();
      }
    };

    const releaseCamera = () => stopCamera();

    document.addEventListener('visibilitychange', handleVisibilityChange);
    window.addEventListener('pagehide', releaseCamera);
    window.addEventListener('beforeunload', releaseCamera);

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('pagehide', releaseCamera);
      window.removeEventListener('beforeunload', releaseCamera);
    };
  }, [stopCamera]);

  const faceBoxStyle =
    trackingStatus.box && videoRef.current?.videoWidth && videoRef.current?.videoHeight
      ? {
          left: `${(trackingStatus.box.x / videoRef.current.videoWidth) * 100}%`,
          top: `${(trackingStatus.box.y / videoRef.current.videoHeight) * 100}%`,
          width: `${(trackingStatus.box.width / videoRef.current.videoWidth) * 100}%`,
          height: `${(trackingStatus.box.height / videoRef.current.videoHeight) * 100}%`,
        }
      : null;

  return (
    <div className="grid gap-6 lg:grid-cols-[1.25fr_0.75fr]">
      <section className="overflow-hidden rounded-3xl border border-white/10 bg-slate-900 shadow-2xl shadow-black/30">
        <div className="border-b border-white/10 px-5 py-4">
          <h1 className="text-xl font-semibold text-white">Camera</h1>
        </div>

        <div className="p-5">
          <div className="relative overflow-hidden rounded-3xl border border-white/10 bg-black">
            <div className="relative aspect-[4/3] w-full bg-slate-950">
              <video 
                ref={videoRef} 
                autoPlay 
                muted 
                playsInline 
                className={`h-full w-full object-cover ${uploadedImagePreview ? 'hidden' : ''}`} 
              />
              
              {uploadedImagePreview && (
                <img 
                  src={uploadedImagePreview} 
                  alt="Uploaded preview" 
                  className="absolute inset-0 h-full w-full object-contain" 
                />
              )}

              {!cameraReady && !uploadedImagePreview && (
                <div className="absolute inset-0 flex items-center justify-center bg-slate-950/90 px-6 text-center">
                  <div>
                    <p className="text-lg font-semibold text-white">{cameraError || modelsError || 'Đang khởi động camera...'}</p>
                    <p className="mt-2 text-sm text-slate-400">
                      Camera sẽ tự bật khi mở app và sẽ dừng khi bạn rời hoặc đóng app.
                    </p>
                  </div>
                </div>
              )}

              <div className="pointer-events-none absolute inset-0">
                {faceBoxStyle && (
                  <div
                    className={`absolute rounded-[20px] border-2 transition-all duration-150 ${
                      trackingStatus.ready ? 'border-emerald-400 bg-emerald-400/10' : 'border-amber-400 bg-amber-300/10'
                    }`}
                    style={faceBoxStyle}
                  />
                )}

                <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/75 via-black/20 to-transparent p-4">
                  <div className="rounded-2xl border border-white/10 bg-black/35 px-4 py-3 backdrop-blur-sm">
                    <p className="text-xs uppercase tracking-[0.18em] text-slate-400">Tracking status</p>
                    <p className={`mt-2 text-sm font-medium ${trackingStatus.ready ? 'text-emerald-300' : 'text-slate-100'}`}>
                      {trackingStatus.message}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-4 flex flex-wrap gap-3">
            <button
              type="button"
              onClick={handleScan}
              disabled={(!cameraReady && !uploadedImagePreview) || isScanning || !trackingStatus.ready}
              className="rounded-2xl bg-gradient-to-r from-blue-500 to-violet-600 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-blue-500/25 transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isScanning ? 'Đang scan multi-frame...' : 'Scan Face (Vote)'}
            </button>
            <button
              type="button"
              onClick={startCamera}
              className="rounded-2xl border border-white/10 bg-white/5 px-5 py-3 text-sm font-semibold text-slate-200 transition hover:bg-white/10"
            >
              Bật / mở lại camera
            </button>
            <button
              type="button"
              onClick={stopCamera}
              className="rounded-2xl border border-white/10 bg-white/5 px-5 py-3 text-sm font-semibold text-slate-200 transition hover:bg-white/10"
            >
              Tắt camera
            </button>
            <button
              type="button"
              onClick={handleUploadClick}
              disabled={isUploading}
              className="rounded-2xl bg-gradient-to-r from-amber-500 to-orange-600 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-amber-500/25 transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isUploading ? 'Đang xử lý...' : '📁 Upload ảnh / video'}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*,video/*"
              onChange={handleFileUpload}
              className="hidden"
            />
          </div>

          <div className="mt-5 grid gap-4 lg:grid-cols-2">
            <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
              <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Camera state</p>
              <p className="mt-2 font-medium text-white">{cameraState}</p>
              <p className="mt-3 text-xs text-slate-400">Models: {modelsReady ? 'ready' : modelsError ? 'error' : 'loading'}</p>
            </div>

            <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
              <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Ảnh vừa chụp</p>
              {capturedImage ? (
                <img src={capturedImage} alt="Captured preview" className="mt-3 aspect-[4/3] w-full rounded-2xl object-cover" />
              ) : (
                <p className="mt-2 text-sm text-slate-400">Chưa có ảnh nào được chụp.</p>
              )}
            </div>
          </div>

          <div className="mt-4 rounded-2xl border border-white/10 bg-black/20 p-4">
            <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Frames quality summary</p>
            <p className="mt-2 text-lg font-semibold text-white">
              Valid: {scanSummary.validFrames} · Rejected: {scanSummary.rejectedFrames}
            </p>
            {scanSummary.rejectedReasons.length > 0 && (
              <p className="mt-2 text-xs text-slate-400">{scanSummary.rejectedReasons.slice(0, 2).join(' ')}</p>
            )}
          </div>

          <canvas ref={canvasRef} className="hidden" />
          <canvas ref={uploadCanvasRef} className="hidden" />
          <video ref={uploadVideoRef} className="hidden" muted playsInline />
        </div>
      </section>

      <section className="space-y-4">
        <div className="rounded-3xl border border-white/10 bg-slate-900 p-5 shadow-2xl shadow-black/30">
          <h2 className="text-xl font-semibold text-white">Thông số nhận dạng</h2>
          <p className="mt-2 text-sm text-slate-400">Panel này hiển thị dữ liệu thật trả về từ backend/AI.</p>

          <div className="mt-5 space-y-4">
            {probabilityConfig.map((item) => (
              <ProbabilityBar key={item.key} label={item.label} value={probabilities[item.key]} color={item.color} />
            ))}
          </div>
        </div>

        <div className="rounded-3xl border border-white/10 bg-slate-900 p-5 shadow-2xl shadow-black/30">
          <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">Kết quả đăng nhập</h3>

          <div className="mt-4 space-y-3 text-sm text-slate-300">
            <div className={`rounded-2xl border p-4 ${loginStatus.card}`}>
              <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Trạng thái</p>
              <p className={`mt-2 text-base font-semibold ${loginStatus.color}`}>{loginStatus.label}</p>
            </div>

            <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
              <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Message</p>
              <p className="mt-2 text-sm text-white">{result.message}</p>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Reason</p>
                <p className="mt-2 font-medium text-white">{result.reason || '-'}</p>
              </div>
              <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Success</p>
                <p className="mt-2 font-medium text-white">{result.success ? 'true' : 'false'}</p>
              </div>
              <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Liveness score</p>
                <p className="mt-2 font-medium text-white">
                  {result.liveness_score != null ? result.liveness_score.toFixed(4) : '-'}
                </p>
              </div>
              <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Best distance</p>
                <p className="mt-2 font-medium text-white">
                  {result.best_distance != null ? result.best_distance.toFixed(4) : '-'}
                </p>
              </div>
              <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Matched votes</p>
                <p className="mt-2 font-medium text-white">
                  {result.matched_votes != null && result.vote_threshold != null
                    ? `${result.matched_votes}/${result.vote_threshold}`
                    : '-'}
                </p>
              </div>
              <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Frames used</p>
                <p className="mt-2 font-medium text-white">{result.frames_used ?? '-'}</p>
              </div>
              <div className="rounded-2xl border border-white/10 bg-black/20 p-4 sm:col-span-2">
                <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Tên nhân viên nhận diện</p>
                <p className="mt-2 text-lg font-semibold text-white">{result.user?.name || '-'}</p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
