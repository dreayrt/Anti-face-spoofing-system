import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as faceapi from '@vladmandic/face-api';
import { registerEmployee } from '../../services/api';
import { evaluateFaceQualityFromCanvas } from '../../utils/faceQuality';

const MODEL_PATH = '/models';
const DETECTION_INTERVAL_MS = 200;
const MIN_REGISTER_SAMPLES = 5;
const MAX_REGISTER_SAMPLES = 10;

const DETECTOR_OPTIONS = new faceapi.TinyFaceDetectorOptions({
  inputSize: 416,
  scoreThreshold: 0.5,
});

const SSD_OPTIONS = new faceapi.SsdMobilenetv1Options({
  minConfidence: 0.3,
});

function generateEmployeeId() {
  const timestamp = Date.now().toString(36).toUpperCase();
  const random = Math.random().toString(36).slice(2, 6).toUpperCase();
  return `EMP-${timestamp}-${random}`;
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

function averageDescriptors(descriptors) {
  if (!descriptors.length) return null;

  const length = descriptors[0]?.length || 0;
  if (!length) return null;

  const sums = new Array(length).fill(0);
  descriptors.forEach((descriptor) => {
    descriptor.forEach((value, index) => {
      sums[index] += Number(value);
    });
  });

  return sums.map((value) => Number((value / descriptors.length).toFixed(8)));
}

export default function RegisterEmployee() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const detectLoopRef = useRef(null);
  const startLockRef = useRef(false);
  const modelsLoadedRef = useRef(false);
  const fileInputRef = useRef(null);
  const videoInputRef = useRef(null);

  const [employeeId, setEmployeeId] = useState('');
  const [name, setName] = useState('');

  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState('');

  const [isScanning, setIsScanning] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  const [scanError, setScanError] = useState('');
  const [saveError, setSaveError] = useState('');
  const [saveSuccess, setSaveSuccess] = useState('');

  const [samples, setSamples] = useState([]);
  const [capturedImage, setCapturedImage] = useState('');
  const [uploadedImage, setUploadedImage] = useState('');
  const [uploadedImageSize, setUploadedImageSize] = useState(null);
  const [lastQualityMetrics, setLastQualityMetrics] = useState(null);

  const [faceStatus, setFaceStatus] = useState({
    detected: false,
    count: 0,
    message: 'Đang khởi tạo camera...',
    box: null,
    descriptor: null,
  });

  const hasEnoughSamples = useMemo(() => samples.length >= MIN_REGISTER_SAMPLES, [samples.length]);

  const resetFaceStatus = useCallback((message = 'Đưa khuôn mặt vào camera để scan.') => {
    setFaceStatus({
      detected: false,
      count: 0,
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

  const cleanupCurrentStream = useCallback(() => {
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
  }, []);

  const stopCamera = useCallback(() => {
    stopDetectionLoop();
    cleanupCurrentStream();
    startLockRef.current = false;
    setCameraReady(false);
    resetFaceStatus(modelsLoadedRef.current ? 'Camera đã dừng.' : 'Đang tải model nhận diện...');
  }, [cleanupCurrentStream, resetFaceStatus, stopDetectionLoop]);

  const loadModels = useCallback(async () => {
    if (modelsLoadedRef.current) return;

    try {
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_PATH),
        faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_PATH),
        faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_PATH),
        faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_PATH),
      ]);
      modelsLoadedRef.current = true;
    } catch (error) {
      console.error('Failed to load models:', error);
      setCameraError('Không thể tải model nhận diện khuôn mặt.');
      throw error;
    }
  }, []);

  const captureFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas || !video.videoWidth || !video.videoHeight) {
      throw new Error('Camera chưa sẵn sàng để chụp ảnh.');
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Không thể khởi tạo canvas để chụp ảnh.');

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const image = canvas.toDataURL('image/jpeg', 0.92);

    return {
      image,
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
        resetFaceStatus('Không phát hiện khuôn mặt. Hãy nhìn thẳng vào camera.');
        detectLoopRef.current = window.setTimeout(runDetection, DETECTION_INTERVAL_MS);
        return;
      }

      if (detections.length > 1) {
        setFaceStatus({
          detected: false,
          count: detections.length,
          message: 'Phát hiện nhiều khuôn mặt. Chỉ giữ 1 người trong khung hình.',
          box: null,
          descriptor: null,
        });
        detectLoopRef.current = window.setTimeout(runDetection, DETECTION_INTERVAL_MS);
        return;
      }

      const detection = detections[0];
      setFaceStatus({
        detected: true,
        count: 1,
        message: 'Đã phát hiện khuôn mặt. Nhấn Scan để lấy mẫu.',
        box: formatBox(detection.detection.box),
        descriptor: toDescriptorArray(detection.descriptor),
      });
    } catch (error) {
      console.error('Face detection failed:', error);
      resetFaceStatus('Lỗi khi theo dõi khuôn mặt. Hãy thử mở lại camera.');
    }

    detectLoopRef.current = window.setTimeout(runDetection, DETECTION_INTERVAL_MS);
  }, [resetFaceStatus]);

  const startCamera = useCallback(async () => {
    if (startLockRef.current) return;

    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError('Trình duyệt không hỗ trợ camera.');
      return;
    }

    startLockRef.current = true;

    try {
      setCameraError('');
      await loadModels();

      stopDetectionLoop();
      cleanupCurrentStream();

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
      resetFaceStatus('Đang dò khuôn mặt...');
      runDetection();
    } catch (error) {
      console.error('Camera start failed:', error);
      setCameraReady(false);

      if (error?.name === 'NotAllowedError') {
        setCameraError('Bạn chưa cấp quyền camera cho trình duyệt.');
      } else if (error?.name === 'NotFoundError') {
        setCameraError('Không tìm thấy camera trên thiết bị.');
      } else if (error?.name === 'NotReadableError') {
        setCameraError('Camera đang bị ứng dụng khác sử dụng.');
      } else if (error?.name === 'AbortError') {
        setCameraError('Camera bị gián đoạn khi khởi động. Vui lòng bấm mở lại camera.');
      } else {
        setCameraError('Không thể mở camera. Vui lòng kiểm tra lại webcam.');
      }
    } finally {
      startLockRef.current = false;
    }
  }, [cleanupCurrentStream, loadModels, resetFaceStatus, runDetection, stopDetectionLoop]);

  const handleScanFace = useCallback(async () => {
    const hasSource = cameraReady || uploadedImage;
    if (!hasSource || isScanning) return;

    if (!faceStatus.detected || !faceStatus.descriptor || !faceStatus.box) {
      setScanError('Chưa phát hiện khuôn mặt hợp lệ để scan.');
      return;
    }

    if (samples.length >= MAX_REGISTER_SAMPLES) {
      setScanError(`Đã đủ tối đa ${MAX_REGISTER_SAMPLES} mẫu. Bạn có thể lưu ngay.`);
      return;
    }

    try {
      setIsScanning(true);
      setScanError('');
      setSaveError('');
      setSaveSuccess('');

      let image, ctx, frameWidth, frameHeight;

      if (cameraReady) {
        // Camera mode: capture from video
        const frame = captureFrame();
        image = frame.image;
        ctx = frame.ctx;
        frameWidth = frame.frameWidth;
        frameHeight = frame.frameHeight;
      } else if (uploadedImage) {
        // Upload mode: draw uploaded image to canvas
        const canvas = canvasRef.current;
        if (!canvas || !uploadedImageSize) throw new Error('Không thể xử lý ảnh đã tải lên.');

        canvas.width = uploadedImageSize.width;
        canvas.height = uploadedImageSize.height;
        ctx = canvas.getContext('2d');

        const img = new Image();
        img.src = uploadedImage;
        await new Promise((resolve) => {
          if (img.complete && img.naturalWidth) resolve();
          else img.onload = resolve;
        });

        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        image = canvas.toDataURL('image/jpeg', 0.92);
        frameWidth = canvas.width;
        frameHeight = canvas.height;
      }

      const quality = evaluateFaceQualityFromCanvas(ctx, frameWidth, frameHeight, faceStatus.box, {
        minBrightness: 72,
        maxBrightness: 205,
        minContrast: 32,
        minSharpness: 8,
        minFaceAreaRatio: 0.06,
      });

      setLastQualityMetrics(quality.metrics);

      if (!quality.ok) {
        setScanError(`Mẫu chưa đạt chất lượng: ${quality.reasons.join(' ')}`);
        return;
      }


      setCapturedImage(image);
      setSamples((prev) => [
        ...prev,
        {
          image,
          descriptor: faceStatus.descriptor,
          quality: quality.metrics,
        },
      ]);
    } catch (error) {
      console.error('Scan failed:', error);
      setScanError(error?.message || 'Lỗi khi scan gương mặt. Vui lòng thử lại.');
    } finally {
      setIsScanning(false);
    }
  }, [cameraReady, captureFrame, faceStatus, isScanning, samples.length, uploadedImage, uploadedImageSize]);

  const handleUploadVideo = useCallback(async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    event.target.value = '';

    try {
      setCameraError('');
      await loadModels();

      stopDetectionLoop();
      cleanupCurrentStream();

      const videoUrl = URL.createObjectURL(file);
      
      if (videoRef.current) {
        videoRef.current.srcObject = null;
        videoRef.current.src = videoUrl;
        videoRef.current.loop = true;
        await videoRef.current.play();
      }

      setCameraReady(true);
      resetFaceStatus('Đang dò khuôn mặt từ video...');
      runDetection();
    } catch (error) {
      console.error('Video upload failed:', error);
      setCameraError('Không thể phát video đã tải lên.');
      setCameraReady(false);
    }
  }, [cleanupCurrentStream, loadModels, resetFaceStatus, runDetection, stopDetectionLoop]);

  const handleUploadImage = useCallback(async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Reset input so the same file can be selected again
    event.target.value = '';

    if (samples.length >= MAX_REGISTER_SAMPLES) {
      setScanError(`Đã đủ tối đa ${MAX_REGISTER_SAMPLES} mẫu. Bạn có thể lưu ngay.`);
      return;
    }

    try {
      setScanError('');
      setSaveError('');
      setSaveSuccess('');
      resetFaceStatus('Đang xử lý ảnh tải lên...');

      // Step 1: Load and display the image IMMEDIATELY in the camera preview
      const imageUrl = URL.createObjectURL(file);
      const img = new Image();
      img.src = imageUrl;
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = () => reject(new Error('Không thể tải ảnh. Vui lòng chọn ảnh hợp lệ.'));
      });

      // Show preview right away (before model loading)
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      setUploadedImageSize({ width: img.width, height: img.height });
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      const base64Image = canvas.toDataURL('image/jpeg', 0.92);
      setUploadedImage(base64Image);

      // Step 2: Load models (may take time on first call)
      await loadModels();

      // Step 3: Detect face and update tracking status (don't auto-add sample)
      let detections = await faceapi
        .detectAllFaces(img, DETECTOR_OPTIONS)
        .withFaceLandmarks(true)
        .withFaceDescriptors();

      // Fallback to SsdMobilenetv1 if TinyFaceDetector fails (common for cropped dataset images)
      if (!detections.length) {
        detections = await faceapi
          .detectAllFaces(img, SSD_OPTIONS)
          .withFaceLandmarks(true)
          .withFaceDescriptors();
      }

      URL.revokeObjectURL(imageUrl);

      if (!detections.length) {
        resetFaceStatus('Không phát hiện khuôn mặt trong ảnh tải lên.');
        setScanError('Không phát hiện khuôn mặt trong ảnh tải lên.');
        return;
      }

      if (detections.length > 1) {
        resetFaceStatus('Phát hiện nhiều khuôn mặt.');
        setScanError('Phát hiện nhiều khuôn mặt trong ảnh tải lên. Chỉ chấp nhận ảnh có 1 người.');
        return;
      }

      const detection = detections[0];
      const box = formatBox(detection.detection.box);
      const descriptor = toDescriptorArray(detection.descriptor);

      // Update face status for tracking box overlay - user can then click Scan
      setFaceStatus({
        detected: true,
        count: 1,
        message: 'Đã phát hiện khuôn mặt. Nhấn Scan để lấy mẫu.',
        box,
        descriptor,
      });
    } catch (error) {
      console.error('Upload failed:', error);
      setScanError(error?.message || 'Lỗi khi xử lý ảnh tải lên. Vui lòng thử lại.');
    }
  }, [loadModels, resetFaceStatus, samples.length]);

  const handleClearSamples = useCallback(() => {
    setSamples([]);
    setCapturedImage('');
    setScanError('');
    setSaveError('');
    setSaveSuccess('');
    setLastQualityMetrics(null);
    resetFaceStatus('Đã xóa mẫu. Đưa khuôn mặt vào camera để scan lại.');
  }, [resetFaceStatus]);

  const handleRemoveLastSample = useCallback(() => {
    setSamples((prev) => prev.slice(0, -1));
  }, []);

  const handleGenerateNewId = useCallback(() => {
    setEmployeeId(generateEmployeeId());
  }, []);

  const handleSave = useCallback(async () => {
    if (isSaving) return;

    setSaveError('');
    setSaveSuccess('');

    const trimmedName = name.trim();
    if (!trimmedName) {
      setSaveError('Vui lòng nhập tên nhân viên.');
      return;
    }

    if (!hasEnoughSamples) {
      setSaveError(`Vui lòng scan ít nhất ${MIN_REGISTER_SAMPLES} mẫu trước khi lưu.`);
      return;
    }

    try {
      setIsSaving(true);

      const prototype = averageDescriptors(samples.map((sample) => sample.descriptor));
      if (!prototype || prototype.length !== 128) {
        setSaveError('Không thể tạo descriptor đại diện từ các mẫu đã scan.');
        return;
      }

      const payload = {
        id: employeeId,
        name: trimmedName,
        samples,
        face_image_base64: samples[0].image,
        face_descriptor: prototype,
        image: samples[0].image,
        descriptor: prototype,
      };

      const response = await registerEmployee(payload);

      if (!response?.success) {
        setSaveError(response?.message || 'Đăng ký thất bại.');
        return;
      }

      setSaveSuccess(
        `Đăng ký thành công nhân viên ${trimmedName} (${employeeId}) với ${response.sample_count || samples.length} mẫu.`
      );
      setName('');
      setEmployeeId(generateEmployeeId());
      setCapturedImage('');
      setSamples([]);
      setScanError('');
      setLastQualityMetrics(null);
    } catch (error) {
      const detail = error?.response?.data?.detail;
      const message =
        typeof detail === 'string'
          ? detail
          : error?.response?.data?.message || 'Không thể kết nối backend để đăng ký.';
      setSaveError(message);
    } finally {
      setIsSaving(false);
    }
  }, [employeeId, hasEnoughSamples, isSaving, name, samples]);

  useEffect(() => {
    setEmployeeId(generateEmployeeId());
  }, []);

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

  // Use video dimensions when camera is active, or uploaded image dimensions as fallback
  const refWidth = cameraReady ? videoRef.current?.videoWidth : uploadedImageSize?.width;
  const refHeight = cameraReady ? videoRef.current?.videoHeight : uploadedImageSize?.height;

  const faceBoxStyle =
    faceStatus.box && refWidth && refHeight
      ? {
          left: `${(faceStatus.box.x / refWidth) * 100}%`,
          top: `${(faceStatus.box.y / refHeight) * 100}%`,
          width: `${(faceStatus.box.width / refWidth) * 100}%`,
          height: `${(faceStatus.box.height / refHeight) * 100}%`,
        }
      : null;

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_1.2fr]">
      <section className="rounded-[28px] border border-white/10 bg-white/5 p-6 shadow-2xl shadow-slate-950/30 backdrop-blur-xl">
        <div className="inline-flex items-center gap-2 rounded-full border border-violet-400/25 bg-violet-500/10 px-3 py-1 text-xs font-semibold text-violet-200">
          Đăng ký nhân viên (multi-sample)
        </div>

        <h2 className="mt-4 text-2xl font-bold tracking-tight text-white">Form đăng ký</h2>

        <div className="mt-8 space-y-5">
          <label className="block">
            <span className="mb-2 flex items-center gap-2 text-sm font-medium text-slate-200">
              <span>ID nhân viên</span>
              <span className="rounded-full bg-amber-500/20 px-2 py-0.5 text-xs text-amber-300">Tự động</span>
            </span>

            <div className="flex items-center gap-3 rounded-2xl border border-white/10 bg-slate-950/40 px-4 py-3">
              <span className="text-slate-400">#</span>
              <input
                type="text"
                value={employeeId}
                readOnly
                className="w-full flex-1 bg-transparent font-mono text-sm text-violet-300 outline-none"
              />
              <button
                type="button"
                onClick={handleGenerateNewId}
                className="rounded-xl bg-white/5 px-3 py-1.5 text-xs font-medium text-slate-300 transition hover:bg-white/10"
              >
                Tạo mới
              </button>
            </div>
          </label>

          <label className="block">
            <span className="mb-2 block text-sm font-medium text-slate-200">Tên nhân viên</span>
            <div className="flex items-center gap-3 rounded-2xl border border-white/10 bg-slate-950/40 px-4 py-3">
              <input
                type="text"
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="Nhập họ và tên nhân viên"
                className="w-full bg-transparent text-white outline-none placeholder:text-slate-500"
              />
            </div>
          </label>

          <div className="rounded-2xl border border-white/10 bg-slate-950/40 p-4">
            <p className="text-sm font-medium text-white">Tiến độ lấy mẫu</p>
            <p className={`mt-1 text-sm ${hasEnoughSamples ? 'text-emerald-300' : 'text-amber-300'}`}>
              Đã lấy {samples.length}/{MIN_REGISTER_SAMPLES} mẫu tối thiểu.
            </p>
            {lastQualityMetrics && (
              <p className="mt-2 text-xs text-slate-400">
                Brightness: {lastQualityMetrics.brightness} · Contrast: {lastQualityMetrics.contrast} · Sharpness:{' '}
                {lastQualityMetrics.sharpness} · Face area: {lastQualityMetrics.faceAreaRatio}
              </p>
            )}
          </div>

          {scanError && (
            <div className="rounded-2xl border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-300">
              {scanError}
            </div>
          )}

          <div className="grid grid-cols-2 gap-3">
            <button
              type="button"
              onClick={handleScanFace}
              disabled={(!cameraReady && !uploadedImage) || isScanning || !faceStatus.detected || isSaving || samples.length >= MAX_REGISTER_SAMPLES}
              className="w-full rounded-2xl bg-gradient-to-r from-blue-500 to-violet-600 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-blue-500/25 transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isScanning ? 'Đang xử lý...' : `Scan mẫu (${samples.length}/${MAX_REGISTER_SAMPLES})`}
            </button>
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isScanning || isSaving || samples.length >= MAX_REGISTER_SAMPLES}
              className="w-full rounded-2xl border border-blue-500/30 bg-blue-500/10 px-4 py-3 text-sm font-semibold text-blue-300 transition hover:bg-blue-500/20 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Tải ảnh lên
            </button>
          </div>
          <input
            type="file"
            accept="image/*"
            ref={fileInputRef}
            onChange={handleUploadImage}
            className="hidden"
          />

          <div className="grid grid-cols-2 gap-3">
            <button
              type="button"
              onClick={handleRemoveLastSample}
              disabled={!samples.length || isSaving}
              className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm font-medium text-slate-200 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Xóa mẫu cuối
            </button>
            <button
              type="button"
              onClick={handleClearSamples}
              disabled={!samples.length || isSaving}
              className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm font-medium text-slate-200 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Xóa toàn bộ mẫu
            </button>
          </div>

          <button
            type="button"
            onClick={handleSave}
            disabled={isSaving || !hasEnoughSamples}
            className="w-full rounded-2xl bg-gradient-to-r from-emerald-500 to-teal-600 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-emerald-500/25 transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isSaving ? 'Đang lưu...' : `Lưu thông tin nhân viên (${samples.length}/${MIN_REGISTER_SAMPLES})`}
          </button>

          {saveSuccess && (
            <div className="rounded-2xl border border-emerald-500/30 bg-emerald-500/10 p-4 text-sm text-emerald-300">
              {saveSuccess}
            </div>
          )}

          {saveError && (
            <div className="rounded-2xl border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-300">
              {saveError}
            </div>
          )}
        </div>
      </section>

      <section className="rounded-[28px] border border-white/10 bg-white/5 p-6 shadow-2xl shadow-blue-950/20 backdrop-blur-xl">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.3em] text-slate-400">Camera preview</p>
            <h3 className="mt-3 text-xl font-bold text-white">Khung scan gương mặt</h3>
            <p className="mt-2 text-sm leading-6 text-slate-300">
              Mỗi lần scan sẽ kiểm tra chất lượng ảnh. Hệ thống khuyên lấy mẫu dưới các điều kiện sáng khác nhau.
            </p>
          </div>
          <div
            className={`rounded-2xl px-4 py-3 text-right text-sm ${
              faceStatus.detected
                ? 'border border-emerald-400/25 bg-emerald-500/10 text-emerald-200'
                : 'border border-amber-400/25 bg-amber-500/10 text-amber-200'
            }`}
          >
            <p className="font-semibold">{faceStatus.count > 0 ? `${faceStatus.count} khuôn mặt` : 'Đang chờ'}</p>
            <p className="mt-1 text-xs">{faceStatus.detected ? 'Sẵn sàng scan' : 'Đang dò'}</p>
          </div>
        </div>

        <div className="relative mt-6 overflow-hidden rounded-[24px] border border-white/10 bg-slate-950/50">
          <div className="aspect-[4/3] w-full">
            <video ref={videoRef} autoPlay muted playsInline className={`h-full w-full object-cover${!cameraReady && uploadedImage ? ' hidden' : ''}`} />

            {!cameraReady && uploadedImage && (
              <img src={uploadedImage} alt="Uploaded preview" className="h-full w-full object-cover" />
            )}

            {!cameraReady && !uploadedImage && (
              <div className="absolute inset-0 flex items-center justify-center bg-slate-950/90 px-4 text-center">
                <div>
                  <p className="text-base font-semibold text-white">{cameraError || 'Đang khởi động camera...'}</p>
                  <p className="mt-2 text-sm text-slate-400">Vui lòng cấp quyền camera để tiếp tục.</p>
                </div>
              </div>
            )}

            {faceBoxStyle && (
              <div
                className={`absolute rounded-[20px] border-4 transition-all duration-200 ${
                  faceStatus.detected
                    ? 'border-emerald-400 bg-emerald-400/10 shadow-lg shadow-emerald-400/30'
                    : 'border-amber-400 bg-amber-400/10'
                }`}
                style={faceBoxStyle}
              />
            )}

            <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 via-black/30 to-transparent p-4">
              <div className="rounded-2xl border border-white/10 bg-black/50 px-4 py-3 backdrop-blur-sm">
                <p className={`text-sm font-medium ${faceStatus.detected ? 'text-emerald-300' : 'text-slate-200'}`}>
                  {faceStatus.message}
                </p>
              </div>
            </div>
          </div>
        </div>

        <canvas ref={canvasRef} className="hidden" />

        {capturedImage && (
          <div className="mt-4 rounded-2xl border border-white/10 bg-slate-950/40 p-4">
            <p className="text-xs uppercase tracking-[0.25em] text-slate-400">Ảnh mẫu gần nhất</p>
            <img src={capturedImage} alt="Captured face" className="mt-2 aspect-video w-full rounded-xl object-cover" />
          </div>
        )}

        <div className="mt-4 flex gap-2">
          <button
            type="button"
            onClick={startCamera}
            className="flex-1 rounded-2xl border border-white/10 bg-white/5 px-4 py-2.5 text-sm font-medium text-slate-200 transition hover:bg-white/10"
          >
            Mở lại camera
          </button>
          <button
            type="button"
            onClick={stopCamera}
            className="flex-1 rounded-2xl border border-white/10 bg-white/5 px-4 py-2.5 text-sm font-medium text-slate-200 transition hover:bg-white/10"
          >
            Tắt camera
          </button>
          <button
            type="button"
            onClick={() => videoInputRef.current?.click()}
            className="flex-1 rounded-2xl border border-white/10 bg-white/5 px-4 py-2.5 text-sm font-medium text-slate-200 transition hover:bg-white/10"
          >
            Tải video lên
          </button>
          <input
            type="file"
            accept="video/*"
            ref={videoInputRef}
            onChange={handleUploadVideo}
            className="hidden"
          />
        </div>
      </section>
    </div>
  );
}
