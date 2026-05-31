export const QUALITY_DEFAULTS = {
  minBrightness: 75,
  maxBrightness: 200,
  minContrast: 35,
  minSharpness: 8,
  minFaceAreaRatio: 0.06,
};

function getBoxDimensions(box) {
  if (!box) return null;

  const x = Number(box.x ?? 0);
  const y = Number(box.y ?? 0);
  const width = Number(box.width ?? box.w ?? 0);
  const height = Number(box.height ?? box.h ?? 0);

  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(width) || !Number.isFinite(height)) {
    return null;
  }

  if (width <= 0 || height <= 0) return null;
  return { x, y, width, height };
}

function computeStats(gray) {
  if (!gray.length) {
    return { brightness: 0, contrast: 0, sharpness: 0 };
  }

  let sum = 0;
  for (let i = 0; i < gray.length; i += 1) sum += gray[i];
  const brightness = sum / gray.length;

  let variance = 0;
  for (let i = 0; i < gray.length; i += 1) {
    const diff = gray[i] - brightness;
    variance += diff * diff;
  }
  const contrast = Math.sqrt(variance / gray.length);

  return { brightness, contrast };
}

export function evaluateFaceQualityFromImageData(imageData, options = {}) {
  const config = { ...QUALITY_DEFAULTS, ...options };

  if (!imageData?.data || !imageData.width || !imageData.height) {
    return {
      ok: false,
      reasons: ['Không thể đọc dữ liệu ảnh để đánh giá chất lượng.'],
      metrics: {
        brightness: 0,
        contrast: 0,
        sharpness: 0,
        faceAreaRatio: 0,
      },
    };
  }

  const { data, width, height } = imageData;
  const gray = new Float32Array(width * height);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = (y * width + x) * 4;
      gray[y * width + x] = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
    }
  }

  const { brightness, contrast } = computeStats(gray);

  let sharpnessSum = 0;
  let sharpnessCount = 0;
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const i = y * width + x;
      const gx = Math.abs(gray[i + 1] - gray[i - 1]);
      const gy = Math.abs(gray[i + width] - gray[i - width]);
      sharpnessSum += gx + gy;
      sharpnessCount += 1;
    }
  }

  const sharpness = sharpnessCount ? sharpnessSum / sharpnessCount : 0;

  const reasons = [];
  if (brightness < config.minBrightness) reasons.push('Ánh sáng quá tối.');
  if (brightness > config.maxBrightness) reasons.push('Ánh sáng quá gắt/cháy sáng.');
  if (contrast < config.minContrast) reasons.push('Độ tương phản thấp, hãy đổi góc sáng.');
  if (sharpness < config.minSharpness) reasons.push('Ảnh mờ, hãy giữ yên khuôn mặt 1-2 giây.');

  return {
    ok: reasons.length === 0,
    reasons,
    metrics: {
      brightness: Number(brightness.toFixed(2)),
      contrast: Number(contrast.toFixed(2)),
      sharpness: Number(sharpness.toFixed(2)),
      faceAreaRatio: 0,
    },
  };
}

export function evaluateFaceQualityFromCanvas(canvasContext, frameWidth, frameHeight, box, options = {}) {
  const config = { ...QUALITY_DEFAULTS, ...options };
  const b = getBoxDimensions(box);

  if (!canvasContext || !b || !frameWidth || !frameHeight) {
    return {
      ok: false,
      reasons: ['Không xác định được vùng khuôn mặt để kiểm tra chất lượng.'],
      metrics: {
        brightness: 0,
        contrast: 0,
        sharpness: 0,
        faceAreaRatio: 0,
      },
    };
  }

  const x = Math.max(0, Math.floor(b.x));
  const y = Math.max(0, Math.floor(b.y));
  const width = Math.max(1, Math.floor(Math.min(b.width, frameWidth - x)));
  const height = Math.max(1, Math.floor(Math.min(b.height, frameHeight - y)));

  if (width <= 0 || height <= 0) {
    return {
      ok: false,
      reasons: ['Không xác định được vùng khuôn mặt hợp lệ.'],
      metrics: {
        brightness: 0,
        contrast: 0,
        sharpness: 0,
        faceAreaRatio: 0,
      },
    };
  }

  const faceAreaRatio = (width * height) / (frameWidth * frameHeight);
  const imageData = canvasContext.getImageData(x, y, width, height);
  const quality = evaluateFaceQualityFromImageData(imageData, config);

  const reasons = [...quality.reasons];
  if (faceAreaRatio < config.minFaceAreaRatio) {
    reasons.push('Khuôn mặt quá nhỏ trong khung hình, vui lòng tiến gần camera.');
  }

  return {
    ok: reasons.length === 0,
    reasons,
    metrics: {
      ...quality.metrics,
      faceAreaRatio: Number(faceAreaRatio.toFixed(4)),
    },
  };
}
