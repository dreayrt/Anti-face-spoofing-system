"""
dsp_utils.py – Digital Signal Processing utilities for anti-spoofing.

Provides frequency-domain analysis tools to detect artifacts typical of
spoofed face images (print attacks, replay attacks):
  - Moiré patterns from printed photos
  - Banding / aliasing from screen replay
  - High-frequency noise differences between real and fake faces

Core functions:
  - compute_fft_2d()          : 2D FFT on a single-channel image
  - compute_power_spectrum()   : Magnitude spectrum from FFT
  - azimuthal_average()        : Radial average of power spectrum
  - extract_frequency_features(): End-to-end image → frequency feature vector
"""

import numpy as np
import torch
import torch.nn as nn


def compute_fft_2d(image_gray: np.ndarray) -> np.ndarray:
    """Compute the 2D Fast Fourier Transform of a grayscale image.

    Args:
        image_gray: Grayscale image as numpy array (H, W), values in [0, 255].

    Returns:
        Complex-valued FFT result shifted so that DC component is at center.
    """
    # Apply windowing (Hann) to reduce spectral leakage at edges
    h, w = image_gray.shape
    window_h = np.hanning(h)
    window_w = np.hanning(w)
    window_2d = np.outer(window_h, window_w)

    windowed = image_gray.astype(np.float64) * window_2d

    # 2D FFT + shift DC to center
    fft_result = np.fft.fft2(windowed)
    fft_shifted = np.fft.fftshift(fft_result)

    return fft_shifted


def compute_power_spectrum(fft_shifted: np.ndarray) -> np.ndarray:
    """Compute the log-scaled power spectrum (magnitude²) from an FFT result.

    Args:
        fft_shifted: Complex FFT result with DC at center (output of compute_fft_2d).

    Returns:
        Log-scaled power spectrum as float64 array of same shape.
    """
    magnitude = np.abs(fft_shifted)
    power = magnitude ** 2

    # Log scale to compress dynamic range (add eps to avoid log(0))
    log_power = np.log1p(power)

    return log_power


def azimuthal_average(power_spectrum: np.ndarray) -> np.ndarray:
    """Compute the radial (azimuthal) average of a 2D power spectrum.

    This reduces the 2D spectrum to a 1D frequency profile, where each
    value represents the average energy at a given spatial frequency.

    Args:
        power_spectrum: 2D power spectrum array (H, W).

    Returns:
        1D array of radially averaged power values, length = min(H, W) // 2.
    """
    h, w = power_spectrum.shape
    cy, cx = h // 2, w // 2

    # Create distance matrix from center
    y_coords, x_coords = np.ogrid[:h, :w]
    distances = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)
    distances = distances.astype(int)

    max_radius = min(cy, cx)
    radial_avg = np.zeros(max_radius)

    for r in range(max_radius):
        mask = distances == r
        if np.any(mask):
            radial_avg[r] = np.mean(power_spectrum[mask])

    return radial_avg


def extract_frequency_features(image_gray: np.ndarray, feature_dim: int = 64) -> np.ndarray:
    """End-to-end pipeline: grayscale image → frequency feature vector.

    Steps:
      1. Compute 2D FFT
      2. Compute log power spectrum
      3. Compute azimuthal (radial) average
      4. Resample to fixed-length feature vector

    Args:
        image_gray: Grayscale image (H, W), uint8 or float.
        feature_dim: Length of the output feature vector.

    Returns:
        1D numpy array of shape (feature_dim,) — frequency features.
    """
    fft_shifted = compute_fft_2d(image_gray)
    power = compute_power_spectrum(fft_shifted)
    radial_profile = azimuthal_average(power)

    # Resample radial profile to fixed length using linear interpolation
    x_old = np.linspace(0, 1, len(radial_profile))
    x_new = np.linspace(0, 1, feature_dim)
    features = np.interp(x_new, x_old, radial_profile)

    # Normalize to zero mean, unit variance
    mean = features.mean()
    std = features.std() + 1e-8
    features = (features - mean) / std

    return features.astype(np.float32)


class DSPModule(nn.Module):
    """PyTorch module that extracts frequency-domain features from CNN feature maps.

    Takes the intermediate feature maps from a CNN backbone and applies
    2D FFT analysis to detect frequency artifacts characteristic of
    spoofed images.

    Architecture:
        Input feature map (B, C, H, W)
            → Channel-wise FFT 2D → Power Spectrum
            → Flatten spatial dims → Conv1D compression
            → AdaptiveAvgPool1D → output_dim-D vector

    Args:
        input_channels: Number of channels in the input feature map (e.g., 1280 for MobileNetV2).
        output_dim: Dimension of the output frequency feature vector (default: 256).
    """

    def __init__(self, input_channels: int = 1280, output_dim: int = 256):
        super().__init__()
        self.input_channels = input_channels
        self.output_dim = output_dim

        # Conv1D layers to compress frequency information
        self.freq_conv = nn.Sequential(
            nn.Conv1d(input_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, feature_maps: torch.Tensor) -> torch.Tensor:
        """Extract frequency features from CNN feature maps.

        Args:
            feature_maps: Tensor of shape (B, C, H, W) from CNN backbone.

        Returns:
            Tensor of shape (B, output_dim) — frequency feature vector.
        """
        B, C, H, W = feature_maps.shape

        # Compute 2D FFT on each channel of each sample
        # torch.fft.fft2 handles complex arithmetic on GPU
        fft_result = torch.fft.fft2(feature_maps.float())
        fft_shifted = torch.fft.fftshift(fft_result, dim=(-2, -1))

        # Power spectrum (magnitude squared), log-scaled
        power = torch.abs(fft_shifted) ** 2
        log_power = torch.log1p(power)  # (B, C, H, W)

        # Flatten spatial dimensions: (B, C, H*W)
        freq_features = log_power.reshape(B, C, H * W)

        # Conv1D compression: (B, C, H*W) → (B, output_dim, H*W)
        freq_features = self.freq_conv(freq_features)

        # Global average pooling: (B, output_dim, H*W) → (B, output_dim)
        freq_features = self.pool(freq_features).squeeze(-1)

        return freq_features
