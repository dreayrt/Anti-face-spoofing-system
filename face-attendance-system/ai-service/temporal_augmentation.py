"""
temporal_augmentation.py – Temporal augmentation for anti-spoofing video sequences.

Applies inter-frame variations to spoof sequences during training, teaching
the LSTM to detect temporal artifacts that indicate spoofing:
  - Brightness flicker (different brightness per frame)
  - Color temperature shifts (channel-wise variation)
  - Temporal noise (different noise patterns per frame)

Live sequences are kept temporally consistent (no temporal augmentation),
so the model learns: consistent = real, inconsistent = spoof.

Usage:
    augmentor = TemporalAugmentor()
    augmented_frames = augmentor(frames, label=1)  # Only augments spoof (label=1)
"""

import random
import torch

# Class labels (must match train.py)
LIVE_LABEL = 0
SPOOF_LABEL = 1


class TemporalAugmentor:
    """Applies temporal augmentation to frame sequences for anti-spoofing training.

    Only augments SPOOF sequences. LIVE sequences pass through unchanged,
    so the model learns that temporal consistency = real.

    Args:
        flicker_prob: Probability of applying brightness flicker.
        color_shift_prob: Probability of applying color temperature shift.
        noise_prob: Probability of applying per-frame noise.
        flicker_intensity: Max brightness offset for flicker (in normalized space).
        color_intensity: Max color channel offset.
        noise_intensity: Std dev of Gaussian noise.
    """

    def __init__(
        self,
        flicker_prob=0.5,
        color_shift_prob=0.3,
        noise_prob=0.3,
        flicker_intensity=0.12,
        color_intensity=0.06,
        noise_intensity=0.025,
    ):
        self.flicker_prob = flicker_prob
        self.color_shift_prob = color_shift_prob
        self.noise_prob = noise_prob
        self.flicker_intensity = flicker_intensity
        self.color_intensity = color_intensity
        self.noise_intensity = noise_intensity

    def __call__(self, frames, label):
        """Apply temporal augmentation to a sequence of frames.

        Args:
            frames: List of tensors, each (C, H, W), after spatial transforms.
            label: Class label (0=live, 1=spoof).

        Returns:
            List of (possibly augmented) tensors.
        """
        if label == LIVE_LABEL:
            # Live sequences: keep temporally consistent — no augmentation
            return frames

        # Spoof sequences: add temporal artifacts
        augmented = list(frames)  # shallow copy

        if random.random() < self.flicker_prob:
            augmented = self._apply_flicker(augmented)

        if random.random() < self.color_shift_prob:
            augmented = self._apply_color_shift(augmented)

        if random.random() < self.noise_prob:
            augmented = self._apply_noise(augmented)

        return augmented

    def _apply_flicker(self, frames):
        """Simulate brightness flicker by adding different offsets per frame.

        Real-world spoofing via screen replay often shows brightness
        fluctuations caused by the screen refresh rate and camera exposure.
        """
        result = []
        for frame in frames:
            offset = random.gauss(0, self.flicker_intensity)
            result.append(frame + offset)
        return result

    def _apply_color_shift(self, frames):
        """Simulate color temperature shift by varying individual channels per frame.

        Screen replay and print attacks often show subtle color shifts
        between frames due to white balance differences and screen color
        reproduction inaccuracies.
        """
        result = []
        for frame in frames:
            shifted = frame.clone()
            # Randomly shift 1-2 channels
            n_channels = random.randint(1, 2)
            channels = random.sample([0, 1, 2], n_channels)
            for ch in channels:
                shifted[ch] = shifted[ch] + random.gauss(0, self.color_intensity)
            result.append(shifted)
        return result

    def _apply_noise(self, frames):
        """Add different Gaussian noise patterns to each frame.

        Different noise patterns per frame simulate the sensor noise
        differences that occur when recording a screen or printed photo,
        as opposed to the consistent noise pattern in live capture.
        """
        result = []
        for frame in frames:
            noise = torch.randn_like(frame) * self.noise_intensity
            result.append(frame + noise)
        return result
