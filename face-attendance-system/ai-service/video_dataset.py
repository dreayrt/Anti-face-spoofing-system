"""
video_dataset.py – Video Sequence Dataset for multi-frame anti-spoofing training.

Groups individual frame images into temporal sequences for training the LSTM
component of the CNN+DSP+LSTM model to detect temporal artifacts:
- Flicker between frames
- Inconsistent blur patterns
- Shadow/lighting changes across frames

Supports three data sources:
  - SiW: Real video sequences with naming pattern {type}_{subject}_{frame}.jpg
  - FF-C23: FaceForensics++ sequences with naming {video_id}_frame_{number}.jpg
  - CelebA-Spoof: Single images -> pseudo-sequences (repeated frames)

Usage:
    dataset = VideoSequenceDataset(
        root_dir="dataset/train",
        transform=get_train_transforms(),
        seq_len=5,
        temporal_augmentor=TemporalAugmentor(),
    )

    # Returns (T, C, H, W) tensor + label
    sequence, label = dataset[0]
"""

import os
import re
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


# Class mapping (must match train.py)
LIVE_LABEL = 0
SPOOF_LABEL = 1


class VideoSequenceDataset(Dataset):
    """PyTorch Dataset that groups frame images into video sequences.

    Args:
        root_dir: Path to the split directory (e.g., dataset/train/).
        transform: Torchvision transforms to apply to each frame.
        seq_len: Number of frames per sequence (default: 5).
        sources: List of data sources to include. None = auto-detect.
        temporal_augmentor: Optional callable for temporal augmentation.
                           Signature: augmentor(frames: list[Tensor], label: int) -> list[Tensor]
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root_dir, transform=None, seq_len=5, sources=None,
                 temporal_augmentor=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.seq_len = seq_len
        self.temporal_augmentor = temporal_augmentor
        self.sequences = []  # List of (list_of_paths, label)

        if sources is None:
            sources = [d.name for d in self.root_dir.iterdir() if d.is_dir()]

        for source in sources:
            source_dir = self.root_dir / source
            if not source_dir.exists():
                print(f"  [VideoDataset] Warning: {source_dir} not found, skipping.")
                continue

            for label_name, label in [("live", LIVE_LABEL), ("spoof", SPOOF_LABEL)]:
                label_dir = source_dir / label_name
                if not label_dir.exists():
                    continue

                if source == "celeba-spoof":
                    self._load_single_images(label_dir, label)
                else:
                    self._load_video_sequences(label_dir, label, source)

        print(f"  [VideoDataset] {self.root_dir.name}/: {len(self.sequences)} sequences "
              f"(seq_len={seq_len}, sources={sources})")

        # Count per class
        labels = [label for _, label in self.sequences]
        print(f"  [VideoDataset] live={labels.count(LIVE_LABEL)}, "
              f"spoof={labels.count(SPOOF_LABEL)}")

    # ── Filename Parsers ─────────────────────────────────────────────────

    def _parse_video_id_siw(self, filename):
        """Parse SiW filename: {type}_{subject-session}_{frame}[_dup{n}].jpg"""
        name = filename.stem
        # Remove _dup suffix
        name_clean = re.sub(r'_dup\d+$', '', name)
        # Split by last underscore to separate frame number
        parts = name_clean.rsplit('_', 1)
        if len(parts) == 2:
            try:
                frame_num = int(parts[1])
                return parts[0], frame_num
            except ValueError:
                pass
        return None, None

    def _parse_video_id_ffc23(self, filename):
        """Parse FF-C23 filename: {video_id}_frame_{number}.jpg"""
        name = filename.stem
        match = re.match(r'(.+)_frame_(\d+)', name)
        if match:
            return match.group(1), int(match.group(2))
        return None, None

    # ── Data Loading ─────────────────────────────────────────────────────

    def _load_video_sequences(self, directory, label, source):
        """Group frames into video sequences by video_id."""
        video_groups = {}  # video_id -> [(frame_num, path)]

        parse_fn = self._parse_video_id_siw if source == "SiW" else self._parse_video_id_ffc23

        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in self.IMAGE_EXTENSIONS:
                continue

            video_id, frame_num = parse_fn(path)
            if video_id is None:
                continue

            # Include subdirectory in video_id for uniqueness (e.g., Deepfakes/000_003)
            try:
                subdir = str(path.parent.relative_to(directory))
            except ValueError:
                subdir = "."
            full_video_id = f"{subdir}/{video_id}" if subdir != "." else video_id

            if full_video_id not in video_groups:
                video_groups[full_video_id] = []
            video_groups[full_video_id].append((frame_num, path))

        # Create sequences from each video group
        total_seqs = 0
        for video_id, frames in video_groups.items():
            # Sort by frame number for temporal order
            frames.sort(key=lambda x: x[0])
            paths = [p for _, p in frames]

            if len(paths) < 2:
                # Single frame -> treat as pseudo-sequence
                padded = paths * self.seq_len
                self.sequences.append((padded[:self.seq_len], label))
                total_seqs += 1
            elif len(paths) < self.seq_len:
                # Short sequence -> pad by repeating last frame
                while len(paths) < self.seq_len:
                    paths.append(paths[-1])
                self.sequences.append((paths[:self.seq_len], label))
                total_seqs += 1
            else:
                # Full sequence -> create multiple subsequences via sliding window
                stride = max(1, self.seq_len // 2)  # ~50% overlap
                for start in range(0, len(paths) - self.seq_len + 1, stride):
                    seq_paths = paths[start:start + self.seq_len]
                    self.sequences.append((seq_paths, label))
                    total_seqs += 1

        print(f"    [{source}/{directory.name}] "
              f"{len(video_groups)} videos -> {total_seqs} sequences")

    def _load_single_images(self, directory, label):
        """Create pseudo-sequences from single images (CelebA-Spoof)."""
        images = sorted([
            p for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in self.IMAGE_EXTENSIONS
        ])

        count = 0
        for img_path in images:
            # Repeat single image seq_len times
            self.sequences.append(([img_path] * self.seq_len, label))
            count += 1

        label_name = "live" if label == LIVE_LABEL else "spoof"
        print(f"    [celeba-spoof/{label_name}] "
              f"{count} single images -> {count} pseudo-sequences")

    # ── PyTorch Dataset Interface ────────────────────────────────────────

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        paths, label = self.sequences[idx]

        # Tạo seed 1 lần duy nhất để đảm bảo tất cả frames trong cùng sequence
        # có cùng augmentation (consistent spatial transforms)
        seed = torch.randint(0, 2**31, (1,)).item()

        frames = []
        for i, path in enumerate(paths):
            try:
                image = Image.open(path).convert("RGB")
            except Exception:
                image = Image.new("RGB", (224, 224), (0, 0, 0))

            try:
                if self.transform:
                    if i == 0:
                        # Chỉ set seed 1 lần cho frame đầu tiên
                        random.seed(seed)
                        torch.manual_seed(seed)
                        np.random.seed(seed % (2**32 - 1))
                    else:
                        # Các frame sau dùng cùng seed → cùng augmentation
                        random.seed(seed)
                        torch.manual_seed(seed)
                        np.random.seed(seed % (2**32 - 1))
                    image = self.transform(image)
                else:
                    import torchvision.transforms.functional as TF
                    image = TF.to_tensor(image)
            except Exception:
                # Fallback: black tensor nếu transform lỗi
                image = torch.zeros(3, 224, 224)

            frames.append(image)

        # Apply temporal augmentation (adds inter-frame variation for spoof)
        try:
            if self.temporal_augmentor is not None:
                frames = self.temporal_augmentor(frames, label)
        except Exception:
            pass  # bỏ qua lỗi temporal augmentation

        # Stack into sequence tensor: (T, C, H, W)
        sequence = torch.stack(frames, dim=0)
        return sequence, label


    def get_class_weights(self):
        """Compute inverse-frequency class weights for loss balancing."""
        labels = [label for _, label in self.sequences]
        counts = np.bincount(labels, minlength=2)
        total = len(labels)
        weights = total / (2 * counts.astype(np.float64) + 1e-8)
        return torch.tensor(weights, dtype=torch.float32)
