"""
test.py – Test/Evaluation script for Swin Transformer Baseline Anti-Spoofing Model.

Evaluates the trained model on the held-out test set and generates
comprehensive metrics charts comparable to the training/validation charts.

Hai chế độ đánh giá:
  [1] Single-frame mode (default): Mỗi ảnh được xử lý độc lập.
      LSTM chạy với seq_len=1 → KHÔNG khai thác temporal pattern.

  [2] Multi-frame mode (--multi-frame): Nhóm các ảnh liên tiếp cùng video
      thành sequences T=seq_len frames. LSTM xử lý đúng temporal sequence
      → khai thác đầy đủ temporal artifacts.

Usage:
    cd ai-service

    # Single-frame (cũ — LSTM seq_len=1)
    python test.py
    python test.py --checkpoint models/weights/antispoof_cnn_dsp_lstm.pth

    # Multi-frame (LSTM đúng — khuyến nghị nếu train với --multi-frame)
    python test.py --multi-frame
    python test.py --multi-frame --seq-len 5
    python test.py --multi-frame --seq-len 8 --batch-size 16

Output:
    test_logs/                                       — Test evaluation charts
        ├── test_confusion_matrix.png                — Confusion matrix heatmap
        ├── test_classification_report.png           — Per-class P/R/F1 bar chart
        ├── test_roc_curve.png                       — ROC curve với AUC
        ├── test_precision_recall_curve.png          — Precision-Recall curve
        ├── test_score_distribution.png              — Score histogram (live vs spoof)
        ├── test_per_source_metrics.png              — Metrics breakdown per data source
        ├── test_overview.png                        — Combined 2x3 overview dashboard
        └── test_results.json                        — Full test metrics in JSON
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI required)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    classification_report, accuracy_score
)

# Add inference dir to path so we can import the model
from model import SwinTransformerBaseline

# Import VideoSequenceDataset cho multi-frame mode
from video_dataset import VideoSequenceDataset


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
WEIGHTS_DIR = Path(__file__).resolve().parent / "models" / "weights"
TEST_LOGS_DIR = Path(__file__).resolve().parent / "test_logs"

# ImageNet normalization (same as preprocessing pipeline)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

# Class mapping
CLASS_NAMES = ["live", "spoof"]
LIVE_LABEL = 0
SPOOF_LABEL = 1


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset (reuse from train.py)
# ═══════════════════════════════════════════════════════════════════════════════

class AntiSpoofDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for anti-spoofing evaluation.

    Loads images from the preprocessed dataset directory structure:
        dataset/{split}/{source}/{live,spoof}/

    Args:
        root_dir: Path to the split directory (e.g., dataset/test/).
        transform: Torchvision transforms to apply.
        sources: List of data sources to include.
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root_dir: Path, transform=None, sources=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []  # List of (image_path, label, source_name)

        if sources is None:
            # Auto-detect available sources
            sources = [d.name for d in self.root_dir.iterdir() if d.is_dir()]

        for source in sources:
            source_dir = self.root_dir / source
            if not source_dir.exists():
                print(f"  [Dataset] Warning: {source_dir} not found, skipping.")
                continue

            # Scan live/ directory
            live_dir = source_dir / "live"
            if live_dir.exists():
                for img_path in self._scan_images(live_dir):
                    self.samples.append((img_path, LIVE_LABEL, source))

            # Scan spoof/ directory (may have subdirs for FF-C23 methods)
            spoof_dir = source_dir / "spoof"
            if spoof_dir.exists():
                for img_path in self._scan_images(spoof_dir):
                    self.samples.append((img_path, SPOOF_LABEL, source))

        print(f"  [Dataset] Loaded {len(self.samples)} images from {self.root_dir.name}/ "
              f"(sources: {sources})")

    def _scan_images(self, directory: Path):
        """Recursively scan directory for image files."""
        images = []
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in self.IMAGE_EXTENSIONS:
                images.append(path)
        return images

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, source = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Return a black image if file is corrupt
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label, source


# ═══════════════════════════════════════════════════════════════════════════════
# Transforms
# ═══════════════════════════════════════════════════════════════════════════════

def get_eval_transforms():
    """Evaluation transforms (deterministic — no augmentation)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# Custom collate functions
# ═══════════════════════════════════════════════════════════════════════════════

def test_collate_fn(batch):
    """Custom collate cho single-frame: (image, label, source) tuples."""
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    sources = [item[2] for item in batch]
    return images, labels, sources


def test_multiframe_collate_fn(batch):
    """Custom collate cho multi-frame: (sequence, label) tuples từ VideoSequenceDataset.

    VideoSequenceDataset trả về (T, C, H, W) sequence + label (không có source).
    Ở đây ta thêm source là chuỗi rỗng để giữ interface thống nhất.
    """
    sequences = torch.stack([item[0] for item in batch])  # (B, T, C, H, W)
    labels = torch.tensor([item[1] for item in batch])
    return sequences, labels


# ═══════════════════════════════════════════════════════════════════════════════
# Test Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_test(model, dataloader, device, resume=False):
    """Single-frame evaluation (LSTM seq_len=1).

    Mỗi ảnh xử lý độc lập. LSTM chạy nhưng chỉ với 1 timestep.
    Dùng cho backward compatibility và so sánh.

    Returns:
        all_probs: np.array of P(live) probabilities for each sample.
        all_preds: np.array of predicted labels (0=live, 1=spoof).
        all_labels: np.array of ground truth labels.
        all_sources: list of source names for each sample.
    """
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    all_sources = []

    resume_file = TEST_LOGS_DIR / "test_resume_single.pt"
    start_batch = 0
    if resume and resume_file.exists():
        try:
            state = torch.load(resume_file, weights_only=False)
            start_batch = state['processed_batches']
            all_probs = state['probs']
            all_preds = state['preds']
            all_labels = state['labels']
            all_sources = state['sources']
            print(f"  [Resume] Resuming Single-Frame eval from batch {start_batch} ({len(all_probs)} samples)")
        except Exception as e:
            print(f"  [Resume] Failed to load checkpoint: {e}. Bắt đầu lại từ đầu.")
            start_batch = 0

    pbar = tqdm(dataloader, desc="[Test-SingleFrame] Evaluating", leave=True)

    for i, (images, labels, sources) in enumerate(pbar):
        if i < start_batch:
            continue

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)                      # forward() → seq_len=1
        probs = torch.softmax(logits, dim=1)        # (B, 2)

        live_probs = probs[:, 0].cpu().numpy()      # P(live)
        _, predicted = torch.max(logits, 1)

        all_probs.extend(live_probs)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_sources.extend(sources)

        # Cập nhật checkpoint mỗi 50 batches
        if (i + 1) % 50 == 0:
            torch.save({
                'processed_batches': i + 1,
                'probs': all_probs,
                'preds': all_preds,
                'labels': all_labels,
                'sources': all_sources
            }, resume_file)

    # Xóa file resume sau khi chạy xong
    if resume_file.exists():
        try:
            resume_file.unlink()
        except:
            pass

    return (
        np.array(all_probs),
        np.array(all_preds),
        np.array(all_labels),
        all_sources,
    )


@torch.no_grad()
def evaluate_test_multiframe(model, dataloader, device, seq_len, resume=False):
    """Multi-frame evaluation — LSTM xử lý đúng T frames temporal sequence.

    Sử dụng model.forward_multi_frame() với input (B, T, 3, 224, 224).
    LSTM có T timesteps để học temporal patterns:
      - Flicker giữa các frame
      - Inconsistent blur
      - Shadow/lighting thay đổi

    Args:
        model: Trained CNNDSPLSTMAntiSpoof model.
        dataloader: DataLoader từ VideoSequenceDataset.
        device: torch.device.
        seq_len: Số frame mỗi sequence (T).

    Returns:
        all_probs: np.array of P(live) probabilities cho mỗi sequence.
        all_preds: np.array of predicted labels (0=live, 1=spoof).
        all_labels: np.array of ground truth labels.
        all_sources: list rỗng (VideoSequenceDataset không track source riêng).
    """
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    resume_file = TEST_LOGS_DIR / "test_resume_multi.pt"
    start_batch = 0
    if resume and resume_file.exists():
        try:
            state = torch.load(resume_file, weights_only=False)
            start_batch = state['processed_batches']
            all_probs = state['probs']
            all_preds = state['preds']
            all_labels = state['labels']
            print(f"  [Resume] Resuming Multi-Frame eval from batch {start_batch} ({len(all_probs)} samples)")
        except Exception as e:
            print(f"  [Resume] Failed to load checkpoint: {e}. Bắt đầu lại từ đầu.")
            start_batch = 0

    pbar = tqdm(
        dataloader,
        desc=f"[Test-MultiFrame T={seq_len}] Evaluating",
        leave=True
    )

    for i, (sequences, labels) in enumerate(pbar):
        if i < start_batch:
            continue

        # sequences: (B, T, 3, 224, 224)
        # labels:    (B,)
        sequences = sequences.to(device)
        labels = labels.to(device)

        # Gọi forward_multi_frame để LSTM xử lý T timesteps
        logits = model.forward_multi_frame(sequences)   # (B, 2)
        probs = torch.softmax(logits, dim=1)            # (B, 2)

        live_probs = probs[:, 0].cpu().numpy()          # P(live)
        _, predicted = torch.max(logits, 1)

        all_probs.extend(live_probs)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Cập nhật progress bar với accuracy tức thời
        batch_acc = (predicted.cpu() == labels.cpu()).float().mean().item()
        pbar.set_postfix(batch_acc=f"{batch_acc:.3f}")

        # Cập nhật checkpoint mỗi 50 batches
        if (i + 1) % 50 == 0:
            torch.save({
                'processed_batches': i + 1,
                'probs': all_probs,
                'preds': all_preds,
                'labels': all_labels
            }, resume_file)

    # Xóa file resume sau khi chạy xong
    if resume_file.exists():
        try:
            resume_file.unlink()
        except:
            pass

    return (
        np.array(all_probs),
        np.array(all_preds),
        np.array(all_labels),
        [],  # no per-source tracking in VideoSequenceDataset
    )


def compute_test_metrics(preds, labels, probs=None):
    """Compute comprehensive test metrics."""
    # Per-class precision, recall, f1
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1], zero_division=0
    )

    # Overall accuracy
    acc = accuracy_score(labels, preds)

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    # Classification report as dict
    cls_report = classification_report(
        labels, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )

    metrics = {
        "accuracy": round(float(acc), 4),
        "total_samples": int(len(labels)),
        "confusion_matrix": cm.tolist(),
    }

    for i, cls_name in enumerate(CLASS_NAMES):
        metrics[cls_name] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    # AUC if probabilities available
    if probs is not None:
        # For ROC: P(live) as score, live=positive class (label=0)
        # We invert: spoof_probs = 1 - live_probs, with spoof as positive
        spoof_probs = 1.0 - probs
        fpr, tpr, _ = roc_curve(labels, spoof_probs, pos_label=1)
        roc_auc = auc(fpr, tpr)
        metrics["roc_auc"] = round(float(roc_auc), 4)

        # Average Precision
        ap = average_precision_score(labels, spoof_probs, pos_label=1)
        metrics["average_precision"] = round(float(ap), 4)

    # Macro/Weighted averages
    metrics["macro_avg"] = {
        "precision": round(float(cls_report["macro avg"]["precision"]), 4),
        "recall": round(float(cls_report["macro avg"]["recall"]), 4),
        "f1": round(float(cls_report["macro avg"]["f1-score"]), 4),
    }
    metrics["weighted_avg"] = {
        "precision": round(float(cls_report["weighted avg"]["precision"]), 4),
        "recall": round(float(cls_report["weighted avg"]["recall"]), 4),
        "f1": round(float(cls_report["weighted avg"]["f1-score"]), 4),
    }

    return metrics


def compute_per_source_metrics(preds, labels, sources):
    """Compute metrics broken down by data source."""
    unique_sources = sorted(set(sources))
    source_metrics = {}

    sources_arr = np.array(sources)

    for source in unique_sources:
        mask = sources_arr == source
        src_preds = preds[mask]
        src_labels = labels[mask]

        if len(src_labels) == 0:
            continue

        precision, recall, f1, support = precision_recall_fscore_support(
            src_labels, src_preds, average=None, labels=[0, 1], zero_division=0
        )
        acc = accuracy_score(src_labels, src_preds)
        cm = confusion_matrix(src_labels, src_preds, labels=[0, 1])

        source_metrics[source] = {
            "accuracy": round(float(acc), 4),
            "total_samples": int(len(src_labels)),
            "live_count": int((src_labels == 0).sum()),
            "spoof_count": int((src_labels == 1).sum()),
            "confusion_matrix": cm.tolist(),
        }

        for i, cls_name in enumerate(CLASS_NAMES):
            source_metrics[source][cls_name] = {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }

    return source_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Test Visualization Charts
# ═══════════════════════════════════════════════════════════════════════════════

# ── Dark Theme Style Setup ────────────────────────────────────────────────────
COLORS = {
    'train': '#2196F3',      # Blue
    'val': '#FF5722',        # Red-Orange
    'test': '#00E676',       # Green accent
    'live_p': '#4CAF50',     # Green
    'live_r': '#8BC34A',     # Light Green
    'live_f1': '#009688',    # Teal
    'spoof_p': '#E91E63',    # Pink
    'spoof_r': '#FF9800',    # Orange
    'spoof_f1': '#9C27B0',   # Purple
    'roc': '#00BCD4',        # Cyan
    'pr': '#FF7043',         # Deep Orange
    'live_hist': '#4CAF50',  # Green
    'spoof_hist': '#F44336', # Red
    'bg': '#1a1a2e',         # Dark background
    'grid': '#333355',       # Grid lines
    'text': '#e0e0e0',       # Text color
    'accent': '#7C4DFF',     # Purple accent
}


def style_ax(ax, title, xlabel, ylabel):
    """Apply dark theme styling to an axis."""
    ax.set_facecolor(COLORS['bg'])
    ax.set_title(title, fontsize=14, fontweight='bold', color=COLORS['text'], pad=12)
    ax.set_xlabel(xlabel, fontsize=11, color=COLORS['text'])
    ax.set_ylabel(ylabel, fontsize=11, color=COLORS['text'])
    ax.tick_params(colors=COLORS['text'], labelsize=9)
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.legend(fontsize=9, framealpha=0.7, facecolor='#2a2a4a', edgecolor=COLORS['grid'],
              labelcolor=COLORS['text'])


def plot_test_confusion_matrix(cm, output_dir, title_suffix=""):
    """Plot and save a confusion matrix heatmap for test results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cm_array = np.array(cm)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    # Heatmap
    im = ax.imshow(cm_array, interpolation='nearest', cmap='YlOrRd')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(colors=COLORS['text'], labelsize=9)
    cbar.outline.set_edgecolor(COLORS['grid'])

    # Labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES, fontsize=12, color=COLORS['text'])
    ax.set_yticklabels(CLASS_NAMES, fontsize=12, color=COLORS['text'])
    ax.set_xlabel('Predicted', fontsize=13, color=COLORS['text'])
    ax.set_ylabel('Actual', fontsize=13, color=COLORS['text'])

    title = f'Test Confusion Matrix{title_suffix}'
    ax.set_title(title, fontsize=15, fontweight='bold', color=COLORS['text'], pad=12)

    # Cell values + percentages
    total = cm_array.sum()
    for i in range(2):
        for j in range(2):
            val = cm_array[i, j]
            pct = val / total * 100
            color = 'white' if val > cm_array.max() / 2 else 'black'
            ax.text(j, i, f'{val}\n({pct:.1f}%)', ha='center', va='center',
                    fontsize=16, fontweight='bold', color=color)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])

    fig.tight_layout()
    fig.savefig(output_dir / 'test_confusion_matrix.png', dpi=150, facecolor=COLORS['bg'])
    plt.close(fig)


def plot_classification_report(metrics, output_dir):
    """Plot per-class Precision/Recall/F1 as a grouped bar chart."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS['bg'])

    classes = CLASS_NAMES
    x = np.arange(len(classes))
    width = 0.25

    precisions = [metrics[c]["precision"] for c in classes]
    recalls = [metrics[c]["recall"] for c in classes]
    f1s = [metrics[c]["f1"] for c in classes]

    bars1 = ax.bar(x - width, precisions, width, label='Precision',
                   color=COLORS['live_p'], alpha=0.9, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x, recalls, width, label='Recall',
                   color=COLORS['spoof_r'], alpha=0.9, edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + width, f1s, width, label='F1-Score',
                   color=COLORS['accent'], alpha=0.9, edgecolor='white', linewidth=0.5)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color=COLORS['text'])

    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in classes], fontsize=12, color=COLORS['text'])
    ax.set_ylim(0, 1.15)

    # Add overall accuracy text
    acc = metrics.get("accuracy", 0)
    ax.text(0.98, 0.95, f'Overall Accuracy: {acc:.4f}',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            color=COLORS['test'], ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#2a2a4a',
                      edgecolor=COLORS['test'], alpha=0.8))

    style_ax(ax, 'Test — Classification Report (per Class)', 'Class', 'Score')
    fig.tight_layout()
    fig.savefig(output_dir / 'test_classification_report.png', dpi=150, facecolor=COLORS['bg'])
    plt.close(fig)


def plot_roc_curve(labels, probs, output_dir):
    """Plot ROC curve with AUC."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Spoof as positive class
    spoof_probs = 1.0 - probs  # P(spoof)
    fpr, tpr, thresholds = roc_curve(labels, spoof_probs, pos_label=1)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(COLORS['bg'])

    # ROC curve
    ax.plot(fpr, tpr, color=COLORS['roc'], linewidth=2.5,
            label=f'ROC curve (AUC = {roc_auc:.4f})')

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], color=COLORS['grid'], linewidth=1.5,
            linestyle='--', label='Random (AUC = 0.5000)', alpha=0.7)

    # Fill under curve
    ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS['roc'])

    # Find and mark optimal point (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx], color='#FF5722',
               s=120, zorder=5, edgecolors='white', linewidth=2)
    ax.annotate(f'Optimal\nFPR={fpr[optimal_idx]:.3f}\nTPR={tpr[optimal_idx]:.3f}\nTh={optimal_threshold:.3f}',
                xy=(fpr[optimal_idx], tpr[optimal_idx]),
                xytext=(30, -40), textcoords='offset points',
                fontsize=9, color=COLORS['text'],
                arrowprops=dict(arrowstyle='->', color='#FF5722', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#2a2a4a',
                          edgecolor='#FF5722', alpha=0.8))

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    style_ax(ax, 'Test — ROC Curve (Spoof Detection)', 'False Positive Rate', 'True Positive Rate')
    fig.tight_layout()
    fig.savefig(output_dir / 'test_roc_curve.png', dpi=150, facecolor=COLORS['bg'])
    plt.close(fig)

    return roc_auc


def plot_precision_recall_curve(labels, probs, output_dir):
    """Plot Precision-Recall curve."""
    output_dir.mkdir(parents=True, exist_ok=True)

    spoof_probs = 1.0 - probs  # P(spoof)
    precision_vals, recall_vals, thresholds = precision_recall_curve(labels, spoof_probs, pos_label=1)
    ap = average_precision_score(labels, spoof_probs, pos_label=1)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(COLORS['bg'])

    ax.plot(recall_vals, precision_vals, color=COLORS['pr'], linewidth=2.5,
            label=f'Precision-Recall (AP = {ap:.4f})')

    # Fill under curve
    ax.fill_between(recall_vals, precision_vals, alpha=0.15, color=COLORS['pr'])

    # Baseline (random classifier) — proportion of positive class
    baseline = (labels == 1).sum() / len(labels)
    ax.axhline(y=baseline, color=COLORS['grid'], linewidth=1.5,
               linestyle='--', label=f'Baseline ({baseline:.3f})', alpha=0.7)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    style_ax(ax, 'Test — Precision-Recall Curve (Spoof Detection)',
             'Recall', 'Precision')
    fig.tight_layout()
    fig.savefig(output_dir / 'test_precision_recall_curve.png', dpi=150, facecolor=COLORS['bg'])
    plt.close(fig)

    return ap


def plot_score_distribution(labels, probs, output_dir, threshold=0.5):
    """Plot histogram of liveness scores for live vs spoof samples."""
    output_dir.mkdir(parents=True, exist_ok=True)

    live_scores = probs[labels == 0]   # P(live) for actual live faces
    spoof_scores = probs[labels == 1]  # P(live) for actual spoof faces

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS['bg'])

    bins = np.linspace(0, 1, 50)
    ax.hist(live_scores, bins=bins, alpha=0.65, color=COLORS['live_hist'],
            label=f'Live (n={len(live_scores)})', edgecolor='white', linewidth=0.5)
    ax.hist(spoof_scores, bins=bins, alpha=0.65, color=COLORS['spoof_hist'],
            label=f'Spoof (n={len(spoof_scores)})', edgecolor='white', linewidth=0.5)

    # Decision threshold line
    ax.axvline(x=threshold, color='#FFD600', linewidth=2.5, linestyle='--',
               label=f'Threshold = {threshold:.3f}', alpha=0.9)

    # Annotate regions
    ax.text(threshold + 0.02, ax.get_ylim()[1] * 0.9, '← SPOOF',
            fontsize=11, fontweight='bold', color=COLORS['spoof_hist'], alpha=0.8)
    ax.text(threshold - 0.02, ax.get_ylim()[1] * 0.9, 'LIVE →',
            fontsize=11, fontweight='bold', color=COLORS['live_hist'], alpha=0.8, ha='right')

    ax.set_xlim(-0.02, 1.02)
    style_ax(ax, 'Test — Liveness Score Distribution', 'P(live) Score', 'Count')
    fig.tight_layout()
    fig.savefig(output_dir / 'test_score_distribution.png', dpi=150, facecolor=COLORS['bg'])
    plt.close(fig)


def plot_per_source_metrics(source_metrics, output_dir):
    """Plot per-source accuracy and F1 scores as a grouped bar chart."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = list(source_metrics.keys())
    if len(sources) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(COLORS['bg'])
    fig.suptitle('Test — Per-Source Performance Breakdown', fontsize=15,
                 fontweight='bold', color=COLORS['text'], y=1.02)

    # ── Chart 1: Accuracy per source ──────────────────────────────────────
    ax = axes[0]
    accs = [source_metrics[s]["accuracy"] for s in sources]
    # Palette hỗ trợ đến 5 nguồn (CelebA + FF-C23 + SiW + ...)
    source_palette = [COLORS['roc'], COLORS['pr'], COLORS['accent'],
                      COLORS['live_p'], COLORS['spoof_p']]
    bars = ax.bar(sources, accs, color=source_palette[:len(sources)],
                  alpha=0.9, edgecolor='white', linewidth=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                f'{acc:.4f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color=COLORS['text'])
    ax.set_ylim(0, 1.15)
    style_ax(ax, 'Accuracy per Source', 'Data Source', 'Accuracy')

    # ── Chart 2: Per-class F1 per source ──────────────────────────────────
    ax = axes[1]
    x = np.arange(len(sources))
    width = 0.35
    live_f1s = [source_metrics[s]["live"]["f1"] for s in sources]
    spoof_f1s = [source_metrics[s]["spoof"]["f1"] for s in sources]

    bars1 = ax.bar(x - width / 2, live_f1s, width, label='Live F1',
                   color=COLORS['live_p'], alpha=0.9, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width / 2, spoof_f1s, width, label='Spoof F1',
                   color=COLORS['spoof_p'], alpha=0.9, edgecolor='white', linewidth=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color=COLORS['text'])

    ax.set_xticks(x)
    ax.set_xticklabels(sources, fontsize=11, color=COLORS['text'])
    ax.set_ylim(0, 1.15)
    style_ax(ax, 'F1-Score per Source & Class', 'Data Source', 'F1-Score')

    fig.tight_layout()
    fig.savefig(output_dir / 'test_per_source_metrics.png', dpi=150, facecolor=COLORS['bg'])
    plt.close(fig)


def plot_test_overview(metrics, labels, probs, preds, cm, output_dir, threshold=0.5):
    """Generate a combined 2x3 overview dashboard for test results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.patch.set_facecolor(COLORS['bg'])
    fig.suptitle('Swin Transformer Baseline Anti-Spoofing — Test Evaluation Overview',
                 fontsize=18, fontweight='bold', color=COLORS['text'], y=0.98)

    # ── 1. Confusion Matrix ───────────────────────────────────────────────
    ax = axes[0, 0]
    cm_array = np.array(cm)
    im = ax.imshow(cm_array, interpolation='nearest', cmap='YlOrRd')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(colors=COLORS['text'], labelsize=8)
    cbar.outline.set_edgecolor(COLORS['grid'])
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES, fontsize=10, color=COLORS['text'])
    ax.set_yticklabels(CLASS_NAMES, fontsize=10, color=COLORS['text'])
    ax.set_xlabel('Predicted', fontsize=10, color=COLORS['text'])
    ax.set_ylabel('Actual', fontsize=10, color=COLORS['text'])
    total = cm_array.sum()
    for i in range(2):
        for j in range(2):
            val = cm_array[i, j]
            pct = val / total * 100
            color = 'white' if val > cm_array.max() / 2 else 'black'
            ax.text(j, i, f'{val}\n({pct:.1f}%)', ha='center', va='center',
                    fontsize=14, fontweight='bold', color=color)
    ax.set_facecolor(COLORS['bg'])
    ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold',
                 color=COLORS['text'], pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])

    # ── 2. Classification Report (Bar) ────────────────────────────────────
    ax = axes[0, 1]
    classes = CLASS_NAMES
    x = np.arange(len(classes))
    width = 0.25
    precisions = [metrics[c]["precision"] for c in classes]
    recalls = [metrics[c]["recall"] for c in classes]
    f1s = [metrics[c]["f1"] for c in classes]
    ax.bar(x - width, precisions, width, label='Precision',
           color=COLORS['live_p'], alpha=0.9, edgecolor='white', linewidth=0.5)
    ax.bar(x, recalls, width, label='Recall',
           color=COLORS['spoof_r'], alpha=0.9, edgecolor='white', linewidth=0.5)
    ax.bar(x + width, f1s, width, label='F1-Score',
           color=COLORS['accent'], alpha=0.9, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in classes], fontsize=10, color=COLORS['text'])
    ax.set_ylim(0, 1.12)
    style_ax(ax, 'Per-Class Metrics', 'Class', 'Score')

    # ── 3. ROC Curve ──────────────────────────────────────────────────────
    ax = axes[0, 2]
    spoof_probs = 1.0 - probs
    fpr, tpr, _ = roc_curve(labels, spoof_probs, pos_label=1)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=COLORS['roc'], linewidth=2,
            label=f'ROC (AUC={roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color=COLORS['grid'], linewidth=1, linestyle='--', alpha=0.7)
    ax.fill_between(fpr, tpr, alpha=0.12, color=COLORS['roc'])
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    style_ax(ax, 'ROC Curve', 'FPR', 'TPR')

    # ── 4. Precision-Recall Curve ─────────────────────────────────────────
    ax = axes[1, 0]
    prec_vals, rec_vals, _ = precision_recall_curve(labels, spoof_probs, pos_label=1)
    ap = average_precision_score(labels, spoof_probs, pos_label=1)
    ax.plot(rec_vals, prec_vals, color=COLORS['pr'], linewidth=2,
            label=f'PR (AP={ap:.4f})')
    ax.fill_between(rec_vals, prec_vals, alpha=0.12, color=COLORS['pr'])
    baseline = (labels == 1).sum() / len(labels)
    ax.axhline(y=baseline, color=COLORS['grid'], linewidth=1, linestyle='--', alpha=0.7)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    style_ax(ax, 'Precision-Recall Curve', 'Recall', 'Precision')

    # ── 5. Score Distribution ─────────────────────────────────────────────
    ax = axes[1, 1]
    live_scores = probs[labels == 0]
    spoof_scores = probs[labels == 1]
    bins = np.linspace(0, 1, 40)
    ax.hist(live_scores, bins=bins, alpha=0.6, color=COLORS['live_hist'],
            label=f'Live (n={len(live_scores)})', edgecolor='white', linewidth=0.5)
    ax.hist(spoof_scores, bins=bins, alpha=0.6, color=COLORS['spoof_hist'],
            label=f'Spoof (n={len(spoof_scores)})', edgecolor='white', linewidth=0.5)
    ax.axvline(x=threshold, color='#FFD600', linewidth=2, linestyle='--',
               label=f'Threshold={threshold:.3f}', alpha=0.9)
    style_ax(ax, 'Score Distribution', 'P(live) Score', 'Count')

    # ── 6. Summary Stats ─────────────────────────────────────────────────
    ax = axes[1, 2]
    ax.set_facecolor(COLORS['bg'])
    ax.axis('off')

    summary_lines = [
        ("OVERALL METRICS", "", True),
        ("", "", False),
        ("Accuracy", f"{metrics['accuracy']:.4f}"),
        ("ROC AUC", f"{metrics.get('roc_auc', 'N/A')}"),
        ("Avg Precision", f"{metrics.get('average_precision', 'N/A')}"),
        ("Total Samples", f"{metrics['total_samples']:,}"),
        ("", "", False),
        ("LIVE CLASS", "", True),
        ("  Precision", f"{metrics['live']['precision']:.4f}"),
        ("  Recall", f"{metrics['live']['recall']:.4f}"),
        ("  F1-Score", f"{metrics['live']['f1']:.4f}"),
        ("  Support", f"{metrics['live']['support']:,}"),
        ("", "", False),
        ("SPOOF CLASS", "", True),
        ("  Precision", f"{metrics['spoof']['precision']:.4f}"),
        ("  Recall", f"{metrics['spoof']['recall']:.4f}"),
        ("  F1-Score", f"{metrics['spoof']['f1']:.4f}"),
        ("  Support", f"{metrics['spoof']['support']:,}"),
    ]

    y_pos = 0.95
    for line in summary_lines:
        if len(line) == 3 and line[2] is True:
            # Header
            ax.text(0.05, y_pos, line[0], fontsize=13, fontweight='bold',
                    color=COLORS['test'], transform=ax.transAxes,
                    fontfamily='monospace')
        elif len(line) == 3 and line[2] is False:
            pass  # blank line
        else:
            key, val = line[0], line[1]
            ax.text(0.05, y_pos, key, fontsize=11, color=COLORS['text'],
                    transform=ax.transAxes, fontfamily='monospace')
            ax.text(0.60, y_pos, val, fontsize=11, fontweight='bold',
                    color='#FFD600', transform=ax.transAxes,
                    fontfamily='monospace')
        y_pos -= 0.05

    ax.set_title('Summary Statistics', fontsize=13, fontweight='bold',
                 color=COLORS['text'], pad=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / 'test_overview.png', dpi=150, facecolor=COLORS['bg'])
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Test Function
# ═══════════════════════════════════════════════════════════════════════════════

def main(args):
    print("=" * 70)
    print("  Swin Transformer Baseline Anti-Spoofing Model — Test Evaluation")
    print("=" * 70)
    print(f"  Checkpoint:     {args.checkpoint}")
    print(f"  Model name:     {args.model_name}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Mode:           {'MULTI-FRAME (LSTM temporal)' if args.multi_frame else 'SINGLE-FRAME (LSTM seq=1)'}")
    if False:
        print(f"  Seq length:     {args.seq_len} frames/sequence")
    print(f"  Dataset dir:    {DATASET_DIR / 'test'}")
    print(f"  Output dir:     {TEST_LOGS_DIR}")
    print("=" * 70)

    # ── Device ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] Using: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load Dataset ─────────────────────────────────────────────────────
    print("\n[Data] Loading test dataset...")
    num_workers = min(2, os.cpu_count() or 1) if os.name == 'nt' else min(4, os.cpu_count() or 1)
    use_pin_memory = device.type == "cuda"

    if False:
        # ── MULTI-FRAME MODE ─────────────────────────────────────────────
        # Dùng VideoSequenceDataset để nhóm frames thành temporal sequences
        # LSTM sẽ xử lý T=seq_len frames thay vì 1 frame
        print(f"  [Multi-Frame] Loading VideoSequenceDataset "
              f"(seq_len={args.seq_len}, NO temporal augmentation — eval mode)")
        test_dataset = VideoSequenceDataset(
            DATASET_DIR / "test",
            transform=get_eval_transforms(),   # Không augmentation khi test
            seq_len=args.seq_len,
            temporal_augmentor=None,           # Tắt hoàn toàn temporal aug
        )

        if len(test_dataset) == 0:
            print("\n[ERROR] No test sequences found!")
            print(f"   Expected data in: {DATASET_DIR / 'test'}/")
            print("   Lưu ý: Multi-frame mode cần video sequences (SiW, FF-C23).")
            print("   CelebA-Spoof sẽ được tạo pseudo-sequences (lặp ảnh).")
            sys.exit(1)

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            collate_fn=test_multiframe_collate_fn,
        )

        # Thống kê
        seq_labels = [label for _, label in test_dataset.sequences]
        n_live = seq_labels.count(LIVE_LABEL)
        n_spoof = seq_labels.count(SPOOF_LABEL)
        print(f"[Data] Test: {len(test_dataset)} sequences "
              f"(live={n_live}, spoof={n_spoof}, T={args.seq_len})")
        print(f"[Data] Tổng frames tương đương: {len(test_dataset) * args.seq_len:,}")

    else:
        # ── SINGLE-FRAME MODE (cũ) ────────────────────────────────────────
        # Mỗi ảnh là 1 sample riêng, LSTM chạy với seq_len=1
        print("  [Single-Frame] Loading AntiSpoofDataset (LSTM seq_len=1)")
        print("  ⚠ Cảnh báo: LSTM không khai thác temporal trong mode này.")
        print("    → Dùng --multi-frame để bật temporal evaluation đúng.")
        test_dataset = AntiSpoofDataset(DATASET_DIR / "test", transform=get_eval_transforms())

        if len(test_dataset) == 0:
            print("\n[ERROR] No test data found!")
            print(f"   Expected data in: {DATASET_DIR / 'test'}/")
            print("   Run preprocessing pipeline first:")
            print("     python -m preprocessing                 (CelebA Spoof)")
            print("     python -m preprocessing.pipeline_ffc23  (FF-C23)")
            print("     python -m preprocessing.pipeline_siw    (SiW)")
            sys.exit(1)

        # Count per class
        test_labels = [label for _, label, _ in test_dataset.samples]
        n_live = test_labels.count(LIVE_LABEL)
        n_spoof = test_labels.count(SPOOF_LABEL)
        print(f"[Data] Test: {len(test_dataset)} images "
              f"(live={n_live}, spoof={n_spoof})")

        # Count per source
        source_counts = {}
        for _, label, source in test_dataset.samples:
            if source not in source_counts:
                source_counts[source] = {"live": 0, "spoof": 0}
            source_counts[source]["live" if label == 0 else "spoof"] += 1
        for source, counts in source_counts.items():
            print(f"  [{source}] live={counts['live']}, spoof={counts['spoof']}, "
                  f"total={counts['live'] + counts['spoof']}")

        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=use_pin_memory,
            collate_fn=test_collate_fn,
        )

    # ── Load Model ───────────────────────────────────────────────────────
    print(f"\n[Model] Building SwinTransformerBaseline (model_name={args.model_name})...")
    model = SwinTransformerBaseline(
        num_classes=2,
        pretrained=False,
        model_name=args.model_name
    ).to(device)

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"\n[ERROR] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"[Model] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        ckpt_epoch = checkpoint.get("epoch", "?")
        ckpt_val_acc = checkpoint.get("val_accuracy", "?")
        ckpt_val_loss = checkpoint.get("val_loss", "?")
        optimal_threshold = checkpoint.get("optimal_threshold", 0.5)
        model_name_used = checkpoint.get("model_name", args.model_name)
        is_swa = checkpoint.get("is_swa", False)
        print(f"[Model] Loaded from epoch {ckpt_epoch} "
              f"(val_loss={ckpt_val_loss}, val_acc={ckpt_val_acc})")
        print(f"[Model] Optimal threshold: {optimal_threshold:.3f}")
        print(f"[Model] Is SWA model: {is_swa}")
    else:
        model.load_state_dict(checkpoint)
        optimal_threshold = 0.5
        print("[Model] Loaded raw state_dict (no metadata)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Total parameters: {total_params:,}")

    model.eval()

    # ── Run Test Evaluation ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    if False:
        print(f"  Running Multi-Frame Test Evaluation (LSTM T={args.seq_len})...")
        print(f"  → model.forward_multi_frame() được gọi — LSTM xử lý đầy đủ temporal")
    else:
        print("  Running Single-Frame Test Evaluation (LSTM seq_len=1)...")
        print("  → model.forward() được gọi — LSTM chỉ xử lý 1 timestep")
    print(f"{'='*70}\n")

    start_time = time.time()
    if False:
        all_probs, all_preds, all_labels, all_sources = evaluate_test_multiframe(
            model, test_loader, device, seq_len=args.seq_len, resume=args.resume
        )
    else:
        all_probs, all_preds, all_labels, all_sources = evaluate_test(
            model, test_loader, device, resume=args.resume
        )
    eval_time = time.time() - start_time

    # ── Compute Metrics ──────────────────────────────────────────────────
    print("\n[Metrics] Computing test metrics...")
    metrics = compute_test_metrics(all_preds, all_labels, all_probs)
    source_metrics = compute_per_source_metrics(all_preds, all_labels, all_sources)

    # Also compute metrics at optimal threshold
    preds_at_threshold = np.where(all_probs >= optimal_threshold, 0, 1)
    threshold_metrics = compute_test_metrics(preds_at_threshold, all_labels, all_probs)

    # ── Print Results ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  TEST RESULTS (argmax prediction)")
    print(f"{'='*70}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  ROC AUC:           {metrics.get('roc_auc', 'N/A')}")
    print(f"  Average Precision: {metrics.get('average_precision', 'N/A')}")
    print(f"  Total samples:     {metrics['total_samples']:,}")
    print()
    for cls_name in CLASS_NAMES:
        m = metrics[cls_name]
        print(f"  {cls_name:>5s}: P={m['precision']:.4f}  "
              f"R={m['recall']:.4f}  F1={m['f1']:.4f}  "
              f"(n={m['support']:,})")
    print(f"\n  Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"                Pred Live  Pred Spoof")
    print(f"  Actual Live   {cm[0][0]:>8d}  {cm[0][1]:>10d}")
    print(f"  Actual Spoof  {cm[1][0]:>8d}  {cm[1][1]:>10d}")

    print(f"\n{'-'*70}")
    print(f"  TEST RESULTS (at optimal threshold = {optimal_threshold:.3f})")
    print(f"{'-'*70}")
    print(f"  Accuracy:          {threshold_metrics['accuracy']:.4f}")
    for cls_name in CLASS_NAMES:
        m = threshold_metrics[cls_name]
        print(f"  {cls_name:>5s}: P={m['precision']:.4f}  "
              f"R={m['recall']:.4f}  F1={m['f1']:.4f}  "
              f"(n={m['support']:,})")

    print(f"\n{'-'*70}")
    print("  PER-SOURCE RESULTS")
    print(f"{'-'*70}")
    if not source_metrics:
        print("  [Info] Per-source metrics không khả dụng trong multi-frame mode.")
        print("         VideoSequenceDataset không track source riêng cho từng sequence.")
        print("         → Chạy single-frame mode để xem per-source breakdown.")
    else:
        for source, sm in source_metrics.items():
            print(f"\n  [{source}] Accuracy: {sm['accuracy']:.4f} "
                  f"(total={sm['total_samples']}, live={sm['live_count']}, spoof={sm['spoof_count']})")
            for cls_name in CLASS_NAMES:
                m = sm[cls_name]
                print(f"     {cls_name:>5s}: P={m['precision']:.4f}  "
                      f"R={m['recall']:.4f}  F1={m['f1']:.4f}  "
                      f"(n={m['support']:,})")

    # -- Generate Charts --------------------------------------------------
    print(f"\n[Charts] Generating test visualization charts...")
    TEST_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Chart 1: Confusion Matrix
    print("  [1/6] Confusion Matrix...")
    plot_test_confusion_matrix(metrics["confusion_matrix"], TEST_LOGS_DIR)

    # Chart 2: Classification Report (Bar chart)
    print("  [2/6] Classification Report...")
    plot_classification_report(metrics, TEST_LOGS_DIR)

    # Chart 3: ROC Curve
    print("  [3/6] ROC Curve...")
    roc_auc_val = plot_roc_curve(all_labels, all_probs, TEST_LOGS_DIR)

    # Chart 4: Precision-Recall Curve
    print("  [4/6] Precision-Recall Curve...")
    ap_val = plot_precision_recall_curve(all_labels, all_probs, TEST_LOGS_DIR)

    # Chart 5: Score Distribution
    print("  [5/6] Score Distribution...")
    plot_score_distribution(all_labels, all_probs, TEST_LOGS_DIR, threshold=optimal_threshold)

    # Chart 6: Per-source metrics
    print("  [6/6] Per-Source Metrics...")
    plot_per_source_metrics(source_metrics, TEST_LOGS_DIR)

    # Overview Dashboard
    print("  [+] Overview Dashboard...")
    plot_test_overview(metrics, all_labels, all_probs, all_preds,
                       metrics["confusion_matrix"], TEST_LOGS_DIR,
                       threshold=optimal_threshold)

    # ── Save Results JSON ────────────────────────────────────────────────
    results = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(checkpoint_path),
        "model_name": args.model_name,
        "eval_mode": "multi_frame" if False else "single_frame",
        "seq_len": args.seq_len if args.multi_frame else 1,
        "lstm_temporal_active": args.multi_frame,
        "optimal_threshold": optimal_threshold,
        "eval_time_seconds": round(eval_time, 1),
        "argmax_metrics": metrics,
        "threshold_metrics": threshold_metrics,
        "per_source_metrics": source_metrics,
    }

    results_path = TEST_LOGS_DIR / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [Results] Saved to {results_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Test Evaluation Complete!")
    print(f"{'='*70}")
    print(f"  Eval time:       {eval_time:.1f}s")
    print(f"  Test Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  ROC AUC:         {metrics.get('roc_auc', 'N/A')}")
    print(f"  Avg Precision:   {metrics.get('average_precision', 'N/A')}")
    print(f"  Charts:          {TEST_LOGS_DIR}/")
    print(f"  Results JSON:    {results_path}")

    # List generated chart files
    print("\n  [Charts] Generated charts:")
    for chart_file in sorted(TEST_LOGS_DIR.glob("*.png")):
        print(f"     - {chart_file.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test/Evaluate Swin Transformer Baseline Anti-Spoofing Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--checkpoint", type=str,
                        default=str(WEIGHTS_DIR / "antispoof_swin_transformer.pth"),
                        help="Path to model checkpoint")
    parser.add_argument("--model-name", type=str, default="swin_v2_t",
                        choices=["swin_v2_t", "swin_v2_s", "swin_v2_b"],
                        help="Swin Transformer architecture (default: swin_v2_t)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation (default: 32)")
    parser.add_argument("--resume", action="store_true",
                        help="Bật chế độ tiếp tục (resume) nếu lần test trước bị ngắt giữa chừng")

    args = parser.parse_args()
    main(args)
