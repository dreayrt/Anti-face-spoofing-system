"""
train.py – Training script for CNN + DSP + LSTM Anti-Spoofing Model (v2).

Trains the hybrid model on preprocessed data from CelebA Spoof + FF-C23.
Includes: FocalLoss, SWA, WeightedRandomSampler, Gradient Clipping,
          SE Attention, Optimal Threshold Tuning.

Usage:
    cd ai-service
    python train.py                           # Train with defaults (EfficientNet-B0, 50 epochs)
    python train.py --epochs 50 --lr 0.0001   # Custom hyperparams
    python train.py --backbone resnet50        # Use ResNet-50 backbone
    python train.py --resume                   # Resume from last checkpoint
    python train.py --spoof-weight 3.0         # Asymmetric spoof penalty

Output:
    models/weights/antispoof_cnn_dsp_lstm.pth  — Best model checkpoint
    models/weights/training_log.json           — Training metrics history
    training_logs/                             — Training charts (loss, accuracy, precision, ...)
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
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI required)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Add inference dir to path so we can import the model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "inference"))
from antispoof_model import CNNDSPLSTMAntiSpoof


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
WEIGHTS_DIR = Path(__file__).resolve().parent / "models" / "weights"
LOGS_DIR = Path(__file__).resolve().parent / "training_logs"

# ImageNet normalization (same as preprocessing pipeline)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

# Class mapping
CLASS_NAMES = ["live", "spoof"]
LIVE_LABEL = 0
SPOOF_LABEL = 1


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class AntiSpoofDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for anti-spoofing training.

    Loads images from the preprocessed dataset directory structure:
        dataset/{split}/{source}/{live,spoof}/

    Args:
        root_dir: Path to the split directory (e.g., dataset/train/).
        transform: Torchvision transforms to apply.
        sources: List of data sources to include (e.g., ['celeba-spoof', 'ff-c23']).
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root_dir: Path, transform=None, sources=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []  # List of (image_path, label)

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
                    self.samples.append((img_path, LIVE_LABEL))

            # Scan spoof/ directory (may have subdirs for FF-C23 methods)
            spoof_dir = source_dir / "spoof"
            if spoof_dir.exists():
                for img_path in self._scan_images(spoof_dir):
                    self.samples.append((img_path, SPOOF_LABEL))

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
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Return a black image if file is corrupt
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self):
        """Compute inverse-frequency class weights for loss balancing."""
        labels = [label for _, label in self.samples]
        counts = np.bincount(labels, minlength=len(CLASS_NAMES))
        total = len(labels)
        weights = total / (len(CLASS_NAMES) * counts.astype(np.float64) + 1e-8)
        return torch.tensor(weights, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Transforms
# ═══════════════════════════════════════════════════════════════════════════════

class Cutout:
    """Randomly mask out rectangular regions from input tensor.

    Forces the model to learn from multiple regions rather than
    relying on a single discriminative area for classification.

    Args:
        n_holes: Number of rectangular regions to cut out.
        length: Side length (pixels) of each square cutout region.
    """

    def __init__(self, n_holes: int = 1, length: int = 32):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = int(np.clip(y - self.length // 2, 0, h))
            y2 = int(np.clip(y + self.length // 2, 0, h))
            x1 = int(np.clip(x - self.length // 2, 0, w))
            x2 = int(np.clip(x + self.length // 2, 0, w))
            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        return img * mask


def get_train_transforms():
    """Training augmentation pipeline (enhanced with Cutout + RandomErasing + RandomAffine)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        Cutout(n_holes=1, length=32),
    ])


def get_eval_transforms():
    """Evaluation transforms (deterministic — no augmentation)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# MixUp Augmentation
# ═══════════════════════════════════════════════════════════════════════════════

def mixup_data(x, y, alpha=0.2):
    """Apply MixUp augmentation to a batch.

    Blends pairs of images and their labels to create soft decision
    boundaries, reducing overconfidence and improving generalization.

    Args:
        x: Input images tensor (B, C, H, W).
        y: Labels tensor (B,).
        alpha: Beta distribution parameter. Higher = more mixing.

    Returns:
        mixed_x: Blended images.
        y_a, y_b: Original label pairs.
        lam: Mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute MixUp loss as weighted sum of losses for both label sets.

    Args:
        criterion: Loss function.
        pred: Model predictions.
        y_a, y_b: Label pairs from mixup_data.
        lam: Mixing coefficient.

    Returns:
        Weighted loss value.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ═══════════════════════════════════════════════════════════════════════════════
# Focal Loss
# ═══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance and hard sample mining.

    From the RetinaNet paper (Lin et al., 2017). Focal Loss down-weights
    well-classified (easy) examples and focuses the model on hard,
    misclassified examples — critical for reducing spoof→live false positives.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Per-class weight tensor [alpha_live, alpha_spoof].
               Higher alpha for spoof = heavier penalty for missing spoof.
        gamma: Focusing parameter. gamma=0 is standard CE, gamma=2 is typical.
               Higher gamma = more focus on hard examples.
        label_smoothing: Label smoothing factor (0.0 = no smoothing).
    """

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha
        else:
            self.alpha = None

    def forward(self, logits, targets):
        """Compute Focal Loss.

        Args:
            logits: Raw model outputs (B, C).
            targets: Ground truth labels (B,).

        Returns:
            Scalar focal loss value.
        """
        num_classes = logits.size(1)

        # Apply label smoothing
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(logits)
                smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth_targets = F.one_hot(targets, num_classes).float()

        # Softmax probabilities
        probs = F.softmax(logits, dim=1)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - probs) ** self.gamma

        # Log probabilities
        log_probs = F.log_softmax(logits, dim=1)

        # Focal loss per sample
        loss = -focal_weight * smooth_targets * log_probs  # (B, C)

        # Apply per-class alpha weights
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_weight = alpha.unsqueeze(0)  # (1, C)
            loss = loss * alpha_weight

        # Sum over classes, mean over batch
        loss = loss.sum(dim=1).mean()

        return loss


# ═══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs,
                    use_mixup=True, mixup_alpha=0.2, max_grad_norm=1.0):
    """Train for one epoch and return average loss + accuracy.

    Args:
        use_mixup: Whether to apply MixUp augmentation.
        mixup_alpha: Beta distribution parameter for MixUp.
        max_grad_norm: Maximum gradient norm for clipping (0 = no clipping).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Apply MixUp augmentation
        if use_mixup:
            images, targets_a, targets_b, lam = mixup_data(images, labels, mixup_alpha)

        # Forward
        logits = model(images)

        # Compute loss (MixUp-aware or standard)
        if use_mixup:
            loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
        else:
            loss = criterion(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for training stability
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        # Metrics (use original labels for accuracy tracking)
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(logits, 1)
        if use_mixup:
            correct += (lam * (predicted == targets_a).sum().item()
                        + (1 - lam) * (predicted == targets_b).sum().item())
        else:
            correct += (predicted == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=loss.item(), acc=correct / total)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, total_epochs):
    """Validate and return average loss + accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Val]  ", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def compute_metrics(preds, labels):
    """Compute precision, recall, F1 for each class."""
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1], zero_division=0
    )

    cm = confusion_matrix(labels, preds, labels=[0, 1])

    metrics = {}
    for i, cls_name in enumerate(CLASS_NAMES):
        metrics[cls_name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def find_optimal_threshold(model, dataloader, device, target_spoof_recall=0.95):
    """Find the optimal classification threshold on the validation set.

    Searches for the threshold that achieves the target spoof recall
    (minimizing spoof→live false positives) while maintaining reasonable
    live accuracy.

    Args:
        model: Trained model in eval mode.
        dataloader: Validation DataLoader.
        device: Torch device.
        target_spoof_recall: Minimum desired spoof recall (default: 0.95).

    Returns:
        optimal_threshold: Best threshold for the decision boundary.
        threshold_metrics: Dict with metrics at the optimal threshold.
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="[Threshold] Scanning", leave=False):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            live_probs = probs[:, 0].cpu().numpy()  # P(live)
            all_probs.extend(live_probs)
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Search thresholds from 0.1 to 0.9
    best_threshold = 0.5
    best_f1 = 0.0
    best_metrics = {}

    for threshold in np.arange(0.1, 0.91, 0.01):
        preds = (all_probs >= threshold).astype(int)  # 1=live (predicted)
        # Convert: live=0, spoof=1 in labels
        # Predicted live=0 when probs >= threshold
        predicted_labels = np.where(preds == 1, 0, 1)  # P(live)>=th → pred=live(0)

        # Spoof recall: of actual spoof (label=1), how many predicted as spoof?
        actual_spoof_mask = all_labels == 1
        if actual_spoof_mask.sum() > 0:
            spoof_recall = (predicted_labels[actual_spoof_mask] == 1).sum() / actual_spoof_mask.sum()
        else:
            spoof_recall = 0.0

        # Live recall: of actual live (label=0), how many predicted as live?
        actual_live_mask = all_labels == 0
        if actual_live_mask.sum() > 0:
            live_recall = (predicted_labels[actual_live_mask] == 0).sum() / actual_live_mask.sum()
        else:
            live_recall = 0.0

        # Spoof precision: of predicted spoof, how many are actually spoof?
        pred_spoof_mask = predicted_labels == 1
        if pred_spoof_mask.sum() > 0:
            spoof_precision = (all_labels[pred_spoof_mask] == 1).sum() / pred_spoof_mask.sum()
        else:
            spoof_precision = 0.0

        # F1 for spoof class
        if spoof_precision + spoof_recall > 0:
            spoof_f1 = 2 * spoof_precision * spoof_recall / (spoof_precision + spoof_recall)
        else:
            spoof_f1 = 0.0

        # We want: spoof_recall >= target AND maximum overall F1
        accuracy = (predicted_labels == all_labels).mean()
        overall_f1 = spoof_f1  # Prioritize spoof detection

        if spoof_recall >= target_spoof_recall and overall_f1 > best_f1:
            best_f1 = overall_f1
            best_threshold = threshold
            best_metrics = {
                "threshold": round(float(threshold), 3),
                "accuracy": round(float(accuracy), 4),
                "spoof_recall": round(float(spoof_recall), 4),
                "spoof_precision": round(float(spoof_precision), 4),
                "spoof_f1": round(float(spoof_f1), 4),
                "live_recall": round(float(live_recall), 4),
                "spoof_fp": int((predicted_labels[actual_spoof_mask] == 0).sum()),
            }

    # If no threshold meets target, find the one with highest spoof recall
    if not best_metrics:
        print(f"  [Threshold] Warning: No threshold achieves spoof_recall>={target_spoof_recall:.2f}")
        print(f"  [Threshold] Finding threshold with highest spoof recall instead...")
        best_spoof_recall = 0.0
        for threshold in np.arange(0.1, 0.91, 0.01):
            preds = (all_probs >= threshold).astype(int)
            predicted_labels = np.where(preds == 1, 0, 1)
            actual_spoof_mask = all_labels == 1
            if actual_spoof_mask.sum() > 0:
                spoof_recall = (predicted_labels[actual_spoof_mask] == 1).sum() / actual_spoof_mask.sum()
            else:
                spoof_recall = 0.0
            if spoof_recall > best_spoof_recall:
                best_spoof_recall = spoof_recall
                best_threshold = threshold
                accuracy = (predicted_labels == all_labels).mean()
                best_metrics = {
                    "threshold": round(float(threshold), 3),
                    "accuracy": round(float(accuracy), 4),
                    "spoof_recall": round(float(spoof_recall), 4),
                    "live_recall": round(float((predicted_labels[all_labels == 0] == 0).sum() / (all_labels == 0).sum()), 4),
                    "spoof_fp": int((predicted_labels[actual_spoof_mask] == 0).sum()),
                }

    print(f"  [Threshold] Optimal threshold: {best_threshold:.3f}")
    print(f"  [Threshold] Metrics at optimal threshold: {best_metrics}")

    return float(best_threshold), best_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Training Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_charts(training_log: list, output_dir: Path):
    """Generate and save all training visualization charts.

    Creates 4 charts:
      1. Loss curves (train vs val)
      2. Accuracy curves (train vs val)
      3. Precision / Recall / F1 curves (per class)
      4. Learning rate schedule

    Args:
        training_log: List of epoch log dicts.
        output_dir: Directory to save chart images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = [log["epoch"] for log in training_log]
    train_loss = [log["train_loss"] for log in training_log]
    val_loss = [log["val_loss"] for log in training_log]
    train_acc = [log["train_acc"] for log in training_log]
    val_acc = [log["val_acc"] for log in training_log]
    lr_values = [log["lr"] for log in training_log]

    # Extract per-class precision/recall/F1 if available
    has_metrics = all("val_precision_live" in log for log in training_log)

    # ── Style setup ──────────────────────────────────────────────────────
    plt.style.use('default')
    COLORS = {
        'train': '#2196F3',      # Blue
        'val': '#FF5722',        # Red-Orange
        'live_p': '#4CAF50',     # Green
        'live_r': '#8BC34A',     # Light Green
        'live_f1': '#009688',    # Teal
        'spoof_p': '#E91E63',    # Pink
        'spoof_r': '#FF9800',    # Orange
        'spoof_f1': '#9C27B0',   # Purple
        'lr': '#607D8B',         # Blue Grey
        'bg': '#1a1a2e',         # Dark background
        'grid': '#333355',       # Grid lines
        'text': '#e0e0e0',       # Text color
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
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=9, framealpha=0.7, facecolor='#2a2a4a', edgecolor=COLORS['grid'],
                  labelcolor=COLORS['text'])

    # ── Chart 1: Loss Curves ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.plot(epochs, train_loss, color=COLORS['train'], linewidth=2,
            marker='o', markersize=4, label='Train Loss')
    ax.plot(epochs, val_loss, color=COLORS['val'], linewidth=2,
            marker='s', markersize=4, label='Val Loss')
    # Mark best (lowest val loss)
    best_idx = np.argmin(val_loss)
    ax.annotate(f'Best: {val_loss[best_idx]:.4f}',
                xy=(epochs[best_idx], val_loss[best_idx]),
                xytext=(15, 15), textcoords='offset points',
                fontsize=9, color=COLORS['val'],
                arrowprops=dict(arrowstyle='->', color=COLORS['val'], lw=1.5))
    style_ax(ax, 'Training & Validation Loss', 'Epoch', 'Loss')
    fig.tight_layout()
    fig.savefig(output_dir / 'loss_curves.png', dpi=150, facecolor=COLORS['bg'])
    plt.close(fig)

    # ── Chart 2: Accuracy Curves ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.plot(epochs, train_acc, color=COLORS['train'], linewidth=2,
            marker='o', markersize=4, label='Train Accuracy')
    ax.plot(epochs, val_acc, color=COLORS['val'], linewidth=2,
            marker='s', markersize=4, label='Val Accuracy')
    # Mark best accuracy
    best_idx = np.argmax(val_acc)
    ax.annotate(f'Best: {val_acc[best_idx]:.4f}',
                xy=(epochs[best_idx], val_acc[best_idx]),
                xytext=(15, -20), textcoords='offset points',
                fontsize=9, color=COLORS['val'],
                arrowprops=dict(arrowstyle='->', color=COLORS['val'], lw=1.5))
    ax.set_ylim(0, 1.05)
    style_ax(ax, 'Training & Validation Accuracy', 'Epoch', 'Accuracy')
    fig.tight_layout()
    fig.savefig(output_dir / 'accuracy_curves.png', dpi=150, facecolor=COLORS['bg'])
    plt.close(fig)

    # ── Chart 3: Precision / Recall / F1 ─────────────────────────────────
    if has_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor(COLORS['bg'])
        fig.suptitle('Precision / Recall / F1-Score per Class', fontsize=15,
                     fontweight='bold', color=COLORS['text'], y=1.02)

        # Live class
        ax = axes[0]
        ax.plot(epochs, [l["val_precision_live"] for l in training_log],
                color=COLORS['live_p'], linewidth=2, marker='o', markersize=3, label='Precision')
        ax.plot(epochs, [l["val_recall_live"] for l in training_log],
                color=COLORS['live_r'], linewidth=2, marker='s', markersize=3, label='Recall')
        ax.plot(epochs, [l["val_f1_live"] for l in training_log],
                color=COLORS['live_f1'], linewidth=2, marker='^', markersize=3, label='F1-Score')
        ax.set_ylim(0, 1.05)
        style_ax(ax, 'Class: LIVE', 'Epoch', 'Score')

        # Spoof class
        ax = axes[1]
        ax.plot(epochs, [l["val_precision_spoof"] for l in training_log],
                color=COLORS['spoof_p'], linewidth=2, marker='o', markersize=3, label='Precision')
        ax.plot(epochs, [l["val_recall_spoof"] for l in training_log],
                color=COLORS['spoof_r'], linewidth=2, marker='s', markersize=3, label='Recall')
        ax.plot(epochs, [l["val_f1_spoof"] for l in training_log],
                color=COLORS['spoof_f1'], linewidth=2, marker='^', markersize=3, label='F1-Score')
        ax.set_ylim(0, 1.05)
        style_ax(ax, 'Class: SPOOF', 'Epoch', 'Score')

        fig.tight_layout()
        fig.savefig(output_dir / 'precision_recall_f1.png', dpi=150, facecolor=COLORS['bg'])
        plt.close(fig)

    # ── Chart 4: Learning Rate Schedule ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.plot(epochs, lr_values, color=COLORS['lr'], linewidth=2, marker='o', markersize=3,
            label='Learning Rate')
    ax.set_yscale('log')
    style_ax(ax, 'Learning Rate Schedule (CosineAnnealing)', 'Epoch', 'Learning Rate')
    fig.tight_layout()
    fig.savefig(output_dir / 'learning_rate.png', dpi=150, facecolor=COLORS['bg'])
    plt.close(fig)

    # ── Chart 5: Combined Overview (2x2) ─────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(COLORS['bg'])
    fig.suptitle('CNN + DSP + LSTM Anti-Spoofing — Training Overview',
                 fontsize=16, fontweight='bold', color=COLORS['text'], y=0.98)

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, color=COLORS['train'], linewidth=2, label='Train')
    ax.plot(epochs, val_loss, color=COLORS['val'], linewidth=2, label='Val')
    style_ax(ax, 'Loss', 'Epoch', 'Loss')

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, train_acc, color=COLORS['train'], linewidth=2, label='Train')
    ax.plot(epochs, val_acc, color=COLORS['val'], linewidth=2, label='Val')
    ax.set_ylim(0, 1.05)
    style_ax(ax, 'Accuracy', 'Epoch', 'Accuracy')

    # Precision (if available)
    ax = axes[1, 0]
    if has_metrics:
        ax.plot(epochs, [l["val_precision_live"] for l in training_log],
                color=COLORS['live_p'], linewidth=2, label='Live')
        ax.plot(epochs, [l["val_precision_spoof"] for l in training_log],
                color=COLORS['spoof_p'], linewidth=2, label='Spoof')
        ax.set_ylim(0, 1.05)
        style_ax(ax, 'Precision', 'Epoch', 'Precision')
    else:
        ax.text(0.5, 0.5, 'Metrics not yet available', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color=COLORS['text'])
        ax.set_facecolor(COLORS['bg'])

    # LR
    ax = axes[1, 1]
    ax.plot(epochs, lr_values, color=COLORS['lr'], linewidth=2, label='LR')
    ax.set_yscale('log')
    style_ax(ax, 'Learning Rate', 'Epoch', 'LR')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / 'training_overview.png', dpi=150, facecolor=COLORS['bg'])
    plt.close(fig)

    print(f"  [Charts] Charts saved to {output_dir}/")


def plot_confusion_matrix(cm: list, output_dir: Path, epoch: int = None):
    """Plot and save a confusion matrix heatmap.

    Args:
        cm: 2x2 confusion matrix as list of lists.
        output_dir: Directory to save the chart.
        epoch: Optional epoch number for the title.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cm_array = np.array(cm)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    # Heatmap
    im = ax.imshow(cm_array, interpolation='nearest', cmap='YlOrRd')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(colors='#e0e0e0', labelsize=9)
    cbar.outline.set_edgecolor('#333355')

    # Labels
    classes = CLASS_NAMES
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=11, color='#e0e0e0')
    ax.set_yticklabels(classes, fontsize=11, color='#e0e0e0')
    ax.set_xlabel('Predicted', fontsize=12, color='#e0e0e0')
    ax.set_ylabel('Actual', fontsize=12, color='#e0e0e0')

    title = 'Confusion Matrix'
    if epoch is not None:
        title += f' (Epoch {epoch})'
    ax.set_title(title, fontsize=14, fontweight='bold', color='#e0e0e0', pad=12)

    # Cell values
    for i in range(2):
        for j in range(2):
            val = cm_array[i, j]
            color = 'white' if val > cm_array.max() / 2 else 'black'
            ax.text(j, i, f'{val}', ha='center', va='center',
                    fontsize=18, fontweight='bold', color=color)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333355')
    ax.spines['bottom'].set_color('#333355')

    fig.tight_layout()
    fig.savefig(output_dir / 'confusion_matrix.png', dpi=150, facecolor='#1a1a2e')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Training Function
# ═══════════════════════════════════════════════════════════════════════════════

def main(args):
    print("=" * 70)
    print("  CNN + DSP + LSTM Anti-Spoofing Model — Training (v2)")
    print("=" * 70)
    print(f"  Backbone:       {args.backbone}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Learning rate:  {args.lr}")
    print(f"  Weight decay:   {args.weight_decay}")
    print(f"  Dropout:        {args.dropout}")
    print(f"  Loss function:  FocalLoss (gamma={args.gamma})")
    print(f"  Label smoothing:{args.label_smoothing}")
    print(f"  Spoof weight:   {args.spoof_weight}")
    print(f"  MixUp alpha:    {args.mixup_alpha}")
    print(f"  Grad clip norm: {args.grad_clip}")
    print(f"  SWA start:      epoch {args.swa_start}")
    print(f"  Early stop:     val_loss (patience={args.patience})")
    print(f"  Dataset dir:    {DATASET_DIR}")
    print(f"  Output dir:     {WEIGHTS_DIR}")
    print("=" * 70)

    # ── Device ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] Using: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Datasets ─────────────────────────────────────────────────────────
    print("\n[Data] Loading datasets...")
    train_dataset = AntiSpoofDataset(DATASET_DIR / "train", transform=get_train_transforms())
    val_dataset = AntiSpoofDataset(DATASET_DIR / "val", transform=get_eval_transforms())

    if len(train_dataset) == 0:
        print("\n[ERROR] No training data found!")
        print(f"   Expected data in: {DATASET_DIR / 'train'}/")
        print("   Run preprocessing pipeline first:")
        print("     python -m preprocessing           (CelebA Spoof)")
        print("     python -m preprocessing.pipeline_ffc23  (FF-C23)")
        sys.exit(1)

    # Class weights for imbalanced data (asymmetric: penalize spoof→live more)
    raw_class_weights = train_dataset.get_class_weights()
    # Apply asymmetric spoof weight multiplier
    asymmetric_weights = torch.tensor([
        raw_class_weights[0],                          # live weight (as-is)
        raw_class_weights[1] * args.spoof_weight,      # spoof weight * multiplier
    ], dtype=torch.float32).to(device)
    print(f"\n[Data] Raw class weights: live={raw_class_weights[0]:.3f}, spoof={raw_class_weights[1]:.3f}")
    print(f"[Data] Asymmetric weights (spoof×{args.spoof_weight}): "
          f"live={asymmetric_weights[0]:.3f}, spoof={asymmetric_weights[1]:.3f}")

    # Count per class
    train_labels = [label for _, label in train_dataset.samples]
    print(f"[Data] Train: {len(train_dataset)} images "
          f"(live={train_labels.count(0)}, spoof={train_labels.count(1)})")
    print(f"[Data] Val:   {len(val_dataset)} images")

    # ── WeightedRandomSampler for balanced batches ───────────────────────
    # Oversample live class so each batch has ~50:50 live:spoof ratio
    sample_weights = []
    n_live = train_labels.count(0)
    n_spoof = train_labels.count(1)
    weight_live = 1.0 / n_live
    weight_spoof = 1.0 / n_spoof
    for _, label in train_dataset.samples:
        if label == LIVE_LABEL:
            sample_weights.append(weight_live)
        else:
            sample_weights.append(weight_spoof)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )
    print(f"[Data] WeightedRandomSampler: live_weight={weight_live:.6f}, "
          f"spoof_weight={weight_spoof:.6f}")

    # DataLoaders
    num_workers = min(4, os.cpu_count() or 1)
    use_pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=use_pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory,
    )

    # ── Model ────────────────────────────────────────────────────────────
    print(f"\n[Model] Building CNN+DSP+LSTM (backbone={args.backbone})...")
    model = CNNDSPLSTMAntiSpoof(
        num_classes=2,
        backbone=args.backbone,
        lstm_hidden=256,
        lstm_layers=2,
        dsp_output_dim=256,
        dropout=args.dropout,
        pretrained=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total parameters:     {total_params:,}")
    print(f"[Model] Trainable parameters: {trainable_params:,}")

    # ── Loss, Optimizer, Scheduler ───────────────────────────────────────
    criterion = FocalLoss(
        alpha=asymmetric_weights.cpu().tolist(),
        gamma=args.gamma,
        label_smoothing=args.label_smoothing,
    )
    print(f"\n[Loss] FocalLoss(alpha={asymmetric_weights.cpu().tolist()}, "
          f"gamma={args.gamma}, label_smoothing={args.label_smoothing})")

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── SWA (Stochastic Weight Averaging) ───────────────────────────────
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-5, anneal_epochs=5)
    swa_start = args.swa_start
    print(f"[SWA] Will activate at epoch {swa_start} with SWA LR=1e-5")

    # ── Resume from checkpoint ───────────────────────────────────────────
    start_epoch = 1
    best_val_loss = float('inf')
    training_log = []

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = WEIGHTS_DIR / "antispoof_cnn_dsp_lstm.pth"
    last_checkpoint_path = WEIGHTS_DIR / "antispoof_last.pth"

    if args.resume and last_checkpoint_path.exists():
        print(f"\n[Resume] Loading checkpoint from {last_checkpoint_path}")
        ckpt = torch.load(last_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float('inf'))
        training_log = ckpt.get("training_log", [])
        print(f"[Resume] Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # ── Training Loop ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  Starting Training...")
    print(f"{'='*70}\n")

    patience = args.patience
    patience_counter = 0
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs,
            use_mixup=(args.mixup_alpha > 0), mixup_alpha=args.mixup_alpha,
            max_grad_norm=args.grad_clip,
        )

        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device, epoch, args.epochs
        )

        # Step scheduler (use SWA scheduler after swa_start epoch)
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        else:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

        epoch_time = time.time() - epoch_start

        # Compute per-class metrics every epoch for charting
        epoch_metrics = compute_metrics(val_preds, val_labels)

        # Log
        epoch_log = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_precision_live": round(epoch_metrics["live"]["precision"], 4),
            "val_recall_live": round(epoch_metrics["live"]["recall"], 4),
            "val_f1_live": round(epoch_metrics["live"]["f1"], 4),
            "val_precision_spoof": round(epoch_metrics["spoof"]["precision"], 4),
            "val_recall_spoof": round(epoch_metrics["spoof"]["recall"], 4),
            "val_f1_spoof": round(epoch_metrics["spoof"]["f1"], 4),
            "lr": round(current_lr, 8),
            "time_sec": round(epoch_time, 1),
        }
        training_log.append(epoch_log)

        # Print epoch summary
        print(f"Epoch {epoch:3d}/{args.epochs} │ "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} │ "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} │ "
              f"LR: {current_lr:.2e} │ {epoch_time:.1f}s")

        # Save best model (based on val_loss for better generalization)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "metrics": epoch_metrics,
                "backbone": args.backbone,
                "training_log": training_log,
            }, checkpoint_path)

            print(f"  [SUCCESS] Best model saved! (val_loss={val_loss:.4f}, val_acc={val_acc:.4f})")

            # Print per-class metrics
            for cls_name in CLASS_NAMES:
                m = epoch_metrics[cls_name]
                print(f"     {cls_name:>5s}: P={m['precision']:.3f} "
                      f"R={m['recall']:.3f} F1={m['f1']:.3f} "
                      f"(n={m['support']})")

            # Save confusion matrix for best model
            plot_confusion_matrix(epoch_metrics["confusion_matrix"], LOGS_DIR, epoch)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[STOP] Early stopping at epoch {epoch} "
                      f"(no improvement in val_loss for {patience} epochs)")
                break

        # Save last checkpoint (for resume)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "training_log": training_log,
        }, last_checkpoint_path)

        # Generate charts after every epoch (live monitoring)
        if len(training_log) >= 2:
            plot_training_charts(training_log, LOGS_DIR)

    # ── SWA: Update Batch Norm statistics ─────────────────────────────────
    if swa_start <= args.epochs:
        print("\n[SWA] Updating BatchNorm statistics for SWA model...")
        # SWA model needs BN statistics recomputed on training data
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

        # Validate SWA model
        print("[SWA] Validating SWA-averaged model...")
        swa_val_loss, swa_val_acc, swa_preds, swa_labels = validate(
            swa_model, val_loader, criterion, device, args.epochs, args.epochs
        )
        swa_metrics = compute_metrics(swa_preds, swa_labels)
        print(f"[SWA] SWA Val Loss: {swa_val_loss:.4f}, Val Acc: {swa_val_acc:.4f}")
        for cls_name in CLASS_NAMES:
            m = swa_metrics[cls_name]
            print(f"  [SWA] {cls_name:>5s}: P={m['precision']:.3f} "
                  f"R={m['recall']:.3f} F1={m['f1']:.3f}")

        # Save SWA model if it's better
        if swa_val_loss < best_val_loss:
            print(f"[SWA] SWA model is BETTER! (loss: {swa_val_loss:.4f} < {best_val_loss:.4f})")
            best_val_loss = swa_val_loss

            # Find optimal threshold on SWA model
            print("\n[Threshold] Finding optimal threshold on SWA model...")
            optimal_threshold, threshold_metrics = find_optimal_threshold(
                swa_model, val_loader, device, target_spoof_recall=0.95
            )

            torch.save({
                "epoch": args.epochs,
                "model_state_dict": swa_model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_accuracy": swa_val_acc,
                "val_loss": swa_val_loss,
                "best_val_loss": best_val_loss,
                "metrics": swa_metrics,
                "backbone": args.backbone,
                "training_log": training_log,
                "is_swa": True,
                "optimal_threshold": optimal_threshold,
                "threshold_metrics": threshold_metrics,
            }, checkpoint_path)
            print(f"  [SUCCESS] SWA model saved with optimal threshold={optimal_threshold:.3f}")

            # Save confusion matrix for SWA model
            plot_confusion_matrix(swa_metrics["confusion_matrix"], LOGS_DIR, epoch=f"SWA")
        else:
            print(f"[SWA] SWA model NOT better (loss: {swa_val_loss:.4f} >= {best_val_loss:.4f})")
            # Still find optimal threshold on best model
            print("\n[Threshold] Finding optimal threshold on best model...")
            # Reload best model
            best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(best_ckpt["model_state_dict"])
            optimal_threshold, threshold_metrics = find_optimal_threshold(
                model, val_loader, device, target_spoof_recall=0.95
            )
            # Update checkpoint with threshold
            best_ckpt["optimal_threshold"] = optimal_threshold
            best_ckpt["threshold_metrics"] = threshold_metrics
            torch.save(best_ckpt, checkpoint_path)
            print(f"  [SUCCESS] Best model updated with optimal threshold={optimal_threshold:.3f}")
    else:
        # No SWA, just find optimal threshold on best model
        print("\n[Threshold] Finding optimal threshold on best model...")
        best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        optimal_threshold, threshold_metrics = find_optimal_threshold(
            model, val_loader, device, target_spoof_recall=0.95
        )
        best_ckpt["optimal_threshold"] = optimal_threshold
        best_ckpt["threshold_metrics"] = threshold_metrics
        torch.save(best_ckpt, checkpoint_path)
        print(f"  [SUCCESS] Best model updated with optimal threshold={optimal_threshold:.3f}")

    # ── Generate Final Charts ────────────────────────────────────────────
    print("\n[Charts] Generating training visualization charts...")
    plot_training_charts(training_log, LOGS_DIR)

    # ── Training Complete ────────────────────────────────────────────────
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  Training Complete!")
    print(f"{'='*70}")
    print(f"  Total time:      {total_time/60:.1f} minutes")
    print(f"  Best val loss:   {best_val_loss:.4f}")
    print(f"  Optimal thresh:  {optimal_threshold:.3f}")
    print(f"  Best checkpoint: {checkpoint_path}")
    print(f"  Charts:          {LOGS_DIR}/")

    # Save training log
    log_path = WEIGHTS_DIR / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"  Training log:    {log_path}")

    # List generated chart files
    print("\n  [Charts] Generated charts:")
    for chart_file in sorted(LOGS_DIR.glob("*.png")):
        print(f"     - {chart_file.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CNN+DSP+LSTM Anti-Spoofing Model (v2 — FocalLoss + SWA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--backbone", type=str, default="efficientnet_b0",
                        choices=["mobilenet_v2", "resnet50", "efficientnet_b0"],
                        help="CNN backbone architecture (default: efficientnet_b0)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate (default: 0.0001)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="AdamW weight decay (default: 0.0001)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience based on val_loss (default: 15)")
    parser.add_argument("--dropout", type=float, default=0.6,
                        help="Dropout rate (default: 0.6)")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing factor (default: 0.1)")
    parser.add_argument("--mixup-alpha", type=float, default=0.2,
                        help="MixUp alpha parameter, 0 to disable (default: 0.2)")
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="Focal Loss gamma (focusing parameter, default: 2.0)")
    parser.add_argument("--spoof-weight", type=float, default=3.0,
                        help="Asymmetric weight multiplier for spoof class (default: 3.0)")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping max norm, 0 to disable (default: 1.0)")
    parser.add_argument("--swa-start", type=int, default=35,
                        help="Epoch to start SWA (default: 35)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")

    args = parser.parse_args()
    main(args)
