"""Quick integration test for the CNN+DSP+LSTM anti-spoofing model."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "inference"))

# Test 1: Imports
print("[Test 1] Importing modules...")
from dsp_utils import DSPModule
from antispoof_model import CNNDSPLSTMAntiSpoof, AntiSpoofPredictor
from mock_model import MockAntiSpoofModel
print("  OK")

# Test 2: Fallback mechanism
print("[Test 2] Testing fallback mechanism...")
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "models", "weights", "antispoof_cnn_dsp_lstm.pth"
)

if os.path.isfile(CHECKPOINT_PATH):
    print("  Checkpoint exists -> would load real model")
else:
    print(f"  Checkpoint not found: {CHECKPOINT_PATH}")
    print("  Fallback to MockAntiSpoofModel -> OK")
    mock = MockAntiSpoofModel()
    print(f"  Mock predict: {mock.predict(__import__('numpy').zeros((224,224,3), dtype='uint8')):.4f}")

# Test 3: Model forward pass
import torch
print("[Test 3] CNN+DSP+LSTM forward pass...")
model = CNNDSPLSTMAntiSpoof(num_classes=2, backbone="mobilenet_v2", pretrained=False)
dummy = torch.randn(1, 3, 224, 224)
logits = model(dummy)
probs = torch.softmax(logits, dim=1)
print(f"  Input:  {dummy.shape}")
print(f"  Logits: {logits.shape} = {logits.detach().numpy()}")
print(f"  Probs:  live={probs[0,0]:.4f}, spoof={probs[0,1]:.4f}")
assert logits.shape == (1, 2)
print("  OK")

# Test 4: Multi-frame
print("[Test 4] Multi-frame forward pass...")
video = torch.randn(1, 3, 3, 224, 224)  # 1 sample, 3 frames
logits_v = model.forward_multi_frame(video)
print(f"  Input:  {video.shape}")
print(f"  Output: {logits_v.shape}")
assert logits_v.shape == (1, 2)
print("  OK")

# Test 5: DSP standalone
print("[Test 5] DSP module...")
import numpy as np
from dsp_utils import extract_frequency_features
gray = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
freq_feat = extract_frequency_features(gray, feature_dim=64)
print(f"  Input:  grayscale {gray.shape}")
print(f"  Output: {freq_feat.shape}")
assert freq_feat.shape == (64,)
print("  OK")

# Summary
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print()
print("=" * 55)
print(f"  ALL TESTS PASSED")
print(f"  Model: {total:,} params ({trainable:,} trainable)")
print("=" * 55)
