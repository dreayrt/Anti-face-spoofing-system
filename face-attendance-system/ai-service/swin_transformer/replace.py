import sys, re

with open('test.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Imports
content = re.sub(r'sys\.path\.insert.*?from antispoof_model import CNNDSPLSTMAntiSpoof', 'from model import SwinTransformerBaseline', content, flags=re.DOTALL)

# 2. Paths
content = content.replace('PROJECT_ROOT = Path(__file__).resolve().parent.parent\nDATASET_DIR = PROJECT_ROOT / "dataset"', 'PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent\nDATASET_DIR = PROJECT_ROOT / "dataset"')

# 3. CLI arguments
content = content.replace('description="Test CNN+DSP+LSTM Anti-Spoofing Model"', 'description="Test Swin Transformer Baseline Anti-Spoofing Model"')
content = content.replace('parser.add_argument("--checkpoint", type=str,\n                        default="models/weights/antispoof_cnn_dsp_lstm.pth",', 'parser.add_argument("--checkpoint", type=str,\n                        default="models/weights/antispoof_swin_transformer.pth",')
content = content.replace('--backbone', '--model-name')
content = content.replace('help="CNN backbone architecture (default: efficientnet_b0)"', 'help="Swin Transformer architecture (default: swin_v2_t)"')
content = content.replace('default="efficientnet_b0"', 'default="swin_v2_t"')

# 4. Model instantiation in main
model_old = r'print\(f"\\n\[Model\] Loading CNN\+DSP\+LSTM \(backbone=\{args\.backbone\}\)\.\.\."\)\n\s+model = CNNDSPLSTMAntiSpoof\(\n\s+num_classes=2,\n\s+backbone=args\.backbone,\n\s+lstm_hidden=256,\n\s+lstm_layers=2,\n\s+dsp_output_dim=256,\n\s+pretrained=False,\n\s+\)\.to\(device\)'
model_new = 'print(f"\\n[Model] Loading SwinTransformerBaseline (model_name={args.model_name})...")\n    model = SwinTransformerBaseline(\n        num_classes=2,\n        pretrained=False,\n        model_name=args.model_name\n    ).to(device)'
content = re.sub(model_old, model_new, content)

# 5. Remove multi_frame references
content = content.replace('if args.multi_frame:', 'if False:')

# 6. Chart titles
content = content.replace('CNN + DSP + LSTM Anti-Spoofing', 'Swin Transformer Baseline Anti-Spoofing')

with open('test.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("test.py updated successfully!")
