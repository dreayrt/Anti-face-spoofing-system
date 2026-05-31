import torch
import torch.nn as nn
import torchvision.models as models

class SwinTransformerBaseline(nn.Module):
    """
    Swin Transformer V2 Baseline cho bài toán Face Anti-Spoofing.
    Đây là mô hình thuần không gian (spatial-only) dựa trên cơ chế Self-Attention,
    được dùng để so sánh hiệu năng với mô hình lai (CNN + DSP + LSTM).

    Args:
        num_classes (int): Số lượng class đầu ra (mặc định là 2: live và spoof).
        pretrained (bool): Dùng trọng số pre-trained trên ImageNet hay không.
        model_name (str): Phiên bản Swin V2 ('swin_v2_t', 'swin_v2_s', 'swin_v2_b').
    """
    def __init__(self, num_classes=2, pretrained=True, model_name='swin_v2_t'):
        super().__init__()
        
        # Chọn kiến trúc Swin Transformer V2
        if model_name == 'swin_v2_t':
            weights = models.Swin_V2_T_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.swin_v2_t(weights=weights)
        elif model_name == 'swin_v2_s':
            weights = models.Swin_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.swin_v2_s(weights=weights)
        elif model_name == 'swin_v2_b':
            weights = models.Swin_V2_B_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.swin_v2_b(weights=weights)
        else:
            raise ValueError(f"Model name {model_name} không được hỗ trợ. Dùng swin_v2_t, swin_v2_s, hoặc swin_v2_b.")
        
        # Lấy số chiều in_features của lớp phân loại gốc
        in_features = self.backbone.head.in_features
        
        # Thay thế lớp classifier (head) cho bài toán nhị phân
        self.backbone.head = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        """
        Input: (B, 3, 224, 224)
        Output: (B, num_classes)
        """
        # Nếu truyền vào multi-frame (B, T, 3, 224, 224) một cách nhầm lẫn,
        # fallback về frame cuối cùng hoặc gộp batch và time.
        # Ở đây ta chỉ hỗ trợ Single Frame cho baseline này.
        if x.dim() == 5:
            x = x[:, -1, ...] # Lấy frame cuối cùng
            
        return self.backbone(x)
