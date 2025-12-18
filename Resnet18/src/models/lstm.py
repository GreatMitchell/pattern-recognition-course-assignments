import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from .resnet18 import FrameFeatureExtractor

class VideoRecognitionModel(nn.Module):
    """
    完整的视频识别模型：2D CNN提取帧特征 + LSTM建模时序
    支持单模态（RGB）或三模态（RGB+Depth+Infrared）拼接
    """
    def __init__(self, num_classes=20, num_frames=10, lstm_hidden_size=256, lstm_num_layers=1, modalities=['rgb'], learn_weights=False):
        super().__init__()
        self.modalities = modalities  # 支持 ['rgb'], ['rgb','depth','infrared'] 等
        self.learn_weights = learn_weights
        
        # 视觉特征提取器（根据模态初始化）
        self.backbones = nn.ModuleDict()
        for mod in modalities:
            self.backbones[mod] = FrameFeatureExtractor(modality=mod)
        
        # 计算总特征维度
        self.feature_dim = sum(self.backbones[mod].feature_dim for mod in modalities)
        
        # 可学习权重（如果启用）
        if learn_weights:
            self.weights = nn.Parameter(torch.ones(len(modalities)))
        else:
            self.weights = None
        
        # 时序建模层（LSTM）
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,  # 拼接后的总特征维度
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(lstm_hidden_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, rgb_clip, depth_clip=None, infrared_clip=None, lengths=None, weights=None):
        """
        输入：
        rgb_clip: (B, T, C, H, W) - 必须提供
        depth_clip, infrared_clip: (B, T, C, H, W) - 可选，根据 modalities 决定是否使用
        lengths: optional tensor (B,) 表示每个样本真实帧数（未填充前）
        weights: optional list or tensor of fixed weights for each modality (e.g., [1.0, 0.5, 0.5])
                 如果提供，将覆盖学习权重
        返回：
        logits: (B, num_classes)
        """
        # 确定使用的权重：优先使用传入的 weights，否则使用学习权重，否则默认 1.0
        if weights is not None:
            weight_tensor = torch.tensor(weights, dtype=torch.float32, device=rgb_clip.device)
        elif self.weights is not None:
            weight_tensor = self.weights
        else:
            weight_tensor = torch.ones(len(self.modalities), dtype=torch.float32, device=rgb_clip.device)
        
        # 1. 提取各模态帧特征并加权拼接
        features_list = []
        mod_idx = 0
        if 'rgb' in self.modalities:
            feat = self.backbones['rgb'](rgb_clip) * weight_tensor[mod_idx]
            features_list.append(feat)
            mod_idx += 1
        if 'depth' in self.modalities and depth_clip is not None:
            feat = self.backbones['depth'](depth_clip) * weight_tensor[mod_idx]
            features_list.append(feat)
            mod_idx += 1
        if 'infrared' in self.modalities and infrared_clip is not None:
            feat = self.backbones['infrared'](infrared_clip) * weight_tensor[mod_idx]
            features_list.append(feat)
            mod_idx += 1
        
        combined_features = torch.cat(features_list, dim=2)  # (B, T, feature_dim)

        # 2. 可变长度处理：如果给出 lengths，则使用 pack_padded_sequence
        if lengths is not None:
            if isinstance(lengths, torch.Tensor):
                lengths_cpu = lengths.cpu()
            else:
                lengths_cpu = torch.tensor(lengths, dtype=torch.long)

            packed = rnn_utils.pack_padded_sequence(combined_features, lengths_cpu, batch_first=True, enforce_sorted=False)
            packed_out, (h_n, c_n) = self.lstm(packed)
            if self.lstm.bidirectional:
                last_forward = h_n[-2]
                last_backward = h_n[-1]
                video_features = torch.cat([last_forward, last_backward], dim=1)
            else:
                video_features = h_n[-1]
        else:
            lstm_out, (h_n, c_n) = self.lstm(combined_features)
            if self.lstm.bidirectional:
                last_forward = h_n[-2]
                last_backward = h_n[-1]
                video_features = torch.cat([last_forward, last_backward], dim=1)
            else:
                video_features = h_n[-1]

        # 3. 分类
        out = self.classifier(video_features)
        return out