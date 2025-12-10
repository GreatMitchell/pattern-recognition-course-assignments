import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from .resnet18 import FrameFeatureExtractor

class VideoRecognitionModel(nn.Module):
    """
    完整的视频识别模型：2D CNN提取帧特征 + LSTM建模时序
    """
    def __init__(self, num_classes=20, num_frames=10, lstm_hidden_size=256, lstm_num_layers=1):
        super().__init__()
        # 视觉特征提取器（三个模态分别一个）
        self.rgb_backbone = FrameFeatureExtractor(modality='rgb')
        self.depth_backbone = FrameFeatureExtractor(modality='depth')
        self.infrared_backbone = FrameFeatureExtractor(modality='infrared')
        
        # 时序建模层（LSTM）
        self.lstm = nn.LSTM(
            input_size=self.rgb_backbone.feature_dim * 3, # 拼接三个模态的特征
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,  # 输入形状为 (batch, seq, feature)
            bidirectional=True  # 使用双向LSTM以获取更丰富的上下文
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(lstm_hidden_size * 2, 128),  # *2 因为是双向LSTM
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, rgb_clip, depth_clip, infrared_clip, lengths=None):
        """
        输入：
        rgb_clip, depth_clip, infrared_clip: (B, T, C, H, W)
        lengths: optional tensor (B,) 表示每个样本真实帧数（未填充前）
        返回：
        logits: (B, num_classes)
        """
        # 1. 提取各模态帧特征 (B, T, feat)
        rgb_features = self.rgb_backbone(rgb_clip)        # (B, T, 512)
        depth_features = self.depth_backbone(depth_clip)
        infrared_features = self.infrared_backbone(infrared_clip)

        # 2. 拼接模态特征 -> (B, T, feat*3)
        combined = torch.cat([rgb_features, depth_features, infrared_features], dim=2)

        # 3. 可变长度处理：如果给出 lengths，则使用 pack_padded_sequence
        if lengths is not None:
            # lengths 应为 CPU 上的 1D LongTensor
            if isinstance(lengths, torch.Tensor):
                lengths_cpu = lengths.cpu()
            else:
                lengths_cpu = torch.tensor(lengths, dtype=torch.long)

            packed = rnn_utils.pack_padded_sequence(combined, lengths_cpu, batch_first=True, enforce_sorted=False)
            packed_out, (h_n, c_n) = self.lstm(packed)
            # h_n shape: (num_layers * num_directions, B, hidden_size)
            # 取最后一层的前向与后向 hidden state（对应索引为 -2, -1）
            if self.lstm.bidirectional:
                last_forward = h_n[-2]  # (B, hidden)
                last_backward = h_n[-1] # (B, hidden)
                video_features = torch.cat([last_forward, last_backward], dim=1)  # (B, hidden*2)
            else:
                video_features = h_n[-1]  # (B, hidden)
        else:
            # 简单情形：用全序列（无padding）并取最后时间步
            lstm_out, (h_n, c_n) = self.lstm(combined)  # lstm_out: (B, T, hidden*2)
            if self.lstm.bidirectional:
                # safer: 从 h_n 提取
                last_forward = h_n[-2]
                last_backward = h_n[-1]
                video_features = torch.cat([last_forward, last_backward], dim=1)
            else:
                video_features = h_n[-1]

        # 4. 分类
        out = self.classifier(video_features)
        return out