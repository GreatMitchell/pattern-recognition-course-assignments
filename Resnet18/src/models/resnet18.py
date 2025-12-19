import torch.nn as nn
import torchvision.models as models
import torch

class FrameFeatureExtractor(nn.Module):
    """
    输入：一批视频片段，形状为 (batch_size, num_frames, C, H, W)
    输出：一批视频特征，形状为 (batch_size, num_frames, feature_dim)
    """
    def __init__(self, modality='rgb', backbone='resnet18', pretrained=True):
        super().__init__()
        # 加载预训练的ResNet，并去掉最后的全连接层
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # 根据模态调整输入通道
        if modality not in ['rgb', 'depth', 'infrared']:
            raise ValueError("modality必须是 'rgb'、 'depth' 或 'infrared'。")
        if modality in ('depth', 'infrared'):
            # 修改第一层卷积以适应单通道输入
            original_conv1 = resnet.conv1
            # bias 参数应为 bool
            has_bias = (original_conv1.bias is not None)
            resnet.conv1 = nn.Conv2d(1, original_conv1.out_channels,
                                     kernel_size=original_conv1.kernel_size,
                                     stride=original_conv1.stride,
                                     padding=original_conv1.padding,
                                     bias=has_bias)
            # 平均预训练权重以适应单通道
            with torch.no_grad():
                resnet.conv1.weight.data.copy_(original_conv1.weight.data.mean(dim=1, keepdim=True))
                if has_bias:
                    resnet.conv1.bias.data.copy_(original_conv1.bias.data)

        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # 保留直到全局池化层
        # 解冻所有参数以进行完整微调
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

        # ResNet18最终的特征维度是512
        self.feature_dim = 512

    def forward(self, x):
        """
        x 形状: (batch_size, num_frames, C, H, W)
        返回形状: (batch_size, num_frames, feature_dim)
        """
        B, T, C, H, W = x.shape
        # 合并批次和帧维度，以便同时处理所有帧
        x = x.view(B * T, C, H, W)
        # 提取特征
        feats = self.feature_extractor(x)   # 形状: (B*T, 512, 1, 1)
        feats = feats.view(B * T, -1)       # 更稳健的扁平化
        feats = feats.view(B, T, -1)        # (B, T, 512)
        return feats