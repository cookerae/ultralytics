# ultralytics/nn/dynamic_head.py

import math
import torch
from torch import nn
import torch.nn.functional as F

# Import necessary components from YOLOv8
from .block import DFL
from .conv import Conv, DWConv
from .head import Detect
from ultralytics.utils.tal import make_anchors

# ==============================================================================
# ⚠️ DynamicHead核心组件 - 正确实现
# ==============================================================================

def hard_sigmoid(x):
    """Hard sigmoid activation function"""
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)

class ScaleAwareAttention(nn.Module):
    """尺度感知注意力 - πL"""
    def __init__(self, num_levels=3):
        super().__init__()
        self.num_levels = num_levels
        self.fc = nn.Linear(num_levels, num_levels)
        
    def forward(self, features):
        # features: list of [B, C, H, W] for each level
        B, C = features[0].shape[:2]
        
        # Global average pooling across spatial and channel dimensions
        level_features = []
        for feat in features:
            # [B, C, H, W] -> [B, 1]
            pooled = F.adaptive_avg_pool2d(feat, 1).view(B, C).mean(dim=1, keepdim=True)
            level_features.append(pooled)
        
        # [B, L]
        level_vector = torch.cat(level_features, dim=1)
        
        # Generate attention weights
        attention = hard_sigmoid(self.fc(level_vector))  # [B, L]
        
        # Apply attention to features
        weighted_features = []
        for i, feat in enumerate(features):
            weight = attention[:, i:i+1, None, None]  # [B, 1, 1, 1]
            weighted_features.append(feat * weight)
            
        return weighted_features

class SpatialAwareAttention(nn.Module):
    """空间感知注意力 - πS，基于可变形卷积"""
    def __init__(self, in_channels, num_levels=3):
        super().__init__()
        self.num_levels = num_levels
        self.in_channels = in_channels
        
        # 简化实现：直接使用卷积代替可变形卷积
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=2)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=3)
        
        self.fusion_conv = nn.Conv2d(in_channels * 3, in_channels, 1)
        self.norm = nn.BatchNorm2d(in_channels)  # 改用BatchNorm2d
        
    def forward(self, x):
        """
        Args:
            x: 单个特征图 [B, C, H, W]
        Returns:
            增强后的特征图 [B, C, H, W]
        """
        # 多尺度空间特征提取
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        
        # 确保所有特征图尺寸一致
        h, w = feat1.shape[2], feat1.shape[3]
        if feat2.shape[2:] != (h, w):
            feat2 = F.interpolate(feat2, size=(h, w), mode='bilinear', align_corners=False)
        if feat3.shape[2:] != (h, w):
            feat3 = F.interpolate(feat3, size=(h, w), mode='bilinear', align_corners=False)
        
        # 特征融合
        fused = torch.cat([feat1, feat2, feat3], dim=1)  # [B, 3*C, H, W]
        output = self.fusion_conv(fused)  # [B, C, H, W]
        output = self.norm(output)
        
        # 残差连接
        output = output + x
        
        return output

class TaskAwareAttention(nn.Module):
    """任务感知注意力 - πC"""
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        hidden_channels = max(in_channels // reduction, 16)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, in_channels * 4)  # α1, α2, β1, β2
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 全局池化
        pooled = self.global_pool(x).view(B, C)  # [B, C]
        
        # 生成控制参数
        params = self.fc2(F.relu(self.fc1(pooled)))  # [B, C*4]
        params = params.view(B, C, 4)  # [B, C, 4]
        
        # 分离参数并normalize到[-1, 1]
        alpha1 = torch.tanh(params[:, :, 0:1])  # [B, C, 1]
        alpha2 = torch.tanh(params[:, :, 1:2])
        beta1 = torch.tanh(params[:, :, 2:3])
        beta2 = torch.tanh(params[:, :, 3:4])
        
        # Reshape for broadcasting
        alpha1 = alpha1.view(B, C, 1, 1)
        alpha2 = alpha2.view(B, C, 1, 1)
        beta1 = beta1.view(B, C, 1, 1)
        beta2 = beta2.view(B, C, 1, 1)
        
        # 双路径激活：max(α1*x + β1, α2*x + β2)
        path1 = alpha1 * x + beta1
        path2 = alpha2 * x + beta2
        output = torch.max(path1, path2)
        
        return output

class DynamicHeadDetect(Detect):
    """
    DynamicHead YOLOv8 Detection Head.
    """

    def __init__(self, nc=80, ch=(), num_blocks=6):
        """
        Initialize the DynamicHeadDetect layer.
        
        Args:
            nc (int): Number of classes.
            ch (tuple): A tuple of input channels for each feature level.
            num_blocks (int): The number of DynamicHead blocks to stack. Default: 6.
        """
        super().__init__(nc, ch)
        self.num_blocks = num_blocks
        self.num_levels = len(ch)
        
        # 统一特征通道数
        self.input_proj = nn.ModuleList([
            nn.Conv2d(c, 256, 1) for c in ch
        ])
        
        # DynamicHead blocks
        self.scale_attns = nn.ModuleList([
            ScaleAwareAttention(self.num_levels) 
            for _ in range(num_blocks)
        ])
        
        self.spatial_attns = nn.ModuleList([
            SpatialAwareAttention(256, self.num_levels) 
            for _ in range(num_blocks)
        ])
        
        self.task_attns = nn.ModuleList([
            TaskAwareAttention(256) 
            for _ in range(num_blocks)
        ])
        
        # 重建检测头 - 使用256通道
        c2, c3 = max((16, 256 // 4, self.reg_max * 4)), max(256, min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(256, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1))
            for _ in range(self.nl)
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(256, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1))
            for _ in range(self.nl)
        )

    def forward(self, x):
        """Forward pass through DynamicHead"""
        # 保存原始输入用于最终输出
        original_x = [xi.clone() for xi in x]
        
        # 1. 输入投影，统一通道数
        projected_features = []
        for i, feat in enumerate(x):
            projected = self.input_proj[i](feat)
            projected_features.append(projected)
        
        # 2. 通过多个DynamicHead blocks处理
        current_features = projected_features
        for i in range(self.num_blocks):
            # Scale-aware attention
            current_features = self.scale_attns[i](current_features)
            
            # Spatial-aware attention - 处理每个level
            for j in range(len(current_features)):
                current_features[j] = self.spatial_attns[i](current_features[j])
            
            # Task-aware attention
            for j in range(len(current_features)):
                current_features[j] = self.task_attns[i](current_features[j])
        
        # 3. 最终预测 - 使用增强后的特征
        for i in range(self.nl):
            original_x[i] = torch.cat((self.cv2[i](current_features[i]), self.cv3[i](current_features[i])), 1)

        if self.training:
            return original_x
        
        # 推理路径 - 直接调用父类的_inference方法
        return self._inference(original_x) if self.export else (self._inference(original_x), original_x)