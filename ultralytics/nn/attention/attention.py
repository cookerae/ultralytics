######################  CBAM  GAM  ####     START   by  AI&CV  ###############################
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

# Conv import removed to avoid circular import - not used in GAM_Attention

class ChannelAttention(nn.Module):
    # Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet    
    def __init__(self, channels: int) -> None:
         super().__init__()
         self.pool = nn.AdaptiveAvgPool2d(1)
         self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
         self.act = nn.Sigmoid()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
         return x * self.act(self.fc(self.pool(x)))

class SpatialAttention(nn.Module):
    # Spatial-attention module    
    def __init__(self, kernel_size=7):
         super().__init__()
         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
         padding = 3 if kernel_size == 7 else 1
         self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
         self.act = nn.Sigmoid()

    def forward(self, x):
         return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))

class CBAM(nn.Module):
# Convolutional Block Attention Module    
    def __init__(self, c1, c2, kernel_size=7): # ch_in, kernels        
        super().__init__()
        self.channel_attention = ChannelAttention(c2)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))



def channel_shuffle(x, groups=2): ##shuffle channel    
    # RESHAPE----->transpose------->Flatten    
    B, C, H, W = x.size()
    out = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous()
    out = out.view(B, C, H, W)
    return out

class GAM_Attention(nn.Module):
    # https://paperswithcode.com/paper/global-attention-mechanism-retain-information    
    def __init__(self, c1, c2, group=True, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c1 / rate), c1)
            )
        self.spatial_attention = nn.Sequential(
            
            nn.Conv2d(c1, c1 // rate, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(c1, int(c1 / rate),
                                                                                                     kernel_size=7,
                                                                                                     padding=3),
            nn.BatchNorm2d(int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // rate, c2, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(int(c1 / rate), c2,
                                                                                                     kernel_size=7,
                                                                                                     padding=3),
            nn.BatchNorm2d(c2) 
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * x_channel_att
        x_spatial_att = self.spatial_attention(x).sigmoid()
        x_spatial_att = channel_shuffle(x_spatial_att, 4)
        out = x * x_spatial_att
        return out

######################  RFAConv  ####     START   by  AI&CV  ###############################

class RFAConv(nn.Module):
    """
    Receptive-Field Attention Convolution (RFAConv)
    
    RFAConv implements receptive-field attention that addresses the parameter sharing problem 
    of standard convolution by generating attention weights for each receptive field.
    
    Paper: RFAConv: Innovating Spatial Attention and Standard Convolutional Operation
    Reference: https://arxiv.org/abs/2304.03198
    """
    
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize RFAConv module.
        
        Args:
            c1 (int): Input channels
            c2 (int): Output channels  
            k (int): Kernel size
            s (int): Stride
            p (int): Padding
            g (int): Groups
            d (int): Dilation
            act (bool): Use activation
        """
        super().__init__()
        
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.s = s
        self.p = k // 2 if p is None else p
        self.g = g
        self.d = d
        
        # 生成感受野空间特征的分组卷积
        self.group_conv = nn.Conv2d(c1, c1 * k * k, kernel_size=k, stride=s, 
                                   padding=self.p, groups=c1, bias=False)
        
        # 用于生成注意力权重的模块
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attention_conv = nn.Conv2d(c1, k * k, 1, bias=False, groups=g)
        
        # 最终的卷积层
        self.final_conv = nn.Conv2d(c1 * k * k, c2, 1, stride=1, padding=0, 
                                   groups=g, bias=False)
        
        # 批归一化和激活
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
        
        # softmax用于归一化注意力权重
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        """
        Forward pass of RFAConv.
        
        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Output tensor after RFAConv processing
        """
        B, C, H, W = x.shape
        
        # 1. 生成感受野空间特征
        # 使用分组卷积展开感受野特征 [B, C*k*k, H', W']
        rf_features = self.group_conv(x)
        
        # 重新整形为 [B, C, k*k, H', W']
        rf_features = rf_features.view(B, C, self.k * self.k, 
                                     rf_features.size(2), rf_features.size(3))
        
        # 2. 生成注意力权重
        # 全局平均池化 [B, C, 1, 1]
        pooled = self.avg_pool(x)
        
        # 生成注意力权重 [B, k*k, 1, 1]
        attention_weights = self.attention_conv(pooled)
        
        # Softmax归一化注意力权重
        attention_weights = self.softmax(attention_weights)
        
        # 扩展维度以匹配感受野特征 [B, 1, k*k, 1, 1]
        attention_weights = attention_weights.unsqueeze(1)
        
        # 3. 应用注意力权重到感受野特征
        # [B, C, k*k, H', W'] * [B, 1, k*k, 1, 1] = [B, C, k*k, H', W']
        attended_features = rf_features * attention_weights
        
        # 4. 重新整形并通过最终卷积
        # [B, C*k*k, H', W']
        attended_features = attended_features.view(B, C * self.k * self.k,
                                                 attended_features.size(3),
                                                 attended_features.size(4))
        
        # 最终卷积 [B, C2, H', W']
        output = self.final_conv(attended_features)
        
        # 批归一化和激活
        output = self.bn(output)
        output = self.act(output)
        
        return output

class RFCBAMConv(nn.Module):
    """
    RFCBAMConv: Receptive-Field CBAM convolution
    结合了RFA和CBAM注意力机制的卷积模块
    """
    
    def __init__(self, c1, c2, k=3, s=1, p=None):
        super().__init__()
        
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.s = s
        self.p = k // 2 if p is None else p
        
        # RFA组件
        self.group_conv = nn.Conv2d(c1, c1 * k * k, kernel_size=k, stride=s,
                                   padding=self.p, groups=c1, bias=False)
        
        # 通道注意力（SE模块）
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // 16, c1, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        # 最终卷积
        self.final_conv = nn.Conv2d(c1 * k * k, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. 通道注意力
        channel_att = self.se(x)
        x_channel = x * channel_att
        
        # 2. 空间注意力
        avg_out = torch.mean(x_channel, dim=1, keepdim=True)
        max_out = torch.max(x_channel, dim=1, keepdim=True)[0]
        spatial_att = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        x_spatial = x_channel * spatial_att
        
        # 3. 感受野特征生成
        rf_features = self.group_conv(x_spatial)
        
        # 4. 最终处理
        output = self.final_conv(rf_features)
        output = self.bn(output)
        output = self.act(output)
        
        return output

######################  RFAConv  ####     END   by  AI&CV  ###############################