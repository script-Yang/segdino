import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import FeatureFusionBlock, _make_scratch
def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(
        self, 
        nclass,
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024],
    ):
        super(DPTHead, self).__init__()
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features, nclass, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv(path_1)
        
        return out


class DPT(nn.Module):
    def __init__(
        self, 
        encoder_size='base', 
        nclass=21,
        features=128, 
        out_channels=[96, 192, 384, 768], 
        use_bn=False,
        backbone = None
    ):
        super(DPT, self).__init__()
        
        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11], 
            'large': [4, 11, 17, 23], 
            'giant': [9, 19, 29, 39]
        }
        
        self.encoder_size = encoder_size
        self.backbone = backbone
        self.head = DPTHead(nclass, self.backbone.embed_dim, features, use_bn, out_channels=out_channels)
        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)
        
    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
        # features = self.backbone.get_intermediate_layers(
        #     x, n = self.intermediate_layer_idx[self.encoder_size], reshape=True, norm=True 
        # )
        features = self.backbone.get_intermediate_layers(
            x, n = self.intermediate_layer_idx[self.encoder_size]
        )
        
        out = self.head(features, patch_h, patch_w)
        out = F.interpolate(out, (patch_h * 16, patch_w * 16), mode='bilinear', align_corners=True)
        return out
