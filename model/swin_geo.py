import torch
import torch.nn as nn
from torchvision.models import swin_t

class SwinGeo(nn.Module):
    """
    Swin-UNet Architecture for Geospatial dense prediction (LST).
    Takes input patches of (B, 13, patch_size, patch_size) and outputs
    dense predictions of (B, 1, patch_size, patch_size).
    Incorporates skip connections from the Swin encoder to the decoder.
    """
    def __init__(self, in_channels=13, out_channels=1, patch_size=32):
        super().__init__()
        
        self.backbone = swin_t(weights=None)
        # Modify the first patch embedding layer
        self.backbone.features[0][0] = nn.Conv2d(in_channels, 96, kernel_size=(4, 4), stride=(4, 4))
        
        # Decoder upsampling blocks with skip connections
        # feat7: 1x1x768. Upsample to 2x2.
        self.up1 = nn.ConvTranspose2d(768, 256, kernel_size=2, stride=2)
        # Concat feat5 (384) + 256 = 640
        self.conv1 = nn.Sequential(
            nn.Conv2d(640, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Upsample to 4x4
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # Concat feat3 (192) + 128 = 320
        self.conv2 = nn.Sequential(
            nn.Conv2d(320, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Upsample to 8x8
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # Concat feat1 (96) + 64 = 160
        self.conv3 = nn.Sequential(
            nn.Conv2d(160, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Final upsample to 32x32 (scale factor 4 to match initial embedding stride 4)
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=4)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        """
        x: (B, 13, H, W)
        """
        # Encoder passes
        x0 = self.backbone.features[0](x)
        feat1 = self.backbone.features[1](x0) # Stage 1 (B, 8, 8, 96)
        
        x2 = self.backbone.features[2](feat1)
        feat3 = self.backbone.features[3](x2) # Stage 2 (B, 4, 4, 192)
        
        x4 = self.backbone.features[4](feat3)
        feat5 = self.backbone.features[5](x4) # Stage 3 (B, 2, 2, 384)
        
        x6 = self.backbone.features[6](feat5)
        feat7 = self.backbone.features[7](x6) # Stage 4 (B, 1, 1, 768)
        
        # Permute to (B, C, H, W)
        f1 = feat1.permute(0, 3, 1, 2).contiguous()
        f3 = feat3.permute(0, 3, 1, 2).contiguous()
        f5 = feat5.permute(0, 3, 1, 2).contiguous()
        f7 = feat7.permute(0, 3, 1, 2).contiguous()
        
        # Decoder
        d1 = self.up1(f7) # (B, 256, 2, 2)
        d1 = torch.cat([d1, f5], dim=1) # (B, 640, 2, 2)
        d1 = self.conv1(d1) # (B, 256, 2, 2)
        
        d2 = self.up2(d1) # (B, 128, 4, 4)
        d2 = torch.cat([d2, f3], dim=1) # (B, 320, 4, 4)
        d2 = self.conv2(d2) # (B, 128, 4, 4)
        
        d3 = self.up3(d2) # (B, 64, 8, 8)
        d3 = torch.cat([d3, f1], dim=1) # (B, 160, 8, 8)
        d3 = self.conv3(d3) # (B, 64, 8, 8)
        
        d4 = self.up4(d3) # (B, 64, 32, 32)
        out = self.final_conv(d4) # (B, 1, 32, 32)
        
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = SwinGeo()
    x = torch.randn(2, 13, 32, 32)
    y = model(x)
    print("Input:", x.shape)
    print("Output:", y.shape)
    print("Parameters:", model.count_parameters())
