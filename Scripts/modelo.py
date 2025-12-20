import torch
import torch.nn as nn

class UNetColorizer(nn.Module):
    def __init__(self):
        super(UNetColorizer, self).__init__()
        
        # --- ENCODER (Downsampling) ---
        # Input: 1 canal (L - Lightness)
        self.enc1 = self.conv_block(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        # --- BOTTLENECK ---
        self.bottleneck = self.conv_block(256, 512)
        
        # --- DECODER (Upsampling) ---
        # Skip Connections: Concatenamos o output do encoder correspondente
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.conv_block(512 + 256, 256) 
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = self.conv_block(256 + 128, 128)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = self.conv_block(128 + 64, 64)
        
        # --- OUTPUT ---
        # Output: 2 canais (a, b - Espectro de cor)
        self.final = nn.Conv2d(64, 2, kernel_size=1)
        # Tanh força o output para [-1, 1], o que facilita o treino com dados normalizados
        self.tanh = nn.Tanh() 

    def conv_block(self, in_c, out_c):
        """Bloco duplo de convolução (Conv -> BN -> ReLU -> Conv -> BN -> ReLU)"""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # Bottleneck
        b = self.bottleneck(p3)
        
        # Decoder
        u1 = self.up1(b)
        u1 = torch.cat([u1, e3], dim=1) # Skip Connection
        d1 = self.dec1(u1)
        
        u2 = self.up2(d1)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)
        
        u3 = self.up3(d2)
        u3 = torch.cat([u3, e1], dim=1)
        d3 = self.dec3(u3)
        
        out = self.final(d3)
        return self.tanh(out)