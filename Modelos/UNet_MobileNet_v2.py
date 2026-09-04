import torch
import torch.nn as nn
import torchvision.models as models

class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) * 2 - Utilizada no caminho de subida (Decoder)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetMobileNetV2(nn.Module):
    """
    U-Net Otimizada: Substituição de Backbone por MobileNetV2
    (Baseado em Růžička et al., 2023 e otimizado com decodificador bilinear Kang et al., 2024)
    
    - Encoder: MobileNetV2 adaptado para aceitar 'in_channels' customizados na entrada.
    - Decoder: Estrutura leve baseada em Interpolação Bilinear (estática).
    - Skip Connections: Concatenações correspondentes aos blocos do MobileNetV2.
    """
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()
        
        # 1. Carrega o extrator de características padrão do MobileNetV2 do Torchvision
        mobilenet = models.mobilenet_v2(pretrained=False)
        encoder_features = mobilenet.features
        
        # 2. Adaptação da primeira camada convolucional para receber os canais do STARCOP (ex: 4 canais)
        # O MobileNetV2 por padrão espera 3 canais de entrada (RGB).
        first_conv = encoder_features[0][0]
        encoder_features[0][0] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False
        )
        
        # 3. Divisão fatiada do MobileNetV2 para mapeamento de skip connections
        self.enc0 = nn.Sequential(*encoder_features[0:2])   # Saída: 256x256, 16 canais (Layer 1)
        self.enc1 = nn.Sequential(*encoder_features[2:4])   # Saída: 128x128, 24 canais (Layer 3)
        self.enc2 = nn.Sequential(*encoder_features[4:7])   # Saída: 64x64,   32 canais (Layer 6)
        self.enc3 = nn.Sequential(*encoder_features[7:14])  # Saída: 32x32,   96 canais (Layer 13)
        self.enc4 = nn.Sequential(*encoder_features[14:19]) # Saída: 16x16,  1280 canais (Layer 18 - Bottleneck)

        # 4. Caminho de Subida (Decoder leve de Upsampling Bilinear Estático)
        # Decoder 1: Reergue de 16x16 para 32x32
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 1280 (up) + 96 (skip connection de enc3) = 1376 canais de entrada
        self.conv1 = DoubleConv(1280 + 96, 128)
        
        # Decoder 2: Reergue de 32x32 para 64x64
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 128 (up) + 32 (skip connection de enc2) = 160 canais de entrada
        self.conv2 = DoubleConv(128 + 32, 64)
        
        # Decoder 3: Reergue de 64x64 para 128x128
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 64 (up) + 24 (skip connection de enc1) = 88 canais de entrada
        self.conv3 = DoubleConv(64 + 24, 32)
        
        # Decoder 4: Reergue de 128x128 para 256x256
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 32 (up) + 16 (skip connection de enc0) = 48 canais de entrada
        self.conv4 = DoubleConv(32 + 16, 16)
        
        # Decoder Final: Reergue de 256x256 para a resolução original 512x512
        self.up_final = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # --- ENCODER (MobileNetV2 adaptado) ---
        s0 = self.enc0(x)   # Resolução: 256x256, 16 canais (Skip 1/2)
        s1 = self.enc1(s0)  # Resolução: 128x128, 24 canais (Skip 1/4)
        s2 = self.enc2(s1)  # Resolução: 64x64,   32 canais (Skip 1/8)
        s3 = self.enc3(s2)  # Resolução: 32x32,   96 canais (Skip 1/16)
        b  = self.enc4(s3)  # Resolução: 16x16,  1280 canais (Bottleneck)
        
        # --- DECODER (Upsampling Bilinear + Skip Connections) ---
        u1 = self.up1(b)
        u1 = torch.cat([u1, s3], dim=1) # Concatenação no nível 1/16
        c1 = self.conv1(u1)
        
        u2 = self.up2(c1)
        u2 = torch.cat([u2, s2], dim=1) # Concatenação no nível 1/8
        c2 = self.conv2(u2)
        
        u3 = self.up3(c2)
        u3 = torch.cat([u3, s1], dim=1) # Concatenação no nível 1/4
        c3 = self.conv3(u3)
        
        u4 = self.up4(c3)
        u4 = torch.cat([u4, s0], dim=1) # Concatenação no nível 1/2
        c4 = self.conv4(u4)
        
        u_final = self.up_final(c4)     # Resolução: 512x512
        logits = self.out_conv(u_final)
        
        return logits