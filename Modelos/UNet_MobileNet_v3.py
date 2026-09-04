import torch
import torch.nn as nn
import torchvision.models as models

class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) * 2 - Utilizada nas conexões do decoder"""
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

class UNetMobileNetV3(nn.Module):
    """
    U-Net Otimizada: Substituição de Backbone por MobileNetV3-Small
    (Inspirada em Růžička et al., 2023 e Herec et al., 2025)
    
    - Encoder: MobileNetV3-Small adaptado para receber canais arbitrários na entrada.
    - Decoder: Estrutura leve baseada em Interpolação Bilinear (estática).
    - Skip Connections: Ajustadas para casar milimetricamente com cada downsampling do encoder.
    """
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()
        
        # 1. Carrega o extrator de características padrão do MobileNetV3-Small
        mobilenet = models.mobilenet_v3_small(pretrained=False)
        encoder_features = mobilenet.features
        
        # 2. Adaptação da primeira camada convolucional para receber 'in_channels' do STARCOP
        first_conv = encoder_features[0][0]
        encoder_features[0][0] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False
        )
        
        # 3. Mapeamento preciso de cada estágio do MobileNetV3-Small para manter correspondência espacial
        self.enc_stage0 = nn.Sequential(encoder_features[0])        # Saída: 256x256, 16 canais (Layer 0)
        self.enc_stage1 = nn.Sequential(encoder_features[1])        # Saída: 128x128, 16 canais (Layer 1)
        self.enc_stage2 = nn.Sequential(*encoder_features[2:4])     # Saída: 64x64,   24 canais (Layers 2-3)
        self.enc_stage3 = nn.Sequential(*encoder_features[4:9])     # Saída: 32x32,   48 canais (Layers 4-8)
        self.enc_stage4 = nn.Sequential(*encoder_features[9:13])    # Saída: 16x16,  576 canais (Layers 9-12 / Bottleneck)

        # 4. Caminho de Subida (Decodificador de Upsampling Bilinear Estático)
        # Decoder 1: Reergue de 16x16 para 32x32
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 576 (do up) + 48 (skip connection do enc_stage3) = 624 canais de entrada
        self.conv1 = DoubleConv(576 + 48, 96)
        
        # Decoder 2: Reergue de 32x32 para 64x64
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 96 (do up) + 24 (skip connection do enc_stage2) = 120 canais de entrada
        self.conv2 = DoubleConv(96 + 24, 48)
        
        # Decoder 3: Reergue de 64x64 para 128x128
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 48 (do up) + 16 (skip connection do enc_stage1) = 64 canais de entrada
        self.conv3 = DoubleConv(48 + 16, 24)

        # Decoder 4: Reergue de 128x128 para 256x256
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 24 (do up) + 16 (skip connection do enc_stage0) = 40 canais de entrada
        self.conv4 = DoubleConv(24 + 16, 16)
        
        # Decoder Final: Reergue de 256x256 para a resolução original 512x512
        self.up_final = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # --- CAMINHO DE EXTRAÇÃO (ENCODER) ---
        s0 = self.enc_stage0(x)      # Resolução: 256x256, 16 canais
        s1 = self.enc_stage1(s0)     # Resolução: 128x128, 16 canais
        s2 = self.enc_stage2(s1)     # Resolução: 64x64,   24 canais
        s3 = self.enc_stage3(s2)     # Resolução: 32x32,   48 canais
        b  = self.enc_stage4(s3)     # Resolução: 16x16,  576 canais (Bottleneck)

        # --- CAMINHO DE RECONSTRUÇÃO (DECODER) ---
        u1 = self.up1(b)             # Upsample para 32x32
        u1 = torch.cat([u1, s3], dim=1) # Concatenação espacial (skip connection 1/16)
        c1 = self.conv1(u1)          # Processamento para 96 canais

        u2 = self.up2(c1)            # Upsample para 64x64
        u2 = torch.cat([u2, s2], dim=1) # Concatenação espacial (skip connection 1/8)
        c2 = self.conv2(u2)          # Processamento para 48 canais

        u3 = self.up3(c2)            # Upsample para 128x128
        u3 = torch.cat([u3, s1], dim=1) # Concatenação espacial (skip connection 1/4)
        c3 = self.conv3(u3)          # Processamento para 24 canais

        u4 = self.up4(c3)            # Upsample para 256x256
        u4 = torch.cat([u4, s0], dim=1) # Concatenação espacial (skip connection 1/2)
        c4 = self.conv4(u4)          # Processamento para 16 canais

        u_final = self.up_final(c4)  # Upsample final para 512x512
        logits = self.out_conv(u_final)
        
        return logits