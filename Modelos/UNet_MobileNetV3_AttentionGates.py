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


class AttentionGate(nn.Module):
    """
    Attention Gate (Oktay et al., 2018 / Ahsan et al., 2025 - AttMetNet)
    Filtra as características do encoder (x) usando o sinal de controle (g) do decodificador,
    suprimindo respostas irrelevantes do fundo e realçando a região da pluma.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        net = self.relu(g1 + x1)
        attn = self.psi(net)
        return x * attn


class UNetMobileNetV3AttentionGates(nn.Module):
    """
    U-Net Otimizada: Encoder MobileNetV3-Small com Attention Gates nas Skip Connections.
    
    Referências:
    - Růžička et al. (2023) "Semantic Segmentation of Methane Plumes with Hyperspectral Machine Learning Models"
    - Ahsan et al. (2025) "AttMetNet: Attention-Enhanced Deep Neural Network for Methane Plume Detection..."
    - Oktay et al. (2018) "Attention U-Net: Learning Where to Look for the Pancreas"
    - Herec, Růžička & Pitoňák (2025) "Optimizing Deep Learning Models for On-Board Methane Detection..."
    """
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()
        
        # 1. Carrega o extrator de características padrão do MobileNetV3-Small
        mobilenet = models.mobilenet_v3_small(weights=None)
        encoder_features = mobilenet.features
        
        # 2. Adaptação da primeira camada convolucional para receber 'in_channels' do STARCOP (ex: RGB + mag1c = 4 canais)
        first_conv = encoder_features[0][0]
        encoder_features[0][0] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False
        )
        
        # 3. Mapeamento dos estágios do MobileNetV3-Small
        self.enc_stage0 = nn.Sequential(encoder_features[0])        # Saída: 256x256, 16 canais
        self.enc_stage1 = nn.Sequential(encoder_features[1])        # Saída: 128x128, 16 canais
        self.enc_stage2 = nn.Sequential(*encoder_features[2:4])     # Saída: 64x64,   24 canais
        self.enc_stage3 = nn.Sequential(*encoder_features[4:9])     # Saída: 32x32,   48 canais
        self.enc_stage4 = nn.Sequential(*encoder_features[9:13])    # Saída: 16x16,  576 canais (Bottleneck)

        # 4. Attention Gates para filtrar cada Skip Connection antes da concatenação
        self.ag1 = AttentionGate(F_g=576, F_l=48, F_int=48)
        self.ag2 = AttentionGate(F_g=96, F_l=24, F_int=24)
        self.ag3 = AttentionGate(F_g=48, F_l=16, F_int=16)
        self.ag4 = AttentionGate(F_g=24, F_l=16, F_int=16)

        # 5. Caminho de Subida (Decoder)
        # Decoder 1: 16x16 -> 32x32
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = DoubleConv(576 + 48, 96)
        
        # Decoder 2: 32x32 -> 64x64
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = DoubleConv(96 + 24, 48)
        
        # Decoder 3: 64x64 -> 128x128
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = DoubleConv(48 + 16, 24)

        # Decoder 4: 128x128 -> 256x256
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4 = DoubleConv(24 + 16, 16)
        
        # Decoder Final: 256x256 -> 512x512
        self.up_final = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # --- ENCODER ---
        s0 = self.enc_stage0(x)      # 256x256, 16 canais
        s1 = self.enc_stage1(s0)     # 128x128, 16 canais
        s2 = self.enc_stage2(s1)     # 64x64,   24 canais
        s3 = self.enc_stage3(s2)     # 32x32,   48 canais
        b  = self.enc_stage4(s3)     # 16x16,  576 canais (Bottleneck)

        # --- DECODER COM ATTENTION GATES ---
        # Estágio 1 (32x32)
        u1 = self.up1(b)
        s3_attn = self.ag1(g=u1, x=s3)
        u1_cat = torch.cat([u1, s3_attn], dim=1)
        c1 = self.conv1(u1_cat)

        # Estágio 2 (64x64)
        u2 = self.up2(c1)
        s2_attn = self.ag2(g=u2, x=s2)
        u2_cat = torch.cat([u2, s2_attn], dim=1)
        c2 = self.conv2(u2_cat)

        # Estágio 3 (128x128)
        u3 = self.up3(c2)
        s1_attn = self.ag3(g=u3, x=s1)
        u3_cat = torch.cat([u3, s1_attn], dim=1)
        c3 = self.conv3(u3_cat)

        # Estágio 4 (256x256)
        u4 = self.up4(c3)
        s0_attn = self.ag4(g=u4, x=s0)
        u4_cat = torch.cat([u4, s0_attn], dim=1)
        c4 = self.conv4(u4_cat)

        # Saída Final (512x512)
        u_final = self.up_final(c4)
        logits = self.out_conv(u_final)
        
        return logits
