import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) * 2 - Mantida para consistência"""
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

class UNetElementWise(nn.Module):
    """
    U-Net Otimizada: Skip Connections por Soma Elemento a Elemento
    (Inspirada na arquitetura LinkNet descrita em Herec et al., 2025)
    
    - Encoder & Decoder: Profundidade reduzida (2 estágios).
    - Skip Connections: Soma elemento a elemento (+) em vez de concatenação (torch.cat).
    - Eficiência: Mantém o número de canais estável no decoder, reduzindo os FLOPs pela metade.
    """
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()
        
        # Caminho de Descida (Encoder)
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)

        # Caminho de Subida (Decoder) - Ajustado para soma elemento a elemento
        # Projetamos a saída da transposição para casar exatamente com os canais do encoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # Como somamos (128 + 128 = 128 canais), o conv1 recebe apenas 128 canais (U-Net clássica receberia 256)
        self.conv1 = DoubleConv(128, 128) 
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # Como somamos (64 + 64 = 64 canais), o conv2 recebe apenas 64 canais (U-Net clássica receberia 128)
        self.conv2 = DoubleConv(64, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # --- ENCODER (Salvando tensores de salto) ---
        x1 = self.down1(x)         # Resolução: 512x512, 64 canais
        x2 = self.down2(self.pool1(x1)) # Resolução: 256x256, 128 canais

        # Bottleneck
        b = self.bottleneck(self.pool2(x2)) # Resolução: 128x128, 256 canais

        # --- DECODER (Fusão por Soma Elemento a Elemento) ---
        u1 = self.up1(b)           # Upsample para 256x256, 128 canais
        
        # OTIMIZAÇÃO: Soma direta no lugar de torch.cat([u1, x2], dim=1)
        u1 = u1 + x2               
        c1 = self.conv1(u1)        # Processa 128 canais

        u2 = self.up2(c1)          # Upsample para 512x512, 64 canais
        
        # OTIMIZAÇÃO: Soma direta no lugar de torch.cat([u2, x1], dim=1)
        u2 = u2 + x1               
        c2 = self.conv2(u2)        # Processa 64 canais

        logits = self.out_conv(c2)
        return logits