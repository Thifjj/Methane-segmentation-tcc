import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) * 2 - Mantida idêntica ao seu baseline"""
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

class UNetDepthReduced(nn.Module):
    """
    U-Net Otimizada: Redução de Camadas (Inspirada em Kang et al. e Neiso et al.)
    - Redução de 4 blocos de downsampling para 2 blocos.
    - O bottleneck passa a operar em 256 canais (em vez de 1024).
    - Preserva o uso de DoubleConv, ConvTranspose2d e Skip Connections por Concatenação.
    Otimização sugerida no artigo de Neiso et al., 2024
    """
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()
        
        # Caminho de Descida (Contração) - Reduzido para 2 estágios
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        # camadas podadas
        
        # self.down3 = DoubleConv(128, 256)
        # self.pool3 = nn.MaxPool2d(2)
        # self.down4 = DoubleConv(256, 512)
        # self.pool4 = nn.MaxPool2d(2)
        
        # O bottleneck agora ocorre diretamente após o segundo nível de pooling
        # Modificado tamanho de entrada 128 -> saida 256
        #self.bottleneck = DoubleConv(512, 1024)
        self.bottleneck = DoubleConv(128, 256)

        # Caminho de Subida (Expansão) - Reduzido para 2 estágios
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128) # 128 (up1) + 128 (skip connection de down2)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)  # 64 (up2) + 64 (skip connection de down1)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Caminho de descida (salvando skip connections)
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))

        # Bottleneck intermediário
        b = self.bottleneck(self.pool2(x2))

        # Caminho de subida (concatenando as skip connections correspondentes)
        u1 = self.up1(b)
        u1 = torch.cat([u1, x2], dim=1) # Concatena com a saída do estágio 2
        c1 = self.conv1(u1)

        u2 = self.up2(c1)
        u2 = torch.cat([u2, x1], dim=1) # Concatena com a saída do estágio 1
        c2 = self.conv2(u2)

        logits = self.out_conv(c2)
        return logits