import torch
import torch.nn as nn
import torch.nn.functional as F

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


class OverlapPatchEmbed(nn.Module):
	"""Convolutional overlapping patch embedding."""

	def __init__(
		self,
		in_channels=4,
		embed_dim=32,
		patch_size=7,
		stride=4,
	):
		super().__init__()
		self.proj = nn.Conv2d(
			in_channels,
			embed_dim,
			kernel_size=patch_size,
			stride=stride,
			padding=patch_size // 2,
		)
		self.norm = nn.BatchNorm2d(embed_dim)

	def forward(self, x):
		return self.norm(self.proj(x))


class MixFFN(nn.Module):
	"""Mix-FFN block using pointwise and depthwise convolutions."""

	def __init__(self, in_features, hidden_features=None):
		super().__init__()
		hidden_features = hidden_features or in_features * 4

		self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
		self.dwconv = nn.Conv2d(
			hidden_features,
			hidden_features,
			kernel_size=3,
			padding=1,
			groups=hidden_features,
		)
		self.act = nn.GELU()
		self.fc2 = nn.Conv2d(hidden_features, in_features, kernel_size=1)

	def forward(self, x):
		x = self.fc1(x)
		x = self.dwconv(x)
		x = self.act(x)
		return self.fc2(x)

class SegFormerB0_Attention(nn.Module):
    """
    Híbrido SegFormer-B0 + Attention Gates.
    Une o All-MLP Decoder do SegFormer com a filtragem espacial (AG) do Attention U-Net.
    """
    def __init__(
        self,
        in_channels=4,
        out_channels=1,
        embed_dims=(32, 64, 160, 256),
        decoder_dim=256,
    ):
        super().__init__()

        if len(embed_dims) != 4:
            raise ValueError("embed_dims must contain four channel dimensions")

        # --- ENCODER (Hierarchical Transformer/CNN Mix) ---
        self.patch_embed1 = OverlapPatchEmbed(in_channels, embed_dims[0], patch_size=7, stride=4)
        self.stage1 = MixFFN(embed_dims[0])

        self.patch_embed2 = OverlapPatchEmbed(embed_dims[0], embed_dims[1], patch_size=3, stride=2)
        self.stage2 = MixFFN(embed_dims[1])

        self.patch_embed3 = OverlapPatchEmbed(embed_dims[1], embed_dims[2], patch_size=3, stride=2)
        self.stage3 = MixFFN(embed_dims[2])

        self.patch_embed4 = OverlapPatchEmbed(embed_dims[2], embed_dims[3], patch_size=3, stride=2)
        self.stage4 = MixFFN(embed_dims[3])

        # --- DECODER PROJECTIONS ---
        # Projetamos todas as saídas do encoder para a mesma dimensão (decoder_dim)
        self.linear1 = nn.Conv2d(embed_dims[0], decoder_dim, kernel_size=1)
        self.linear2 = nn.Conv2d(embed_dims[1], decoder_dim, kernel_size=1)
        self.linear3 = nn.Conv2d(embed_dims[2], decoder_dim, kernel_size=1)
        self.linear4 = nn.Conv2d(embed_dims[3], decoder_dim, kernel_size=1)

        # --- ATTENTION GATES ---
        # Como todos os mapas foram projetados para 'decoder_dim', F_g e F_l são iguais.
        # F_int é metade para reduzir custo computacional (prática comum).
        F_int = decoder_dim // 2
        
        self.ag3 = AttentionGate(F_g=decoder_dim, F_l=decoder_dim, F_int=F_int)
        self.ag2 = AttentionGate(F_g=decoder_dim, F_l=decoder_dim, F_int=F_int)
        self.ag1 = AttentionGate(F_g=decoder_dim, F_l=decoder_dim, F_int=F_int)

        # --- SEGFORMER FUSION HEAD ---
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(decoder_dim, out_channels, kernel_size=1)

    def forward(self, x):
        height, width = x.shape[2:]

        # 1. Extração de Características (Encoder)
        x1 = self.stage1(self.patch_embed1(x))   # Resolução: 1/4
        x2 = self.stage2(self.patch_embed2(x1))  # Resolução: 1/8
        x3 = self.stage3(self.patch_embed3(x2))  # Resolução: 1/16
        x4 = self.stage4(self.patch_embed4(x3))  # Resolução: 1/32

        # 2. Projeção Unificada (C1, C2, C3, C4)
        c1 = self.linear1(x1)
        c2 = self.linear2(x2)
        c3 = self.linear3(x3)
        c4 = self.linear4(x4)

        # 3. Cascata Top-Down com Attention Gates
        # c4 é o mais profundo. Interpolar para o tamanho de c3 e usar como "Sinal de Controle" (g)
        c4_up_to_3 = F.interpolate(c4, size=c3.shape[2:], mode="bilinear", align_corners=True)
        c3_attn = self.ag3(g=c4_up_to_3, x=c3)

        # Usar o c3_attn como controle para filtrar c2
        c3_up_to_2 = F.interpolate(c3_attn, size=c2.shape[2:], mode="bilinear", align_corners=True)
        c2_attn = self.ag2(g=c3_up_to_2, x=c2)

        # Usar o c2_attn como controle para filtrar c1
        c2_up_to_1 = F.interpolate(c2_attn, size=c1.shape[2:], mode="bilinear", align_corners=True)
        c1_attn = self.ag1(g=c2_up_to_1, x=c1)

        # 4. Fusão SegFormer Tradicional
        # Agora que c1, c2 e c3 foram filtrados, trazemos TODOS para a resolução de c1 (1/4)
        target_size = c1.shape[2:]
        
        c4_final = F.interpolate(c4, size=target_size, mode="bilinear", align_corners=True)
        c3_final = F.interpolate(c3_attn, size=target_size, mode="bilinear", align_corners=True)
        c2_final = F.interpolate(c2_attn, size=target_size, mode="bilinear", align_corners=True)
        c1_final = c1_attn # já está no target_size

        # Concatena e funde as 4 escalas limpas de ruído
        fused = self.linear_fuse(torch.cat([c1_final, c2_final, c3_final, c4_final], dim=1))
        logits = self.out_conv(fused)
        
        # 5. Upsampling Final para a resolução original da imagem
        return F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=True)