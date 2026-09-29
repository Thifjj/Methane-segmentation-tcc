import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SegFormerB0(nn.Module):
	"""SegFormer-B0-style Araújo &amp; Zortea (2025)."""

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

		self.patch_embed1 = OverlapPatchEmbed(
			in_channels, embed_dims[0], patch_size=7, stride=4
		)
		self.stage1 = MixFFN(embed_dims[0])

		self.patch_embed2 = OverlapPatchEmbed(
			embed_dims[0], embed_dims[1], patch_size=3, stride=2
		)
		self.stage2 = MixFFN(embed_dims[1])

		self.patch_embed3 = OverlapPatchEmbed(
			embed_dims[1], embed_dims[2], patch_size=3, stride=2
		)
		self.stage3 = MixFFN(embed_dims[2])

		self.patch_embed4 = OverlapPatchEmbed(
			embed_dims[2], embed_dims[3], patch_size=3, stride=2
		)
		self.stage4 = MixFFN(embed_dims[3])

		self.linear1 = nn.Conv2d(embed_dims[0], decoder_dim, kernel_size=1)
		self.linear2 = nn.Conv2d(embed_dims[1], decoder_dim, kernel_size=1)
		self.linear3 = nn.Conv2d(embed_dims[2], decoder_dim, kernel_size=1)
		self.linear4 = nn.Conv2d(embed_dims[3], decoder_dim, kernel_size=1)

		self.linear_fuse = nn.Sequential(
			nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1, bias=False),
			nn.BatchNorm2d(decoder_dim),
			nn.ReLU(inplace=True),
		)
		self.out_conv = nn.Conv2d(decoder_dim, out_channels, kernel_size=1)

	def forward(self, x):
		height, width = x.shape[2:]

		x1 = self.stage1(self.patch_embed1(x))
		x2 = self.stage2(self.patch_embed2(x1))
		x3 = self.stage3(self.patch_embed3(x2))
		x4 = self.stage4(self.patch_embed4(x3))

		target_size = x1.shape[2:]
		c1 = self.linear1(x1)
		c2 = F.interpolate(
			self.linear2(x2), size=target_size, mode="bilinear", align_corners=False
		)
		c3 = F.interpolate(
			self.linear3(x3), size=target_size, mode="bilinear", align_corners=False
		)
		c4 = F.interpolate(
			self.linear4(x4), size=target_size, mode="bilinear", align_corners=False
		)

		fused = self.linear_fuse(torch.cat([c1, c2, c3, c4], dim=1))
		logits = self.out_conv(fused)
		return F.interpolate(
			logits, size=(height, width), mode="bilinear", align_corners=False
		)
