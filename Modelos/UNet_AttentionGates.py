import torch
import torch.nn as nn


class DoubleConv(nn.Module):
	"""Two consecutive convolutional blocks."""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		return self.double_conv(x)


class AttentionGate(nn.Module):
	"""Attention Gate (Ahsan et al., 2025 - AttMetNet)."""

	def __init__(self, F_g, F_l, F_int):
		super().__init__()

		self.W_g = nn.Sequential(
			nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
			nn.BatchNorm2d(F_int),
		)
		self.W_x = nn.Sequential(
			nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
			nn.BatchNorm2d(F_int),
		)
		self.psi = nn.Sequential(
			nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
			nn.BatchNorm2d(1),
			nn.Sigmoid(),
		)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, g, x):
		g1 = self.W_g(g)
		x1 = self.W_x(x)
		net = self.relu(g1 + x1)
		attn = self.psi(net)
		return x * attn


class UNetAttentionGates(nn.Module):
	"""Attention U-Net (AttMetNet) para segmentação de metano."""

	def __init__(self, in_channels=4, out_channels=1):
		super().__init__()

		self.enc1 = DoubleConv(in_channels, 64)
		self.pool1 = nn.MaxPool2d(2)
		self.enc2 = DoubleConv(64, 128)
		self.pool2 = nn.MaxPool2d(2)
		self.enc3 = DoubleConv(128, 256)
		self.pool3 = nn.MaxPool2d(2)
		self.enc4 = DoubleConv(256, 512)
		self.pool4 = nn.MaxPool2d(2)

		self.bottleneck = DoubleConv(512, 1024)

		self.ag1 = AttentionGate(F_g=512, F_l=512, F_int=256)
		self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
		self.dec1 = DoubleConv(1024, 512)

		self.ag2 = AttentionGate(F_g=256, F_l=256, F_int=128)
		self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
		self.dec2 = DoubleConv(512, 256)

		self.ag3 = AttentionGate(F_g=128, F_l=128, F_int=64)
		self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
		self.dec3 = DoubleConv(256, 128)

		self.ag4 = AttentionGate(F_g=64, F_l=64, F_int=32)
		self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
		self.dec4 = DoubleConv(128, 64)

		self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

	def forward(self, x):
		e1 = self.enc1(x)
		e2 = self.enc2(self.pool1(e1))
		e3 = self.enc3(self.pool2(e2))
		e4 = self.enc4(self.pool3(e3))
		b = self.bottleneck(self.pool4(e4))

		u1 = self.up1(b)
		s4 = self.ag1(g=u1, x=e4)
		d1 = self.dec1(torch.cat([u1, s4], dim=1))

		u2 = self.up2(d1)
		s3 = self.ag2(g=u2, x=e3)
		d2 = self.dec2(torch.cat([u2, s3], dim=1))

		u3 = self.up3(d2)
		s2 = self.ag3(g=u3, x=e2)
		d3 = self.dec3(torch.cat([u3, s2], dim=1))

		u4 = self.up4(d3)
		s1 = self.ag4(g=u4, x=e1)
		d4 = self.dec4(torch.cat([u4, s1], dim=1))

		return self.out_conv(d4)
