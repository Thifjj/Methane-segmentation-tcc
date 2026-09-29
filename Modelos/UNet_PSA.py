import torch
import torch.nn as nn


class DoubleConv(nn.Module):
	"""Two consecutive convolutional blocks."""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		return self.block(x)


class PSA_Module(nn.Module):
	"""Pyramid Squeeze Attention Module (Chen et al., 2025 - MPSUNet)."""

	def __init__(self, in_channels, reduction=4):
		super().__init__()

		self.c1 = in_channels // 4
		self.c2 = in_channels // 4
		self.c3 = in_channels // 4
		self.c4 = in_channels - (self.c1 + self.c2 + self.c3)

		self.conv1 = nn.Conv2d(
			self.c1, self.c1, kernel_size=3, padding=1, groups=1, bias=False
		)
		self.conv2 = nn.Conv2d(
			self.c2, self.c2, kernel_size=5, padding=2, groups=2, bias=False
		)
		self.conv3 = nn.Conv2d(
			self.c3, self.c3, kernel_size=7, padding=3, groups=4, bias=False
		)
		self.conv4 = nn.Conv2d(
			self.c4, self.c4, kernel_size=9, padding=4, groups=8, bias=False
		)

		self.bn = nn.BatchNorm2d(in_channels)
		self.relu = nn.ReLU(inplace=True)
		self.gap = nn.AdaptiveAvgPool2d(1)

		hidden_dim = max(8, in_channels // reduction)
		self.se = nn.Sequential(
			nn.Linear(in_channels, hidden_dim, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(hidden_dim, in_channels, bias=False),
			nn.Sigmoid(),
		)

	def forward(self, x):
		b, c, _, _ = x.shape
		x1, x2, x3, x4 = torch.split(
			x, [self.c1, self.c2, self.c3, self.c4], dim=1
		)

		f1 = self.conv1(x1)
		f2 = self.conv2(x2)
		f3 = self.conv3(x3)
		f4 = self.conv4(x4)

		feat = self.relu(self.bn(torch.cat([f1, f2, f3, f4], dim=1)))
		attn = self.se(self.gap(feat).view(b, c)).view(b, c, 1, 1)
		return feat * attn


class UNetPSA(nn.Module):
	"""U-Net with PSA attention in the skip connections."""

	def __init__(self, in_channels=4, out_channels=1):
		super().__init__()

		self.enc1 = DoubleConv(in_channels, 64)
		self.pool1 = nn.MaxPool2d(2)
		self.psa1 = PSA_Module(64)

		self.enc2 = DoubleConv(64, 128)
		self.pool2 = nn.MaxPool2d(2)
		self.psa2 = PSA_Module(128)

		self.enc3 = DoubleConv(128, 256)
		self.pool3 = nn.MaxPool2d(2)
		self.psa3 = PSA_Module(256)

		self.enc4 = DoubleConv(256, 512)
		self.pool4 = nn.MaxPool2d(2)
		self.psa4 = PSA_Module(512)

		self.bottleneck = DoubleConv(512, 1024)

		self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
		self.dec1 = DoubleConv(1024, 512)
		self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
		self.dec2 = DoubleConv(512, 256)
		self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
		self.dec3 = DoubleConv(256, 128)
		self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
		self.dec4 = DoubleConv(128, 64)

		self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

	def forward(self, x):
		e1 = self.enc1(x)
		e2 = self.enc2(self.pool1(e1))
		e3 = self.enc3(self.pool2(e2))
		e4 = self.enc4(self.pool3(e3))
		b = self.bottleneck(self.pool4(e4))

		d1 = self.dec1(torch.cat([self.up1(b), self.psa4(e4)], dim=1))
		d2 = self.dec2(torch.cat([self.up2(d1), self.psa3(e3)], dim=1))
		d3 = self.dec3(torch.cat([self.up3(d2), self.psa2(e2)], dim=1))
		d4 = self.dec4(torch.cat([self.up4(d3), self.psa1(e1)], dim=1))

		return self.out_conv(d4)
