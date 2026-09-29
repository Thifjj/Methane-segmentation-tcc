import torch
import torch.nn as nn
import torchvision.models as models


class DoubleConv(nn.Module):
	"""Ruzicka et. al. 2020"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.double_conv = nn.Sequential(
			nn.Conv2d(
				in_channels,
				out_channels,
				kernel_size=3,
				padding=1,
				bias=False,
			),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(
				out_channels,
				out_channels,
				kernel_size=3,
				padding=1,
				bias=False,
			),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		return self.double_conv(x)


class UNetResNet34(nn.Module):
	"""U-Net with a ResNet-34 backbone."""

	def __init__(self, in_channels=4, out_channels=1):
		super().__init__()

		resnet = models.resnet34(weights=None)

		self.conv1 = nn.Conv2d(
			in_channels,
			64,
			kernel_size=7,
			stride=2,
			padding=3,
			bias=False,
		)
		self.bn1 = resnet.bn1
		self.relu = resnet.relu
		self.maxpool = resnet.maxpool

		self.encoder1 = resnet.layer1  # 64 channels, resolution 1/4
		self.encoder2 = resnet.layer2  # 128 channels, resolution 1/8
		self.encoder3 = resnet.layer3  # 256 channels, resolution 1/16
		self.encoder4 = resnet.layer4  # 512 channels, resolution 1/32

		self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
		self.dec_conv1 = DoubleConv(256 + 256, 256)

		self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
		self.dec_conv2 = DoubleConv(128 + 128, 128)

		self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
		self.dec_conv3 = DoubleConv(64 + 64, 64)

		self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
		self.dec_conv4 = DoubleConv(64 + 64, 32)

		self.up_final = nn.Upsample(
			scale_factor=2,
			mode="bilinear",
			align_corners=True,
		)
		self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

	def forward(self, x):
		x0 = self.relu(self.bn1(self.conv1(x)))
		x_pool = self.maxpool(x0)

		e1 = self.encoder1(x_pool)
		e2 = self.encoder2(e1)
		e3 = self.encoder3(e2)
		e4 = self.encoder4(e3)

		d1 = self.dec_conv1(torch.cat([self.up1(e4), e3], dim=1))
		d2 = self.dec_conv2(torch.cat([self.up2(d1), e2], dim=1))
		d3 = self.dec_conv3(torch.cat([self.up3(d2), e1], dim=1))
		d4 = self.dec_conv4(torch.cat([self.up4(d3), x0], dim=1))

		return self.out_conv(self.up_final(d4))
