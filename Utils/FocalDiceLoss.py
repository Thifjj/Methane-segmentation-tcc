import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalDiceLoss(nn.Module):
	"""Combina as perdas Focal e Dice para segmentação binária."""

	def __init__(
		self,
		alpha=0.25,
		gamma=2.0,
		weight_focal=1.0,
		weight_dice=1.0,
		smooth=1e-6,
	):
		super().__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.weight_focal = weight_focal
		self.weight_dice = weight_dice
		self.smooth = smooth

	def forward(self, logits, targets, weight_map=None):
		"""Calcula a perda a partir de logits e máscaras binárias.

		Args:
			logits: Tensor de formato [B, 1, H, W], sem aplicação de sigmoide.
			targets: Máscara binária de formato [B, 1, H, W].
			weight_map: Mapa opcional de pesos, com o mesmo formato.
		"""
		probs = torch.sigmoid(logits)

		bce_loss = F.binary_cross_entropy_with_logits(
			logits, targets, reduction="none"
		)
		p_t = probs * targets + (1 - probs) * (1 - targets)
		focal_weight = (1 - p_t) ** self.gamma
		alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
		focal_loss = alpha_factor * focal_weight * bce_loss

		if weight_map is not None:
			focal_loss = focal_loss * weight_map
		focal_loss = focal_loss.mean()

		probs_flat = probs.reshape(-1)
		targets_flat = targets.reshape(-1)
		intersection = (probs_flat * targets_flat).sum()
		dice_score = (2 * intersection + self.smooth) / (
			probs_flat.sum() + targets_flat.sum() + self.smooth
		)
		dice_loss = 1 - dice_score

		return self.weight_focal * focal_loss + self.weight_dice * dice_loss