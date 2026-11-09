"""Filename-only migration; custom files and existing legacy files take precedence."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = json.loads((ROOT / "checkpoints.json").read_text())
RENAMES = {str(Path(row["old"]).relative_to("Modelos_treinados")):
           str(Path(row["new"]).relative_to("Modelos_treinados")) for row in CHECKPOINTS}
ARCHITECTURES = {Path(row["new"]).name: row["architecture"] for row in CHECKPOINTS}

# Git renames and calibration hashes prove these historical paths identify the same objects.
RENAMES.update({
    "Mobile_Net_v3_mag1c_rgb.pth": "unet_mobilenet_v3_bce_mag1c_rgb.pth",
    "Mobile_Net_v3_dpu_mag1c_rgb.pth": "unet_mobilenet_v3_bce_mag1c_rgb.pth",
    "Mobile_Net_v3_mag1c_rgb_dpu.pth": "unet_mobilenet_v3_bce_mag1c_rgb.pth",
    "MobileNetV3_AttentionGates_mag1c_rgb.pth": "unet_mobilenet_v3_attentiongates_focaldice_mag1c_rgb.pth",
    "UnetMobilenetV3AttentionGates_dpu_only_remaining_mag1c_rgb.pth": "unet_mobilenet_v3_attentiongates_dpu_only_remaining_val_test_mag1c_rgb.pth",
    "remaining_all_dpu_retrain_original/UnetMobilenetV3AttentionGates_dpu_BCEloss_mag1c_rgb.pth": "unet_mobilenet_v3_attentiongates_dpu_bce_remaining_all_mag1c_rgb.pth",
    "remaining_all_dpu_retrain_original/UnetMobilenetV3AttentionGates_dpu_FocalDiceLoss_mag1c_rgb.pth": "unet_mobilenet_v3_attentiongates_dpu_focaldice_remaining_all_mag1c_rgb.pth",
})


def resolve_checkpoint_path(path):
    path = Path(path)
    if path.exists():
        return path
    try:
        relative = str(path.resolve().relative_to(ROOT))
    except ValueError:
        return path
    return ROOT / RENAMES[relative] if relative in RENAMES else path


def training_checkpoint_path(stem):
    path = resolve_checkpoint_path(ROOT / f"{stem}.pth")
    if not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", path.stem) or path.parent != ROOT:
        raise ValueError("Use um nome lowercase snake_case: <architecture>_<loss>_<variant>_<input>.")
    return path
