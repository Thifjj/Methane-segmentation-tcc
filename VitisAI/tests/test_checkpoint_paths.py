"""Filename migration checks; no training, inference or quantization."""
import hashlib
import tempfile
import unittest
from pathlib import Path

from Modelos_treinados.checkpoint_paths import (
    CHECKPOINTS, ROOT, resolve_checkpoint_path, training_checkpoint_path,
)
from VitisAI.common import build_model, resolve_model


class CheckpointPathTests(unittest.TestCase):
    def test_inventory_paths_hashes_and_lfs_identity(self):
        self.assertEqual(len({row["new"] for row in CHECKPOINTS}), len(CHECKPOINTS))
        for row in CHECKPOINTS:
            with self.subTest(path=row["new"]):
                path = ROOT.parent / row["new"]
                if not path.is_file():
                    self.fail(f"Missing registered checkpoint: {path}")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(digest, row["sha256"])
                if row["lfs_oid"]:
                    self.assertEqual(digest, row["lfs_oid"])
                self.assertEqual(resolve_checkpoint_path(ROOT.parent / row["old"]).resolve(), path.resolve())
                self.assertEqual(resolve_model(checkpoint=path)[1], row["architecture"])

    def test_custom_file_is_never_redirected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "UNET_mag1c_rgb.pth"
            self.assertEqual(resolve_checkpoint_path(path), path)
            path.touch()
            self.assertEqual(resolve_checkpoint_path(path), path)

    def test_historical_dpu_paths_keep_class_and_weights(self):
        name, architecture = resolve_model(checkpoint=ROOT / "Mobile_Net_v3_dpu_mag1c_rgb.pth")
        self.assertEqual((name, architecture), ("mobilenet_v3_dpu", "UNetMobileNetV3_dpu"))
        model, path = build_model(name, ROOT / "Mobile_Net_v3_dpu_mag1c_rgb.pth", 4)
        self.assertFalse(model.up_final.align_corners)
        self.assertEqual(path.name, "unet_mobilenet_v3_bce_mag1c_rgb.pth")
        self.assertEqual(resolve_checkpoint_path(ROOT / "UnetMobilenetV3AttentionGates_dpu_only_remaining_mag1c_rgb.pth").name,
                         "unet_mobilenet_v3_attentiongates_dpu_only_remaining_val_test_mag1c_rgb.pth")

    def test_distinct_deleted_experiment_is_not_substituted(self):
        path = ROOT / "UnetMobilenetV3AttentionGates_dpu_FocalDiceLossmag1c_rgb.pth"
        self.assertEqual(resolve_checkpoint_path(path), path)
        self.assertFalse(path.exists())

    def test_training_stems_and_legacy_resume(self):
        self.assertEqual(training_checkpoint_path("UNET_mag1c_rgb").name, "unet_baseline_mag1c_rgb.pth")
        self.assertEqual(training_checkpoint_path("unet_baseline_bce_new_mag1c_rgb").name,
                         "unet_baseline_bce_new_mag1c_rgb.pth")
        for stem in ("UnknownModel", "../outside", "bad-name", "two__underscores"):
            with self.subTest(stem=stem), self.assertRaises(ValueError):
                training_checkpoint_path(stem)
