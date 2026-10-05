"""Regressoes do carregamento por checkpoint e do contrato de calibracao."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (DEFAULT_PRODUCTS, build_model, calibration_contract,
                    file_sha256, resolve_model, validate_calibration)
from Modelos import UNetMobileNetV3


class ModelLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.checkpoint = cls.root / 'Mobile_Net_v3_dpu_mag1c_rgb.pth'
        torch.save(UNetMobileNetV3().state_dict(), cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_path_selects_dpu_architecture(self):
        name, architecture = resolve_model(checkpoint=self.checkpoint)
        model, _ = build_model(name, self.checkpoint, 4, architecture)
        self.assertEqual(architecture, 'UNetMobileNetV3_dpu')
        self.assertTrue(all(layer.align_corners is False for layer in
                            (model.up1, model.up2, model.up3, model.up4, model.up_final)))

    def test_custom_filename_requires_explicit_architecture(self):
        with self.assertRaisesRegex(ValueError, 'architecture'):
            resolve_model(checkpoint=self.root / 'custom.pth')
        self.assertEqual(resolve_model(checkpoint=self.root / 'custom.pth',
                                       architecture='UNetMobileNetV3_dpu'),
                         ('custom', 'UNetMobileNetV3_dpu'))
        model, _ = build_model('custom', self.checkpoint, 4, 'UNetMobileNetV3_dpu')
        self.assertFalse(model.up_final.align_corners)

    def test_incompatible_channels_fail_strict_loading(self):
        with self.assertRaises(RuntimeError):
            build_model('mobilenet_v3_dpu', self.checkpoint, 3)

    def test_manifest_rejects_other_architecture_with_same_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'quant_info.json'
            config.write_text('{}')
            arguments = ('mobilenet_v3_dpu', self.checkpoint, DEFAULT_PRODUCTS,
                         128, 128, 'DPUCZDX8G_ISA1_B4096', True)
            manifest = calibration_contract(*arguments)
            manifest['artifacts'] = {'quant_info.json': file_sha256(config)}
            path = root / 'calibration_manifest.json'
            path.write_text(json.dumps(manifest))
            validate_calibration(root, *arguments)
            with self.assertRaisesRegex(ValueError, 'architecture'):
                validate_calibration(root, *arguments, architecture='UNetMobileNetV3')
            # Os manifestos oficiais antigos continuam validos.
            manifest.pop('architecture')
            path.write_text(json.dumps(manifest))
            validate_calibration(root, *arguments)


if __name__ == '__main__':
    unittest.main()
