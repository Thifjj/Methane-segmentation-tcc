"""Configuracao de resolucao completa e prioridade dos argumentos CLI."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantize_model import parse_args


class CalibrationConfigTests(unittest.TestCase):
    def arguments(self):
        config = Path(__file__).resolve().parents[1] / 'configs/mobilenet_v3_dpu_512.json'
        return ['--config', str(config), '--quant-mode', 'calib',
                '--csv', '/dataset/train.csv', '--data-root', '/dataset']

    def test_full_image_profile(self):
        args = parse_args(self.arguments())
        self.assertEqual((args.height, args.width), (512, 512))
        self.assertFalse(args.patching)
        self.assertFalse(args.balanced)
        self.assertEqual(args.architecture, 'UNetMobileNetV3_dpu')
        self.assertEqual(args.subset_len, 300)
        self.assertEqual(args.model, 'mobilenet_v3_dpu_512')
        self.assertEqual(args.output_dir, 'build/vitis_ai/quantize')

    def test_cli_overrides_profile(self):
        args = parse_args(self.arguments() + ['--subset-len', '600', '--refine-layers', '24'])
        self.assertEqual(args.subset_len, 600)
        self.assertEqual(args.refine_layers, 24)


if __name__ == '__main__':
    unittest.main()
