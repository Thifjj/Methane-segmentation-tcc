"""Every model uses the same reproducible PTQ calibration profile."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantize_model import parse_args


class CalibrationConfigTests(unittest.TestCase):
    def arguments(self, model, *extra):
        return parse_args([
            "--model", model,
            "--quant-mode", "calib",
            "--csv", "/dataset/train.csv",
            "--data-root", "/dataset",
            *extra,
        ])

    def test_every_model_uses_shared_profile(self):
        for model in ("attentiongates_dpu_only_remaining", "mobilenet_v3_dpu_512", "hyperstarcop"):
            args = self.arguments(model)
            self.assertEqual((args.subset_len, args.range_samples, args.refine_layers), (100, 512, 0))

    def test_different_calibration_profile_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.arguments("mobilenet_v3_dpu_512", "--subset-len", "300")

    def test_deploy_allows_single_export_sample(self):
        args = parse_args([
            "--model", "mobilenet_v3_dpu_512",
            "--quant-mode", "test", "--deploy",
            "--csv", "/dataset/train.csv",
            "--data-root", "/dataset",
            "--subset-len", "1",
        ])
        self.assertEqual((args.subset_len, args.range_samples, args.refine_layers), (1, 512, 0))


if __name__ == "__main__":
    unittest.main()
