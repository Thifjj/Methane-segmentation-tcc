"""Executa sequencialmente as duas perdas usando o loop original de Treinamento_Unet."""

from Modelos.UNet_MobileNetV3_AttentionGates_DPU import UNetMobileNetV3AttentionGatesDPU
from Treinos.Treinamento_Unet import treinar_modelo


PRODUCTS = ["mag1c", "TOA_AVIRIS_460nm", "TOA_AVIRIS_550nm", "TOA_AVIRIS_640nm"]
OUTPUT = "remaining_all_dpu_retrain_original"


def main():
    for loss_name in ("BCEloss", "FocalDiceLoss"):
        name = f"{OUTPUT}/UnetMobilenetV3AttentionGates_dpu_{loss_name}_mag1c_rgb"
        treinar_modelo(UNetMobileNetV3AttentionGatesDPU, name, 0.0, PRODUCTS,
                      loss_name=loss_name)


if __name__ == "__main__":
    main()
