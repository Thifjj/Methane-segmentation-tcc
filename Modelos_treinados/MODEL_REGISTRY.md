# Trained checkpoint registry

Prepared before filename migration. Paths below are relative to `Modelos_treinados/`.

Convention: `<architecture>_<loss>_<variant>_<input>.pth`; omit unconfirmed fields.
`old` denotes only the preserved old filename, not an inferred training run.
HyperSTARCOP retains its official Lightning filename and contents.

| Old path | New path | Exported class |
|---|---|---|
| `HyperSTARCOP_oficial/final_checkpoint_model.ckpt` | `HyperSTARCOP_oficial/final_checkpoint_model.ckpt` | `HyperSTARCOPOficial` |
| `MobileNetV3_AttentionGates_BCE_mag1c_rgb.pth` | `unet_mobilenet_v3_attentiongates_bce_mag1c_rgb.pth` | `UNetMobileNetV3AttentionGates` |
| `MobileNetV3_AttentionGates_BCE_mag1c_rgb_old.pth` | `unet_mobilenet_v3_attentiongates_bce_old_mag1c_rgb.pth` | `UNetMobileNetV3AttentionGates` |
| `MobileNetV3_AttentionGates_FocalDiceLossmag1c_rgb.pth` | `unet_mobilenet_v3_attentiongates_focaldice_mag1c_rgb.pth` | `UNetMobileNetV3AttentionGates` |
| `MobileNet_v3_FocalDiceLoss_mag1c_rgb.pth` | `unet_mobilenet_v3_focaldice_mag1c_rgb.pth` | `UNetMobileNetV3` |
| `Mobile_Net_v2_mag1c_rgb.pth` | `unet_mobilenet_v2_mag1c_rgb.pth` | `UNetMobileNetV2` |
| `Mobile_Net_v3_BCELoss_mag1c_rgb.pth` | `unet_mobilenet_v3_bce_mag1c_rgb.pth` | `UNetMobileNetV3` |
| `Modelo_artigo/MobileNet_v3_FocalDiceLoss_mag1c_rgb.pth` | `Modelo_artigo/unet_mobilenet_v3_focaldice_mag1c_rgb.pth` | `UNetMobileNetV3` |
| `Modelo_artigo/Mobile_Net_v3_BCELoss_mag1c_rgb.pth` | `Modelo_artigo/unet_mobilenet_v3_bce_mag1c_rgb.pth` | `UNetMobileNetV3` |
| `Modelo_artigo/UNetMobileNetV3AttentionGatesDPU_BCE_mag1c_rgb.pth` | `Modelo_artigo/unet_mobilenet_v3_attentiongates_dpu_bce_retrained_mag1c_rgb.pth` | `UNetMobileNetV3AttentionGatesDPU` |
| `Modelo_artigo/UnetMobilenetV3AttentionGates_dpu_FocalDiceLoss_mag1c_rgb.pth` | `Modelo_artigo/unet_mobilenet_v3_attentiongates_dpu_focaldice_remaining_all_mag1c_rgb.pth` | `UNetMobileNetV3AttentionGatesDPU` |
| `UNET_SkipConnections_mag1c_rgb.pth` | `unet_skip_connections_mag1c_rgb.pth` | `UNetElementWise` |
| `UNET_depth_reduced_mag1c_rgb.pth` | `unet_depth_reduced_mag1c_rgb.pth` | `UNetDepthReduced` |
| `UNET_mag1c_rgb.pth` | `unet_baseline_mag1c_rgb.pth` | `UNetBaseline` |
| `UNetMobileNetV3AttentionGatesDPU_BCE_mag1c_rgb.pth` | `unet_mobilenet_v3_attentiongates_dpu_bce_retrained_mag1c_rgb.pth` | `UNetMobileNetV3AttentionGatesDPU` |
| `UNetMobileNetV3AttentionGates_mag1c_rgb.pth` | `unet_mobilenet_v3_attentiongates_focaldice_remaining_all_mag1c_rgb.pth` | `UNetMobileNetV3AttentionGates` |
| `UNet_ResNet34_mag1c_rgb.pth` | `unet_resnet34_mag1c_rgb.pth` | `UNetResNet34` |
| `UNet_SegFormer_AttGate_BCE_mag1c_rgb.pth` | `segformer_b0_attentiongates_bce_mag1c_rgb.pth` | `SegFormerB0_Attention` |
| `UNet_SegFormer_AttGate_FD_mag1c_rgb.pth` | `segformer_b0_attentiongates_focaldice_mag1c_rgb.pth` | `SegFormerB0_Attention` |
| `UNet_SegFormer_AttGate_mag1c_rgb.pth` | `segformer_b0_attentiongates_mag1c_rgb.pth` | `SegFormerB0_Attention` |
| `UNet_SegFormer_FD_mag1c_rgb.pth` | `segformer_b0_focaldice_mag1c_rgb.pth` | `SegFormerB0` |
| `UNet_SegFormer_mag1c_rgb.pth` | `segformer_b0_mag1c_rgb.pth` | `SegFormerB0` |
| `UnetMobilenetV3AttentionGates_dpu_BCEloss_mag1c_rgb.pth` | `unet_mobilenet_v3_attentiongates_dpu_bce_remaining_all_mag1c_rgb.pth` | `UNetMobileNetV3AttentionGatesDPU` |
| `UnetMobilenetV3AttentionGates_dpu_FocalDiceLoss_mag1c_rgb.pth` | `unet_mobilenet_v3_attentiongates_dpu_focaldice_remaining_all_mag1c_rgb.pth` | `UNetMobileNetV3AttentionGatesDPU` |
| `UnetMobilenetV3AttentionGates_dpu_only_remaining_val_test_mag1c_rgb.pth` | `unet_mobilenet_v3_attentiongates_dpu_only_remaining_val_test_mag1c_rgb.pth` | `UNetMobileNetV3AttentionGatesDPU` |

## Evidence and provenance

All 24 `.pth` files are bare tensor state dictionaries: they contain no loss, dataset, epoch, or optimizer metadata. Architecture identification uses exact state-dictionary key/shape matching followed by strict loading. Full file SHA-256, tensor SHA-256 (sorted name, dtype, shape, bytes), sizes, and original LFS object IDs are recorded in `checkpoints.json`. All tracked checkpoint hashes matched their HEAD LFS OIDs before migration. Four SegFormer checkpoints were already untracked local files; they remain distinct and retain LFS coverage via `.gitattributes`.

- `UNetBaseline`, `UNetDepthReduced`, `UNetElementWise`, `UNetMobileNetV2`, `UNetResNet34`, and `SegFormerB0` match their corresponding modules in `Modelos/`. Loss and exact training subset are unconfirmed for filenames without loss tokens. Inputs are four channels, named mag1c/RGB by the original files and callers.
- `SegFormerB0` is the actual exported SegFormer class, not a U-Net. Attention variants use `SegFormerB0_Attention` in `Modelos/UNet_SegFormer_AttentionGates.py`. BCE and FD labels come from supplied filenames; manifests independently bind these files to their classes and hashes. FD is expanded to focaldice following the project loss convention; no optimizer metadata survives. The unsuffixed checkpoints retain their own identities.
- Non-attention MobileNetV3 BCE/FocalDice weights fit both `UNetMobileNetV3` and `UNetMobileNetV3_dpu`. Their parameter keys/shapes are identical; `align_corners` is not serialized. Filenames therefore use the shared architecture family. Existing callers retain their selected standard or DPU class and interpolation. Vitis manifests confirm DPU use, while standard benchmark aliases use the standard class.
- Git `683585f` renamed the original MobileNetV3 weights to `Mobile_Net_v3_BCELoss_mag1c_rgb.pth` and the original attention weights to `MobileNetV3_AttentionGates_FocalDiceLossmag1c_rgb.pth`. Git `d832415` labels the latter training family FocalDice.
- `UNetMobileNetV3AttentionGates_mag1c_rgb.pth` was added in `e53e953`; that revision of `Treinos/Treinamento_Unet.py` uses FocalDiceLoss and STARCOP_train_remaining_all. It is distinct from the other FocalDice attention checkpoint.
- The DPU attention class has per-channel Hardsigmoid attention; the standard class has scalar Sigmoid attention. State shapes distinguish them. These classes are never interchanged.
- DPU BCE `remaining_all` hash `b56dae…` matches the historical `remaining_all_dpu_retrain_original/` file at `683585f`. DPU FocalDice `remaining_all` was moved intact from that directory at `f3c3dd7`. Exact split membership is not recoverable from weights.
- DPU BCE `retrained` (`66da84…`) is distinct from `remaining_all` (`b56dae…`); `benchmark_arm/README.md` and the `ptq_nativa_bce_comparacao` manifests identify the retrained checkpoint. Keep both.
- `only_remaining_val_test` preserves the original subset/provenance tokens. Git `683585f` renamed `only_remaining` without changing its LFS object, and existing calibration manifests confirm SHA-256 `9ea504…`. Loss is unconfirmed.
- Attention BCE `old` and current checkpoints differ in both file and tensor hashes. Git `6e1481f` introduced the old backup while updating the current file. No unsupported epoch or dataset labels are added.
- HyperSTARCOP uses `HyperSTARCOPOficial` / `carregar_hyperstarcop`; its official `config.yaml` records BCEWithLogitsLoss. The official checkpoint is unchanged.

## Identical checkpoints preserved

Each pair below has identical complete-file SHA-256 and tensor SHA-256; both directory copies remain:

- `unet_mobilenet_v3_focaldice_mag1c_rgb.pth` and `Modelo_artigo/unet_mobilenet_v3_focaldice_mag1c_rgb.pth`: `61938b4f555c9fd6d208c2288aff1807ad2e262ef0076e33e2e826db23c5b726`.
- `unet_mobilenet_v3_bce_mag1c_rgb.pth` and `Modelo_artigo/unet_mobilenet_v3_bce_mag1c_rgb.pth`: `e6c4c228f44a31f558075bbea96f74bd69b24a54a03ba8f5e92d063d7d051944`.
- `Modelo_artigo/unet_mobilenet_v3_attentiongates_dpu_bce_retrained_mag1c_rgb.pth` and `unet_mobilenet_v3_attentiongates_dpu_bce_retrained_mag1c_rgb.pth`: `66da8401cdae87d9f259bfa50c7c39de947761842534ce604010494d32df25ad`.
- `Modelo_artigo/unet_mobilenet_v3_attentiongates_dpu_focaldice_remaining_all_mag1c_rgb.pth` and `unet_mobilenet_v3_attentiongates_dpu_focaldice_remaining_all_mag1c_rgb.pth`: `bb2dd6299d44a80b8c0c9072b3bbd2625de3ad7c6a3acf22dca51e9dcd77667e`.

No other pair matches either file SHA-256 or tensor SHA-256. Similar names do not establish identity.

## Path compatibility and missing objects

`Modelos_treinados/checkpoint_paths.py` reads the authoritative migration inventory. It redirects missing old paths inside this checkpoint tree only; existing files and external custom checkpoint paths retain precedence. Historical aliases supported by Git object/hash evidence are included explicitly. CLI model aliases and architecture choices remain unchanged.

- `attentiongates_dpu_easy_remaining`: missing before migration; historical SHA-256 `d2f1087cba0aa67a1d183689fce12378834b67b44a43c0df5e3eabebbe360493`. No available checkpoint is substituted. Supply the original LFS object to reproduce it.
- Former `Mobile_Net_v3_mag1c_rgb.pth` / `Mobile_Net_v3_dpu_mag1c_rgb.pth` resolve to the BCE weights: Git rename and calibration SHA-256 `e6c4c2…` prove identity. The requested class remains unchanged.
- Former `MobileNetV3_AttentionGates_mag1c_rgb.pth` resolves to its Git-renamed FocalDice weights; it does not resolve to the different `UNetMobileNetV3AttentionGates…` checkpoint.
- `remaining_all_dpu_retrain_original/` paths resolve to their exact migrated objects. The no-underscore `…FocalDiceLossmag1c_rgb.pth` historical DPU checkpoint is a separate deleted experiment, not a spelling alias; it must not be silently replaced. The ONNX `attentiongates_dpu_focaldice_artigo` alias retains that missing historical file (SHA-256 `71e0777e497555b67446a9f45525c68043d3091eca8ad510965347bf9b1c3cc5`, Git `683585f`); it is not substituted with the different remaining_all experiment. The CPU/GPU artigo alias already selected remaining_all before migration and keeps doing so.

Historical CSV/JSON results, ONNX binaries/metadata, calibration manifests, quantization files and XModels are preserved verbatim. Calibration validation compares model identifier, architecture, checkpoint hash and settings; filename changes do not change these contracts. Use the historical `--model` and, where needed, `--architecture` with the migrated checkpoint; explicit model identifiers are not rewritten.

## Updated references

VitisAI registry/loading and native BCE validation; CPU/GPU loader; ARM exporter and export-contract checks; training/evaluation path handling; notebook source names; current usage documentation. Historical artifact/result metadata is intentionally excluded. Training accepts standardized lowercase snake_case stems and resolves known legacy stems before saving; notebook examples use the current BCE loss.

## Verification

- All 25 local files exist at their registered paths and preserve SHA-256 exactly; all 21 originally tracked checkpoints (including unchanged HyperSTARCOP) match their original LFS OIDs. The 20 staged `git mv` operations preserve pointer bytes exactly. Four previously untracked checkpoints were moved without staging.
- All 24 PyTorch state dictionaries passed `load_state_dict(strict=True)` with their listed class. Official HyperSTARCOP passed its loader with strict network validation. No forward pass, interpolation, or tensor conversion was changed.
- 13 available Vitis registry aliases, 25 CPU/GPU aliases, and 19 ARM exporter entries passed resolution and strict loading. The ARM FocalDice artigo alias still references its unavailable original experiment; the easy_remaining alias is also unavailable. Architecture-only entries with no registered weights still require an explicit checkpoint.
- 60 existing calibration manifests passed `validate_calibration` using migrated weights and original artifact hashes/model identifiers/settings. One further manifest requires the unavailable easy_remaining object. No manifests were edited.
- `python -m unittest discover -s VitisAI/tests -v`: 13 tests passed, including five migration regressions (hashes/LFS identities, old paths, distinct deleted experiment, custom path precedence, training stems).
- `python benchmark_arm/tests/test_exportados.py`: first ONNX contract passed; the suite then stopped on the pre-existing missing easy_remaining checkpoint. ONNX resolution and strict PyTorch loading were checked separately for every available exporter entry; no ONNX files were exported or changed.
- Syntax parsing passed for changed Python sources; notebook JSON is valid and saved outputs were preserved. Source searches found no unhandled references to renamed paths. Remaining old names occur in compatibility mappings/tests, historical documents, or preserved results/manifests.
- No training, benchmarks, metric recalculation, quantization, compilation, or ONNX regeneration was run. Native Vitis runtime execution was not run: `pytorch_nndct` and the externally referenced `diagnostico_bce_vitisai35.py` are unavailable in this checkout. Hash/contract validation was exercised without those runtime tools.
- Existing user changes to architecture exports, SegFormer architecture and Vitis common code, plus pre-existing artifact modifications/untracked experiments, were preserved. Only checkpoint renames were staged by `git mv`; source and documentation changes are unstaged.

## Direct checkpoint references

The following source locations now use each migrated basename (the full directory mapping above preserves root versus artigo provenance). Training/evaluation and notebook names are additionally built dynamically. Unreferenced files remain preserved.

| New basename | Source references |
|---|---|
| `final_checkpoint_model.ckpt` | `VitisAI/common.py`, `benchmark_nao_embarcado/model_loader.py`, `benchmark_arm/exportar_to_onnx/scripts/exportar.py` |
| `unet_mobilenet_v3_attentiongates_bce_mag1c_rgb.pth` | `benchmark_nao_embarcado/model_loader.py`, `benchmark_arm/exportar_to_onnx/scripts/exportar.py` |
| `unet_mobilenet_v3_attentiongates_bce_old_mag1c_rgb.pth` | No direct literal caller; registered for explicit loading |
| `unet_mobilenet_v3_attentiongates_focaldice_mag1c_rgb.pth` | `VitisAI/common.py`, `benchmark_nao_embarcado/model_loader.py`, `benchmark_arm/exportar_to_onnx/scripts/exportar.py` |
| `unet_mobilenet_v3_focaldice_mag1c_rgb.pth` | `benchmark_nao_embarcado/model_loader.py`, `benchmark_arm/exportar_to_onnx/scripts/exportar.py` |
| `unet_mobilenet_v2_mag1c_rgb.pth` | `VitisAI/common.py`, `benchmark_nao_embarcado/model_loader.py`, `benchmark_arm/exportar_to_onnx/scripts/exportar.py` |
| `unet_mobilenet_v3_bce_mag1c_rgb.pth` | `VitisAI/common.py`, `benchmark_nao_embarcado/model_loader.py`, `benchmark_arm/exportar_to_onnx/scripts/exportar.py` |
| `unet_mobilenet_v3_attentiongates_dpu_bce_retrained_mag1c_rgb.pth` | `benchmark_nao_embarcado/model_loader.py`, `benchmark_nao_embarcado/benchmark_dpu.py`, `benchmark_arm/exportar_to_onnx/scripts/exportar.py` |
| `unet_mobilenet_v3_attentiongates_dpu_focaldice_remaining_all_mag1c_rgb.pth` | `benchmark_nao_embarcado/model_loader.py` |
| `unet_skip_connections_mag1c_rgb.pth` | `VitisAI/common.py`, `benchmark_nao_embarcado/model_loader.py`, `benchmark_arm/exportar_to_onnx/scripts/exportar.py` |
| `unet_depth_reduced_mag1c_rgb.pth` | `VitisAI/common.py`, `benchmark_nao_embarcado/model_loader.py`, `benchmark_arm/exportar_to_onnx/scripts/exportar.py` |
| `unet_baseline_mag1c_rgb.pth` | `VitisAI/common.py`, `benchmark_nao_embarcado/model_loader.py`, `benchmark_arm/exportar_to_onnx/scripts/exportar.py` |
| `unet_mobilenet_v3_attentiongates_focaldice_remaining_all_mag1c_rgb.pth` | `VitisAI/common.py`, `benchmark_nao_embarcado/model_loader.py`, `benchmark_arm/exportar_to_onnx/scripts/exportar.py` |
| `unet_resnet34_mag1c_rgb.pth` | `VitisAI/common.py`, `benchmark_arm/exportar_to_onnx/scripts/exportar.py` |
| `segformer_b0_attentiongates_bce_mag1c_rgb.pth` | `VitisAI/common.py` |
| `segformer_b0_attentiongates_focaldice_mag1c_rgb.pth` | No direct literal caller; registered for explicit loading |
| `segformer_b0_attentiongates_mag1c_rgb.pth` | No direct literal caller; registered for explicit loading |
| `segformer_b0_focaldice_mag1c_rgb.pth` | No direct literal caller; registered for explicit loading |
| `segformer_b0_mag1c_rgb.pth` | `VitisAI/common.py`, `benchmark_arm/exportar_to_onnx/scripts/exportar.py` |
| `unet_mobilenet_v3_attentiongates_dpu_bce_remaining_all_mag1c_rgb.pth` | `benchmark_nao_embarcado/model_loader.py` |
| `unet_mobilenet_v3_attentiongates_dpu_only_remaining_val_test_mag1c_rgb.pth` | `VitisAI/common.py`, `benchmark_arm/exportar_to_onnx/scripts/exportar.py` |
