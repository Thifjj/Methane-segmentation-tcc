"""Confere contratos e hashes dos cinco ONNX exportados; execute na raiz."""
import hashlib
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "benchmark_arm/exportar_to_onnx/modelos_convertidos_onnx"
MODELS = {
    "mobilenet_v3_attentiongates": (512, "UNetMobileNetV3AttentionGates_mag1c_rgb.pth"),
    "attentiongates_dpu_easy_remaining_512": (512, "UnetMobilenetV3AttentionGates_dpu_easy_remaining_mag1c_rgb.pth"),
    "attentiongates_dpu_only_remaining_512": (512, "UnetMobilenetV3AttentionGates_dpu_only_remaining_mag1c_rgb.pth"),
    "mobilenet_v3_dpu": (512, "Mobile_Net_v3_dpu_mag1c_rgb.pth"),
    "mobilenet_v3": (512, "Mobile_Net_v3_mag1c_rgb.pth"),
}

for name, (size, checkpoint) in MODELS.items():
    path = OUTPUT / f"{name}.onnx"
    graph = onnx.load(path)
    onnx.checker.check_model(graph, full_check=True)
    props = {p.key: p.value for p in graph.metadata_props}
    assert props["modelo"] == name.removesuffix("_512")
    assert props["ordem_canais"] == "mag1c,460,550,640"
    assert int(props["tamanho_patch"]) == size and props["batch_dinamico"] == "False"
    assert graph.ir_version == 8 and graph.opset_import[0].version == 16
    for tensor, shape in [(graph.graph.input[0], [1, 4, size, size]),
                          (graph.graph.output[0], [1, 1, size, size])]:
        assert tensor.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        assert [d.dim_value for d in tensor.type.tensor_type.shape.dim] == shape
    assert hashlib.sha256((ROOT / "Modelos_treinados" / checkpoint).read_bytes()).hexdigest() == props["checkpoint_sha256"]
    print(name, "OK")
print("Five exported model contracts OK")
