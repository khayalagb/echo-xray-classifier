"""ONNX export + static quantization, with an fp32/ONNX/static latency comparison.

ponytail: static-only, not static-vs-dynamic. Dynamic only quantizes weights
(activations stay fp32), which barely moves the needle on a conv-heavy model
like ResNet18 — static quantizes both after a calibration pass and is where
the real latency win is for CNNs, so it's the one worth shipping (README #6).
"""
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from src.dataset import load_manifest, patient_level_split
from src.model import build_model
from src.transforms import build_transforms

ROOT = Path(__file__).resolve().parent.parent


def export_onnx(model, path: Path):
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy, str(path), input_names=["input"], output_names=["output"], opset_version=13)


def benchmark(session_or_model, sample, n=200, onnx=False):
    for _ in range(10):  # warmup
        _run(session_or_model, sample, onnx)
    start = time.perf_counter()
    for _ in range(n):
        _run(session_or_model, sample, onnx)
    return (time.perf_counter() - start) / n * 1000  # ms/inference


def _run(session_or_model, sample, onnx):
    if onnx:
        session_or_model.run(None, {"input": sample})
    else:
        with torch.no_grad():
            session_or_model(torch.from_numpy(sample))


def main():
    meta = json.load(open(ROOT / "outputs" / "train_meta.json"))
    model = build_model()
    model.load_state_dict(torch.load(ROOT / "outputs" / "model.pt", map_location="cpu"))
    model.eval()

    onnx_path = ROOT / "outputs" / "model.onnx"
    export_onnx(model, onnx_path)

    # static quantization — needs a calibration pass over representative data
    static_model = torch.ao.quantization.QuantWrapper(model)
    static_model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
    torch.ao.quantization.prepare(static_model, inplace=True)

    df = load_manifest()
    calib_df = patient_level_split(df)["val"].sample(min(100, len(df)), random_state=42)
    transform = build_transforms(meta["mean"], meta["std"], train=False)
    with torch.no_grad():
        for _, row in calib_df.iterrows():
            img = Image.open(ROOT / "data" / "processed" / row["path"]).convert("L")
            static_model(transform(img).unsqueeze(0))
    torch.ao.quantization.convert(static_model, inplace=True)

    sample = np.random.randn(1, 3, 224, 224).astype(np.float32)
    ort_session = ort.InferenceSession(str(onnx_path))

    results = {
        "pytorch_fp32_ms": benchmark(model, sample),
        "onnx_fp32_ms": benchmark(ort_session, sample, onnx=True),
        "static_quant_ms": benchmark(static_model, sample),
    }
    print(json.dumps(results, indent=2))
    json.dump(results, open(ROOT / "outputs" / "quantization_benchmark.json", "w"), indent=2)
    torch.save(static_model.state_dict(), ROOT / "outputs" / "model_static_quant.pt")


if __name__ == "__main__":
    main()
