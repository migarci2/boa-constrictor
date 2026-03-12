import json
import time
from pathlib import Path

import torch

from boa import BOA
from quantization import estimate_model_size_bytes, make_quantized_copy


def _throughput_mib_s(num_bytes: int, seconds: float) -> float:
    if seconds <= 0:
        return float("inf")
    return num_bytes / (1024 * 1024) / seconds


def run_boa_benchmark(
    model,
    input_path: str | Path,
    output_dir: str | Path,
    name: str,
    device: str,
    chunks_count: int,
    progress: bool = True,
    keep_artifacts: bool = False,
):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    boa_path = output_dir / f"{name}.boa"
    dec_path = output_dir / f"{name}_decompressed.bin"
    original_size = input_path.stat().st_size

    boa = BOA(device, str(boa_path), model)

    t0 = time.perf_counter()
    boa.compress(data_path=str(input_path), chunks_count=chunks_count, progress=progress)
    compress_s = time.perf_counter() - t0

    compressed_size = boa_path.stat().st_size

    t1 = time.perf_counter()
    decompressed = boa.decompress(progress=progress)
    decompress_s = time.perf_counter() - t1

    reference = input_path.read_bytes()
    roundtrip_ok = decompressed == reference

    if keep_artifacts:
        dec_path.write_bytes(decompressed)
    else:
        if dec_path.exists():
            dec_path.unlink()

    metrics = {
        "name": name,
        "original_size_bytes": original_size,
        "compressed_size_bytes": compressed_size,
        "compression_ratio": (original_size / compressed_size) if compressed_size > 0 else float("inf"),
        "compression_seconds": compress_s,
        "decompression_seconds": decompress_s,
        "compression_throughput_mib_s": _throughput_mib_s(original_size, compress_s),
        "decompression_throughput_mib_s": _throughput_mib_s(original_size, decompress_s),
        "roundtrip_ok": bool(roundtrip_ok),
        "model_size_bytes": estimate_model_size_bytes(model),
        "boa_path": str(boa_path),
    }
    return metrics


def benchmark_quantized_variants(
    model,
    input_path: str | Path,
    output_dir: str | Path,
    base_name: str,
    device: str,
    chunks_count: int,
    bits_list: list[int],
    progress: bool = True,
    keep_artifacts: bool = False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for bits in bits_list:
        variant_model = make_quantized_copy(model, bits)
        variant_name = f"{base_name}_w{bits}"
        model_path = output_dir / f"{variant_name}.pt"
        torch.save(variant_model, model_path)
        metrics = run_boa_benchmark(
            variant_model,
            input_path=input_path,
            output_dir=output_dir,
            name=variant_name,
            device=device,
            chunks_count=chunks_count,
            progress=progress,
            keep_artifacts=keep_artifacts,
        )
        metrics["weight_bits"] = int(bits)
        metrics["model_path"] = str(model_path)
        report.append(metrics)
    return report


def write_benchmark_report(path: str | Path, payload: dict):
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2))
