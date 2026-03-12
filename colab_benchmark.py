import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser(description="Generate and run a Colab-friendly BOA benchmark config.")
    p.add_argument("--data", required=True, help="Path to the training/compression binary file.")
    p.add_argument("--name", default="mingru_colab", help="Experiment name.")
    p.add_argument("--backbone", default="mingru", choices=["mamba", "mingru"], help="Backbone to benchmark.")
    p.add_argument("--device", default="cuda", help="Torch device to use.")
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "fp8"], help="Training precision.")
    p.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    p.add_argument("--seq-len", type=int, default=4096, help="Sequence length.")
    p.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    p.add_argument("--d-model", type=int, default=256, help="Model width.")
    p.add_argument("--num-layers", type=int, default=4, help="Number of layers.")
    p.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    p.add_argument("--chunks-count", type=int, default=1000, help="Number of compression chunks.")
    p.add_argument("--quant-bits", default="8,4", help="Comma-separated weight-only benchmark bitwidths.")
    p.add_argument("--gpu-streams", type=int, default=512, help="BOA_GPU_STREAMS override for Colab runs.")
    p.add_argument("--compress-file", default="", help="Optional alternate file to compress.")
    p.add_argument("--extra-main-args", default="", help="Extra args appended to main.py, e.g. '--evaluate'.")
    p.add_argument("--no-run", action="store_true", help="Only write the config, do not launch main.py.")
    p.add_argument("--no-verify", action="store_true", help="Skip --verify when launching main.py.")
    p.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    return p.parse_args()


def parse_bits(bits_arg: str):
    bits_arg = (bits_arg or "").strip()
    if not bits_arg:
        return []
    return [int(x.strip()) for x in bits_arg.split(",") if x.strip()]


def build_config(args, data_path: Path):
    return {
        "name": args.name,
        "file_path": str(data_path),
        "progress": not args.no_progress,
        "device": args.device,
        "precision": args.precision,
        "dataloader": {
            "seq_len": int(args.seq_len),
            "batch_size": int(args.batch_size),
        },
        "model": {
            "backbone": args.backbone,
            "d_model": int(args.d_model),
            "num_layers": int(args.num_layers),
        },
        "training": {
            "lr": float(args.lr),
            "epochs": int(args.epochs),
        },
        "compression": {
            "chunks_count": int(args.chunks_count),
            "file_to_compress": args.compress_file,
        },
        "benchmark": {
            "quantization_bits": parse_bits(args.quant_bits),
            "keep_artifacts": False,
        },
        "splits": [0.8, 0.1, 0.1],
    }


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    exp_dir = repo_root / "experiments" / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    config_path = exp_dir / f"{args.name}.yaml"
    config = build_config(args, data_path)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    print(f"[colab] wrote config: {config_path}")
    print(f"[colab] benchmark bits: {config['benchmark']['quantization_bits']}")
    print(f"[colab] backbone: {args.backbone}")

    if args.no_run:
        return

    env = os.environ.copy()
    env["BOA_GPU_STREAMS"] = str(args.gpu_streams)

    cmd = [
        sys.executable,
        "main.py",
        "--config",
        str(config_path),
        "--show-timings",
    ]
    if not args.no_verify:
        cmd.append("--verify")
    if args.no_progress:
        cmd.append("--no-progress")
    if args.extra_main_args.strip():
        cmd.extend(args.extra_main_args.strip().split())

    print("[colab] running:", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, env=env, check=True)

    report_path = exp_dir / f"{args.name}_benchmark_report.json"
    if report_path.exists():
        print(f"[colab] benchmark report: {report_path}")


if __name__ == "__main__":
    main()
