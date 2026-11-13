"""Check what this machine can do before any long run (standard library only; run it first on the cluster).

    python scripts/preflight.py

Prints GPUs, driver and CUDA, nvcc, disk space, Python, internet reachability (Hugging Face), and whether the
training libraries import, including the Mamba kernels the Granite 4.0 hybrid wants. Ends with a verdict that
says whether to train the hybrid `granite-4.0-h-1b` or fall back to the plain `granite-4.0-1b`.
"""

from __future__ import annotations

import importlib
import shutil
import socket
import subprocess
import sys


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable ({type(exc).__name__})"


def version(module: str) -> str | None:
    try:
        return str(getattr(importlib.import_module(module), "__version__", "imported"))
    except Exception as exc:  # noqa: BLE001 - any import failure is the answer
        return None if isinstance(exc, ImportError) else f"error: {type(exc).__name__}"


def main() -> int:
    print(f"python {sys.version.split()[0]} at {sys.executable}")
    print("\n== GPUs (never use index 6 on ai-n002)")
    print(
        run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv"])
        or "nvidia-smi: none"
    )
    print("\n== CUDA toolkit")
    nvcc = shutil.which("nvcc")
    print(
        run([nvcc, "--version"]).splitlines()[-1]
        if nvcc
        else "nvcc: not found (the Mamba kernels cannot be built without it)"
    )
    print("\n== disk (current directory)")
    total, used, free = shutil.disk_usage(".")
    print(f"free {free / 1e9:.0f} GB of {total / 1e9:.0f} GB")
    print("\n== internet")
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
        net = True
    except OSError:
        net = False
    print("huggingface.co reachable" if net else "huggingface.co NOT reachable: stage the model and wheels by hand")
    print("\n== libraries")
    found = {
        m: version(m) for m in ("torch", "transformers", "peft", "datasets", "accelerate", "mamba_ssm", "causal_conv1d")
    }
    for m, v in found.items():
        print(f"{m:<14}{v or 'missing'}")
    cuda = False
    if found["torch"]:
        torch = importlib.import_module("torch")
        cuda = bool(torch.cuda.is_available())
        print(f"torch {torch.__version__}, CUDA build {torch.version.cuda}, cuda available: {cuda}")
    print("\n== verdict")
    missing = [m for m in ("torch", "transformers", "peft") if not found[m]]
    if missing:
        print(f"install first: {', '.join(missing)} (uv sync --group train in finetune/)")
    if not cuda:
        print("no usable GPU from Python: fix this before training")
    if found["mamba_ssm"] and found["causal_conv1d"]:
        print("Mamba kernels present: train ibm-granite/granite-4.0-h-1b (the hybrid)")
    else:
        print(
            "Mamba kernels missing: either build them (needs nvcc) or train ibm-granite/granite-4.0-1b (plain, slower at inference)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
