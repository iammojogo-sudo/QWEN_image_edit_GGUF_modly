import json
import platform
import subprocess
import sys
from pathlib import Path

IS_WIN = platform.system() == "Windows"


def venv_python(venv):
    return venv / ("Scripts/python.exe" if IS_WIN else "bin/python")


def pip(venv, *args):
    subprocess.run([str(venv_python(venv)), "-m", "pip"] + list(args), check=True)


def setup(python_exe, ext_dir, gpu_sm):
    venv = ext_dir / "venv"
    if not venv.exists():
        print("creating venv...")
        subprocess.run([str(python_exe), "-m", "venv", str(venv)], check=True)
    else:
        print("venv exists, skipping creation")

    pip(venv, "install", "--upgrade", "pip", "wheel", "setuptools")

    if gpu_sm >= 100:
        index = "https://download.pytorch.org/whl/cu128"
        torch_pkgs = ["torch>=2.7.0", "torchvision>=0.22.0"]
    elif gpu_sm >= 70:
        index = "https://download.pytorch.org/whl/cu124"
        torch_pkgs = ["torch==2.6.0", "torchvision==0.21.0"]
    else:
        index = "https://download.pytorch.org/whl/cu118"
        torch_pkgs = ["torch==2.5.1", "torchvision==0.20.1"]

    print("installing torch (sm%d)..." % gpu_sm)
    pip(venv, "install", *torch_pkgs, "--index-url", index)

    print("installing dependencies...")
    pip(venv, "install",
        "diffusers>=0.37.0",
        "transformers>=4.53.0",
        "accelerate>=0.34.0",
        "huggingface_hub>=0.24.0",
        "gguf>=0.10.0",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "Pillow",
        "numpy",
    )

    # bitsandbytes powers the optional 4-bit text encoder used by 'balanced' mode.
    # It's CUDA-only and the wheel can occasionally fail; the streaming Low VRAM mode
    # doesn't need it, so don't let a failure abort the whole install.
    print("installing bitsandbytes (4-bit text encoder for Balanced mode)...")
    try:
        pip(venv, "install", "bitsandbytes>=0.44.0")
    except subprocess.CalledProcessError:
        print("bitsandbytes failed to install — Balanced mode falls back to the default "
              "encoder; Low VRAM mode is unaffected")

    print("done")


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        setup(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) == 2:
        a = json.loads(sys.argv[1])
        setup(Path(a["python_exe"]), Path(a["ext_dir"]), int(a["gpu_sm"]))
    else:
        print("usage: setup.py <python_exe> <ext_dir> <gpu_sm>")
        sys.exit(1)
