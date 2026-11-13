"""
Trainer environment installer for EasyTrainer.

Derived from LoRA_Easy_Training_scripts_Backend (derrian-distro / hollowstrawberry, GPL-3.0),
trimmed to what the EasyTrainer GUI uses. Compared with the original:
  * no `git` and no git submodules: sd-scripts is already in ./sd_scripts (see vendor/README.md)
  * no backend-server config.json, no Windows branches (the GUI runs venv/bin/python)
  * the pip steps below are the same, in the same order, as the original `installer.py colab`

Run it from the trainer directory (the GUI does this for you):

    python3.10 installer.py

It creates sd_scripts/venv and installs, in order:
  1. the pinned torch / CUDA / xformers stack            (TORCH_STACK below)
  2. sd_scripts/requirements.txt  (includes `-e .`, which makes ./sd_scripts/library importable)
  3. custom_scheduler/            (LoraEasyCustomOptimizer: Came optimizer + Rex scheduler)
  4. requirements.txt             (extra optimizer / logging packages, installed last on purpose)
"""
import os
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------------------
# TORCH_STACK: the only version-sensitive block. Installed with --no-deps, so every
# nvidia-* library has to be listed here and has to match the torch build.
# NOTE: these are the pins the original installer shipped with (torch 2.5.1 / cu124).
#       The bundled sd-scripts documents "PyTorch 2.6.0 or later" as its requirement;
#       see vendor/README.md before changing anything here.
# --------------------------------------------------------------------------------------
TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
TORCH_PACKAGES = ["torch==2.5.1", "torchvision==0.20.1"]
XFORMERS = "xformers==0.0.29.post1"
TORCH_DEPS = [
    "nvidia-cudnn-cu12==9.1.1.17",  # fix non-existent 9.1.0.70
    "typing-extensions==4.15.0",
    "filelock", "sympy", "networkx", "jinja2", "fsspec", "numpy", "pillow",
    "nvidia-cuda-nvrtc-cu12==12.4.127",
    "nvidia-cuda-runtime-cu12==12.4.127",
    "nvidia-cuda-cupti-cu12==12.4.127",
    "nvidia-cublas-cu12==12.4.5.8",
    "nvidia-cufft-cu12==11.2.1.3",
    "nvidia-curand-cu12==10.3.5.147",
    "nvidia-cusolver-cu12==11.6.1.9",
    "nvidia-cusparse-cu12==12.3.1.170",
    "nvidia-nccl-cu12==2.21.5",
    "nvidia-nvtx-cu12==12.4.127",
    "nvidia-nvjitlink-cu12==12.4.127",
    "triton==3.1.0",
]

ACCELERATE_CONFIG = """\
command_file: null
commands: null
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: 'NO'
downcase_fp16: 'NO'
dynamo_backend: 'NO'
fsdp_config: {}
gpu_ids: '0'
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
megatron_lm_config: {}
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_name: null
tpu_zone: null
use_cpu: false"""


def pip_install(pip, *args):
    subprocess.check_call([pip, "install", *args])


def setup_accelerate():
    path = Path.home() / ".cache" / "huggingface" / "accelerate" / "default_config.yaml"
    if path.exists():
        print("Default accelerate config already exists, skipping.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ACCELERATE_CONFIG)


def setup_venv(pip):
    pip_install(pip, "-U", *TORCH_PACKAGES, "--no-deps", "--index-url", TORCH_INDEX)
    pip_install(pip, "-U", *TORCH_DEPS, "--extra-index-url", TORCH_INDEX)
    pip_install(pip, "-U", XFORMERS, "--no-deps", "--index-url", TORCH_INDEX)
    pip_install(pip, "-U", "-r", "requirements.txt")             # cwd = sd_scripts
    pip_install(pip, "-U", "../custom_scheduler/.")
    pip_install(pip, "-U", "-r", "../requirements.txt")


def main():
    if sys.version_info < (3, 10):
        sys.exit("ERROR: Python 3.10 or newer is required to create the trainer environment.")
    if not sys.platform.startswith("linux"):
        sys.exit("ERROR: the trainer environment is only set up on Linux (Colab, WSL2 or a Linux machine).")
    sd_scripts = Path(__file__).resolve().parent / "sd_scripts"
    if not (sd_scripts / "requirements.txt").is_file():
        sys.exit(f"ERROR: {sd_scripts} is missing. The bundled sd-scripts should be copied here before installing.")

    os.chdir(sd_scripts)
    print("creating venv and installing requirements")
    subprocess.check_call([sys.executable, "-m", "venv", "venv"])
    setup_venv(os.path.join("venv", "bin", "pip"))
    setup_accelerate()
    print("completed installing")


if __name__ == "__main__":
    main()
