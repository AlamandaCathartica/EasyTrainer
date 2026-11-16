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
# TORCH_STACK: the only version-sensitive block.
# torch, torchvision and xformers are a matched set (each torchvision / xformers wheel is built
# against ONE torch version). The bundled sd-scripts needs PyTorch >= 2.6.0.
# The three versions are also written to constraints.txt, and every later pip step runs with
# `-c constraints.txt`, so nothing in requirements.txt can silently upgrade torch while leaving
# torchvision behind ("operator torchvision::nms does not exist"). If a requirement really needs
# another torch, pip stops with a resolver conflict that names the package.
# --------------------------------------------------------------------------------------
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
PYPI_INDEX = "https://pypi.org/simple"  # the nvidia-* / triton dependencies of torch come from here
TORCH_VERSION = "2.7.1"
TORCHVISION_VERSION = "0.22.1"
XFORMERS_VERSION = "0.0.31.post1"  # built for torch 2.7.1

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


def setup_venv(venv_dir):
    pip = os.path.join(venv_dir, "bin", "pip")
    python = os.path.join(venv_dir, "bin", "python")
    constraints = os.path.abspath(os.path.join(venv_dir, "constraints.txt"))
    with open(constraints, "w") as f:
        f.write(f"torch=={TORCH_VERSION}\ntorchvision=={TORCHVISION_VERSION}\nxformers=={XFORMERS_VERSION}\n")

    # 1. the matched torch stack, with its dependencies (nvidia-*, triton, ...) resolved by pip
    pip_install(pip, f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}",
                f"xformers=={XFORMERS_VERSION}", "--index-url", TORCH_INDEX, "--extra-index-url", PYPI_INDEX)
    check_torch(python)
    # 2-4. everything else, forbidden from touching the torch stack
    pip_install(pip, "-U", "-c", constraints, "-r", "requirements.txt")             # cwd = sd_scripts
    pip_install(pip, "-U", "-c", constraints, "../custom_scheduler/.")
    pip_install(pip, "-U", "-c", constraints, "-r", "../requirements.txt")
    check_torch(python, with_transformers=True)


def check_torch(python, with_transformers=False):
    """Fail here, with the versions printed, instead of in the middle of a training run."""
    code = ("import torch, torchvision; from torchvision.ops import nms; "
            "print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| cuda build', torch.version.cuda)")
    if with_transformers:
        code += "; from transformers import CLIPTextModel"
    if subprocess.call([python, "-c", code]) != 0:
        subprocess.call([os.path.join(os.path.dirname(python), "pip"), "list", "--format=freeze"])
        sys.exit("ERROR: the torch / torchvision / transformers stack in the venv does not import (versions listed above).")


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
    setup_venv("venv")
    setup_accelerate()
    print("completed installing")


if __name__ == "__main__":
    main()
