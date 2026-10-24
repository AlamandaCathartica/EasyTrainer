#!/usr/bin/env python3
"""
EasyTrainer GUI - backend.

A small web app that configures, launches and monitors an SDXL Lora training run
(same pipeline as Lora_Trainer_XL.ipynb). It ONLY uses the Python
standard library (+ psutil when it happens to be installed, which it is on
Colab), so it adds nothing to install and cannot be broken
by a PyPI change.

    python kohya_gui.py                 # http://127.0.0.1:7860
    python kohya_gui.py --demo          # fake training run, to preview the UI
    python kohya_gui.py --port 8080 --no-browser

Monitoring comes from three places:
  * the trainer's stdout   -> log, progress, avr_loss curve, epoch markers
  * nvidia-smi / psutil    -> GPU, VRAM, CPU, RAM, disk, trainer process stats
  * the generated TOMLs    -> previewed in the "System & config" tab
"""
import argparse
import collections
import json
import math
import mimetypes
import os
import platform
import random
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
  import psutil
except ImportError:  # the GUI still works, with /proc fallbacks
  psutil = None

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "web")
PARENT = os.path.dirname(HERE)  # where train_network_wrapper.py / offline_bundle.py live

# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------
def detect_colab():
  return os.path.isdir("/content") and (
    "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ or os.path.isdir("/content/sample_data"))

CFG = {          # filled in by main()
  "colab": detect_colab(),
  "demo": False,
  "root": None,            # local root dir (non-Colab)
  "bundle_root": "/content/drive/MyDrive/Loras/_gui",
  "host": "127.0.0.1",
  "port": 7860,
  "token": None,           # set when a public tunnel is on
  "tunnel_url": None,
  "tunnel_error": None,
}

SUPPORTED_IMAGES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

DEFAULTS = {
  "project_name": "",
  "folder_structure": "project",       # project | category
  "dataset_folder": "",                # optional explicit override
  "training_model": "illustrious",     # pony | illustrious | noobai_vpred | noobai_eps | animagine | sdxl
  "custom_model_url": "",              # link, or a path in your Drive / on disk
  "custom_is_diffusers": False,
  "custom_is_vpred": False,
  "wandb_key": "",
  "resolution": 1024,
  "caption_extension": ".txt",
  "shuffle_tags": True,
  "activation_tags": 1,
  "num_repeats": 2,
  "preferred_unit": "epochs",          # epochs | steps
  "how_many": 10,
  "save_every_n_epochs": 1,
  "keep_only_last_n_epochs": 10,
  "unet_lr": "3e-4",
  "text_encoder_lr": "6e-5",
  "lr_scheduler": "cosine_with_restarts",
  "lr_scheduler_number": 3,
  "lr_warmup_ratio": 0.05,
  "min_snr_gamma_enabled": True,
  "min_snr_gamma": 8.0,
  "ip_noise_gamma_enabled": False,
  "ip_noise_gamma": 0.05,
  "multinoise": True,
  "lora_type": "LoRA",
  "network_dim": 8,
  "network_alpha": 4,
  "conv_dim": 16,
  "conv_alpha": 8,
  "train_batch_size": 4,
  "cross_attention": "sdpa",           # sdpa | xformers
  "precision": "full fp16",            # float | full fp16 | full bf16 | mixed fp16 | mixed bf16
  "cache_latents": True,
  "cache_latents_to_drive": True,
  "cache_text_encoder_outputs": False,
  "optimizer": "AdamW8bit",
  "recommended_values": True,
  "optimizer_args": "",
  "continue_from_lora": "",
  "continue_mode": False,              # set by the "Continue training" tab
  "continue_suffix": "_cont",          # new files are saved as <project><suffix>[-N]-000001.safetensors
  "custom_dataset": "",
  "seed": 42,
  "install_mode": "online",            # online | skip
  "ignore_step_limit": False,
  "dm_model": "both",                  # both | a key of TAGGER_MODELS | custom
  "dm_custom_model": "",
  "dm_threshold": 0.25,
  "dm_include_characters": False,
  "dm_blacklist": "virtual youtuber, parody, style parody, official alternate costume, official alternate hairstyle, official alternate hair length, alternate costume, alternate hairstyle, alternate hair length, alternate hair color",
  "dm_activation_tag": "",
  "dm_top_n": 30,
  "settings_version": 2,               # settings saved by the old SD1.5 GUI are ignored
}

HF = "https://huggingface.co"
# preset -> (download url, fixed local file name or None, v-pred)
PRESETS = {
  "pony": ("https://civitai.com/api/download/models/290640", "ponyDiffusionV6XL.safetensors", False),
  "animagine": ("https://civitai.com/api/download/models/293564", "animagineXLV3.safetensors", False),
  "illustrious": (f"{HF}/OnomaAIResearch/Illustrious-xl-early-release-v0/resolve/main/Illustrious-XL-v0.1.safetensors", None, False),
  "noobai_eps": (f"{HF}/Laxhar/noobai-XL-1.1/resolve/main/NoobAI-XL-v1.1.safetensors", None, False),
  "noobai_vpred": (f"{HF}/Laxhar/noobai-XL-Vpred-1.0/resolve/main/NoobAI-XL-Vpred-v1.0.safetensors", None, True),
  "sdxl": (f"{HF}/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors", None, False),
}
VAE_URL = f"{HF}/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors"
REPO = "https://github.com/hollowstrawberry/LoRA_Easy_Training_scripts_Backend"

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
TQDM_ANY = re.compile(r"\d+%\|")
TQDM_STEPS = re.compile(r"(\d+)/(\d+)\s*\[([\d:]+)<([\d:?]+),\s*([\d.?]+)\s*(it/s|s/it)")
AVR_LOSS = re.compile(r"avr_loss=([0-9.]+(?:[eE][+-]?\d+)?)")
EPOCH_RE = re.compile(r"^\s*epoch\s+(\d+)\s*/\s*(\d+)\s*$", re.I)
TOTAL_STEPS_RE = re.compile(r"total optimization steps[^\d]*(\d+)", re.I)
NUM_EPOCHS_RE = re.compile(r"num epochs[^\d]*(\d+)", re.I)


def now():
  return time.time()


def to_float(v, default=0.0):
  try:
    return float(v)
  except (TypeError, ValueError):
    return default


def to_int(v, default=0):
  try:
    return int(float(v))
  except (TypeError, ValueError):
    return default


# --------------------------------------------------------------------------
# minimal TOML writer (so we don't depend on the `toml` package being installed
# *before* the install step runs)
# --------------------------------------------------------------------------
def toml_value(v):
  if isinstance(v, bool):
    return "true" if v else "false"
  if isinstance(v, int):
    return str(v)
  if isinstance(v, float):
    return repr(v)
  if isinstance(v, str):
    return json.dumps(v, ensure_ascii=False)
  if isinstance(v, (list, tuple)):
    return "[" + ", ".join(toml_value(x) for x in v) + "]"
  raise TypeError(f"Unsupported TOML value: {v!r}")


def toml_dumps(sections):
  """sections: ordered dict of table-name -> dict of scalars/lists."""
  out = []
  for name, table in sections.items():
    out.append(f"[{name}]")
    for k, v in table.items():
      out.append(f"{k} = {toml_value(v)}")
    out.append("")
  return "\n".join(out)


def toml_loads(text):
  try:
    import tomllib  # py>=3.11
    return tomllib.loads(text)
  except ImportError:
    pass
  try:
    import toml
    return toml.loads(text)
  except ImportError:
    return None


# --------------------------------------------------------------------------
# paths & dataset
# --------------------------------------------------------------------------
def work_dir():
  if CFG["colab"]:
    return "/content"
  return os.path.join(CFG["root"], "_work")


def paths_for(cfg):
  project = cfg["project_name"].strip()
  if CFG["colab"]:
    root = "/content"
    base_project = os.path.join(root, "drive/MyDrive/Loras")
    base_category = os.path.join(root, "drive/MyDrive/lora_training")
  else:
    root = work_dir()
    base_project = base_category = CFG["root"]
  trainer_dir = os.path.join(work_dir(), "trainer")
  kohya_dir = os.path.join(trainer_dir, "sd_scripts")
  if cfg["folder_structure"] == "project":
    main_dir = base_project
    log_folder = os.path.join(main_dir, "_logs")
    config_folder = os.path.join(main_dir, project)
    images_folder = os.path.join(main_dir, project, "dataset")
    output_folder = os.path.join(main_dir, project, "output")
  else:
    main_dir = base_category
    images_folder = os.path.join(main_dir, "datasets", project)
    output_folder = os.path.join(main_dir, "output", project)
    config_folder = os.path.join(main_dir, "config", project)
    log_folder = os.path.join(main_dir, "log")
  if cfg.get("dataset_folder", "").strip():
    images_folder = os.path.expanduser(cfg["dataset_folder"].strip())
  return {
    "root": root, "main": main_dir, "trainer": trainer_dir, "repo": kohya_dir, "images": images_folder,
    "output": output_folder, "config": config_folder, "logs": log_folder,
    "venv_python": os.path.join(kohya_dir, "venv", "bin", "python"),
    "venv_pip": os.path.join(kohya_dir, "venv", "bin", "pip"),
    "train_script": os.path.join(kohya_dir, "sdxl_train_network.py"),
    "training_toml": os.path.join(config_folder, "training_config.toml"),
    "dataset_toml": os.path.join(config_folder, "dataset_config.toml"),
    "models": "/content" if CFG["colab"] else os.path.join(CFG["root"], "models"),
  }


def dataset_info(cfg, paths=None):
  """Counts images/captions and the resulting step numbers (mirrors validate_dataset)."""
  paths = paths or paths_for(cfg)
  info = {"ok": False, "folders": [], "images": 0, "captions": 0, "error": None}
  batch = max(1, to_int(cfg["train_batch_size"], 1))
  folders = []
  custom = (cfg.get("custom_dataset") or "").strip()
  if custom:
    parsed = toml_loads(custom)
    if parsed is None:
      info["warning"] = "Custom dataset not validated (no TOML parser available)."
    else:
      try:
        for sub in parsed["datasets"][0]["subsets"]:
          folders.append((sub["image_dir"], to_int(sub.get("num_repeats", 1), 1), bool(sub.get("is_reg", False))))
      except Exception:
        info["error"] = "Your custom dataset is invalid. Check the template."
        return info
  else:
    folders.append((paths["images"], max(1, to_int(cfg["num_repeats"], 1)), False))
  pre_steps = 0
  for folder, rep, is_reg in folders:
    entry = {"path": folder, "repeats": rep, "reg": is_reg, "images": 0, "captions": 0, "exists": os.path.isdir(folder)}
    if entry["exists"]:
      try:
        names = os.listdir(folder)
      except OSError as e:
        info["error"] = f"Cannot read {folder}: {e}"
        names = []
      cap = (cfg.get("caption_extension") or ".txt").strip().lower() or ".txt"
      entry["images"] = sum(1 for f in names if f.lower().endswith(SUPPORTED_IMAGES))
      entry["captions"] = sum(1 for f in names if f.lower().endswith(cap))
      entry["invalid"] = [f for f in sorted(names) if not f.lower().endswith((cap, ".npz") + SUPPORTED_IMAGES)][:100]
      seen = {}
      for f in sorted(names):
        if f.lower().endswith(SUPPORTED_IMAGES):
          stem = os.path.splitext(f)[0]
          if stem in seen:
            entry["duplicates"] = [seen[stem], f]
            break
          seen[stem] = f
    info["folders"].append(entry)
    info["images"] += entry["images"]
    info["captions"] += entry["captions"]
    pre_steps += entry["images"] * rep
  steps_per_epoch = pre_steps / batch
  if cfg["preferred_unit"] == "epochs":
    total = int(to_int(cfg["how_many"]) * steps_per_epoch)
    epochs = to_int(cfg["how_many"])
  else:
    total = to_int(cfg["how_many"])
    epochs = int(total / steps_per_epoch) if steps_per_epoch else 0
  info.update({"pre_steps": pre_steps, "steps_per_epoch": steps_per_epoch,
               "total_steps": total, "epochs": epochs, "batch": batch,
               "warmup_steps": int(total * to_float(cfg["lr_warmup_ratio"]))})
  info["ok"] = all(f["exists"] and f["images"] for f in info["folders"]) and not info["error"]
  return info


# --------------------------------------------------------------------------
# training / dataset config (port of create_config in Lora_Trainer.ipynb)
# --------------------------------------------------------------------------
def model_spec(cfg):
  """Resolves the selected/custom model like the notebook's top cell."""
  custom = cfg["custom_model_url"].strip()
  diffusers = bool(cfg["custom_is_diffusers"]) and bool(custom)
  vpred = bool(cfg["custom_is_vpred"]) and bool(custom)
  fixed = None
  if custom:
    url = custom
  else:
    url, fixed, pv = PRESETS.get(cfg["training_model"], PRESETS["illustrious"])
    vpred = vpred or pv
  return {"url": url, "file": fixed, "diffusers": diffusers, "vpred": vpred}


def effective_learning(cfg):
  """Applies the 'recommended values' logic of the XL notebook."""
  unet, te = to_float(cfg["unet_lr"]), to_float(cfg["text_encoder_lr"])
  dim, alpha = to_int(cfg["network_dim"]), to_float(cfg["network_alpha"])
  alpha = int(alpha) if alpha == int(alpha) else alpha   # keep fractional alphas (e.g. 0.5) from continued Loras
  precision = cfg["precision"]
  mixed = "fp16" if "fp16" in precision else "bf16" if "bf16" in precision else "no"
  full = "full" in precision
  opt = cfg["optimizer"]
  args = [a.strip() for a in cfg["optimizer_args"].split(" ") if a.strip()]
  if cfg["recommended_values"]:
    if any(x in opt.lower() for x in ("dadapt", "prodigy")):
      unet = te = 0.75
      alpha = dim
      full = False
    if opt == "Prodigy":
      args = ["decouple=True", "weight_decay=0.01", "betas=[0.9,0.999]", "d_coef=2", "use_bias_correction=True", "safeguard_warmup=True"]
    elif opt == "AdamW8bit":
      args = ["weight_decay=0.1", "betas=[0.9,0.99]"]
    elif opt == "AdaFactor":
      args = ["scale_parameter=False", "relative_step=False", "warmup_init=False"]
    elif opt == "Came":
      args = ["weight_decay=0.04"]
  opt_type = "LoraEasyCustomOptimizer.came.CAME" if opt == "Came" else opt
  sched, sched_type, sched_args = cfg["lr_scheduler"], None, None
  if "rex" in sched:
    sched = "cosine"
    sched_type = "LoraEasyCustomOptimizer.RexAnnealingWarmRestarts.RexAnnealingWarmRestarts"
    sched_args = ["min_lr=1e-9", "gamma=0.9", "d=0.9"]
  return {"unet_lr": unet, "text_encoder_lr": te, "network_dim": dim, "network_alpha": alpha,
          "mixed_precision": mixed, "full_precision": full, "optimizer": opt_type,
          "optimizer_args": args or None, "lr_scheduler": sched, "lr_scheduler_type": sched_type,
          "lr_scheduler_args": sched_args, "lr_warmup_ratio": to_float(cfg["lr_warmup_ratio"]),
          "cycles": to_int(cfg["lr_scheduler_number"])}


def build_configs(cfg, paths, model_file, info, rex_args=None):
  e = effective_learning(cfg)
  spec = model_spec(cfg)
  num = e["cycles"]
  net_args = None
  if cfg["lora_type"].lower() == "locon":
    net_args = [f"conv_dim={to_int(cfg['conv_dim'])}", f"conv_alpha={to_int(cfg['conv_alpha'])}"]
  unit_epochs = cfg["preferred_unit"] == "epochs"
  max_epochs = to_int(cfg["how_many"]) if unit_epochs else None
  max_steps = None if unit_epochs else to_int(cfg["how_many"])
  save_every = to_int(cfg["save_every_n_epochs"]) or max_epochs or 1
  keep_last = to_int(cfg["keep_only_last_n_epochs"]) or max_epochs or 9999
  sched = e["lr_scheduler"]
  warmup = int(info["total_steps"] * e["lr_warmup_ratio"])
  cte = bool(cfg["cache_text_encoder_outputs"])
  mixed, full = e["mixed_precision"], e["full_precision"]
  vpred = spec["vpred"]
  xf = cfg["cross_attention"] == "xformers"
  wandb = cfg["wandb_key"].strip()
  sched_args = list(e["lr_scheduler_args"]) + list(rex_args or []) if e["lr_scheduler_args"] else None

  raw = {
    "network_arguments": {
      "unet_lr": e["unet_lr"], "text_encoder_lr": e["text_encoder_lr"] if not cte else 0,
      "network_dim": e["network_dim"], "network_alpha": e["network_alpha"],
      "network_module": "networks.lora", "network_args": net_args,
      "network_train_unet_only": e["text_encoder_lr"] == 0 or cte,
      "network_weights": cfg["continue_from_lora"].strip() or None,
    },
    "optimizer_arguments": {
      "learning_rate": e["unet_lr"], "lr_scheduler": sched,
      "lr_scheduler_type": e["lr_scheduler_type"], "lr_scheduler_args": sched_args,
      "lr_scheduler_num_cycles": num if sched == "cosine_with_restarts" else None,
      "lr_scheduler_power": num if sched == "polynomial" else None,
      "lr_warmup_steps": warmup if sched not in ("cosine", "constant") else None,
      "optimizer_type": e["optimizer"], "optimizer_args": e["optimizer_args"],
      "loss_type": "l2", "max_grad_norm": 1.0,
    },
    "training_arguments": {
      "lowram": True, "pretrained_model_name_or_path": model_file,
      "vae": "stabilityai/sdxl-vae" if spec["diffusers"] else os.path.join(paths["models"], "sdxl_vae.safetensors"),
      "max_train_steps": max_steps, "max_train_epochs": max_epochs,
      "train_batch_size": to_int(cfg["train_batch_size"], 1), "seed": to_int(cfg["seed"], 42),
      "max_token_length": 225, "xformers": xf, "sdpa": not xf,
      "min_snr_gamma": to_float(cfg["min_snr_gamma"]) if cfg["min_snr_gamma_enabled"] else None,
      "ip_noise_gamma": to_float(cfg["ip_noise_gamma"]) if cfg["ip_noise_gamma_enabled"] else None,
      "no_half_vae": True, "gradient_checkpointing": True, "gradient_accumulation_steps": 1,
      "max_data_loader_n_workers": 1, "persistent_data_loader_workers": True,
      "mixed_precision": mixed, "full_fp16": mixed == "fp16" and full, "full_bf16": mixed == "bf16" and full,
      "cache_latents": bool(cfg["cache_latents"]), "cache_latents_to_disk": bool(cfg["cache_latents_to_drive"]),
      "cache_text_encoder_outputs": cte, "min_timestep": 0, "max_timestep": 1000, "prior_loss_weight": 1.0,
      "multires_noise_iterations": 6 if cfg["multinoise"] else None,
      "multires_noise_discount": 0.3 if cfg["multinoise"] else None,
      "v_parameterization": vpred or None, "scale_v_pred_loss_like_noise_pred": vpred or None,
      "zero_terminal_snr": vpred or None,
    },
    "saving_arguments": {
      "save_precision": "fp16", "save_model_as": "safetensors",
      "save_every_n_epochs": save_every, "save_last_n_epochs": keep_last,
      "output_name": cfg.get("_output_name") or cfg["project_name"], "output_dir": paths["output"],
      "log_prefix": cfg.get("_output_name") or cfg["project_name"], "logging_dir": paths["logs"],
      "wandb_api_key": wandb or None, "log_with": "wandb" if wandb else None,
    },
  }
  train_cfg = {k: {a: b for a, b in v.items() if b is not None} for k, v in raw.items()}

  caption_ext = cfg["caption_extension"].strip()
  if caption_ext and not any(f.get("captions") for f in info["folders"]):
    caption_ext = ""
  general = {
    "resolution": to_int(cfg["resolution"], 1024),
    "shuffle_caption": bool(cfg["shuffle_tags"]) and not cte,
    "keep_tokens": to_int(cfg["activation_tags"]), "flip_aug": False,
    "caption_extension": caption_ext, "enable_bucket": True, "bucket_no_upscale": False,
    "bucket_reso_steps": 64, "min_bucket_reso": 256, "max_bucket_reso": 4096,
  }
  ds_text = toml_dumps({"general": general})
  custom = (cfg.get("custom_dataset") or "").strip()
  if custom:
    ds_text += "\n" + custom + "\n"
  else:
    sub = {"num_repeats": to_int(cfg["num_repeats"], 1), "image_dir": paths["images"]}
    if not caption_ext:
      sub["class_tokens"] = cfg["project_name"]
    ds_text += "\n[[datasets]]\n\n[[datasets.subsets]]\n" + "\n".join(f"{k} = {toml_value(v)}" for k, v in sub.items()) + "\n"
  return toml_dumps(train_cfg), ds_text


def validate_config(cfg, paths, info):
  """Returns an error string or None."""
  name = cfg["project_name"].strip()
  if not name or any(c in name for c in " .()\"'\\/"):
    return "Please choose a valid project name (no spaces or . ( ) \" ' \\ /)."
  if cfg["preferred_unit"] not in ("epochs", "steps") or to_int(cfg["how_many"]) < 1:
    return "'How many' must be at least 1."
  if CFG["demo"]:
    return None
  if info.get("error"):
    return info["error"]
  for f in info["folders"]:
    if not f["exists"]:
      return f"The folder {f['path']} doesn't exist."
    if not f["images"]:
      return f"Your folder {f['path']} has no images."
    if f.get("invalid"):
      shown = "./" + os.path.basename(f["path"].rstrip("/\\"))
      lines = [f"Unsupported file on {shown}/{n}" for n in f["invalid"]]
      if len(f["invalid"]) >= 100:
        lines.append("…and possibly more. Remove these files and try again.")
      else:
        lines.append("Remove or move these files out of the dataset folder, then try again.")
      return "\n".join(lines)
    if f.get("duplicates"):
      return (f"Two images share the same name: {f['duplicates'][0]} and {f['duplicates'][1]}.\n"
              "Rename one of them (the name without the extension must be unique).")
  custom = cfg["custom_model_url"].strip()
  if custom.startswith("/") and not os.path.isfile(custom) and not cfg["custom_is_diffusers"]:
    return "The custom training model you specified was not found."
  if cfg.get("continue_mode") and not cfg["continue_from_lora"].strip():
    return "Pick the Lora you want to continue from."
  if cfg["continue_from_lora"].strip():
    p = cfg["continue_from_lora"].strip()
    if not (p.endswith(".safetensors") and os.path.exists(p)):
      return "Invalid path to existing Lora (must be an existing .safetensors file)."
  if info["total_steps"] > 10000 and not cfg["ignore_step_limit"]:
    return (f"Your total steps ({info['total_steps']}) are too high. You probably made a mistake. "
            "Enable 'Ignore the 10000 step safety limit' if this is intended.")
  if cfg["install_mode"] == "skip" and not os.path.exists(paths["venv_python"]):
    return f"Install mode is 'Installed' but the trainer was not found at {paths['repo']}. Use 'Online' once."
  if CFG["colab"] and not os.path.isdir("/content/drive/MyDrive"):
    return "Google Drive is not mounted. Run the first cell of GUI_Trainer.ipynb."
  return None


# --------------------------------------------------------------------------
# continue training: read a saved Lora's metadata, pick a safe output name
# --------------------------------------------------------------------------
EPOCH_FILE_RE = re.compile(r"-(\d{2,6})\.safetensors$")   # the patched trainer writes name-05.safetensors (2 digits)


def read_lora_meta(path):
  """Reads only the safetensors header (no torch needed). Returns a dict or None."""
  try:
    with open(path, "rb") as f:
      n = int.from_bytes(f.read(8), "little")
      if n <= 0 or n > 50_000_000:
        return None
      header = json.loads(f.read(n))
  except (OSError, ValueError):
    return None
  md = header.get("__metadata__") or {}
  if not isinstance(md, dict) or not any(k.startswith("ss_") for k in md):
    return {"has_meta": False, "dim": None, "alpha": None}

  def num(k, cast=int):
    try:
      return cast(float(md.get(k)))
    except (TypeError, ValueError):
      return None

  net_args = {}
  try:
    parsed = json.loads(md.get("ss_network_args") or "null")
    if isinstance(parsed, dict):
      net_args = parsed
  except ValueError:
    pass
  conv_dim = to_int(net_args.get("conv_dim"), 0) or None
  conv_alpha = to_float(net_args.get("conv_alpha"), 0) or None
  algo = str(net_args.get("algo", "lora")).lower()
  m = EPOCH_FILE_RE.search(os.path.basename(path))
  epoch = num("ss_epoch") or (int(m.group(1)) if m else None)
  return {
    "has_meta": True, "dim": num("ss_network_dim"), "alpha": num("ss_network_alpha", float),
    "conv_dim": conv_dim, "conv_alpha": conv_alpha, "algo": algo,
    "lora_type": "LoCon" if conv_dim and algo == "lora" else "LoRA",
    "epoch": epoch, "steps": num("ss_steps"), "resolution": md.get("ss_resolution"),
    "unet_lr": md.get("ss_unet_lr"), "text_encoder_lr": md.get("ss_text_encoder_lr"),
    "base_model": md.get("ss_sd_model_name"), "images": num("ss_num_train_images"),
    "optimizer": md.get("ss_optimizer"),
  }


def continue_output_name(cfg, paths):
  """<project><suffix>, plus -2, -3... if files with that name already exist (never overwrite)."""
  suffix = re.sub(r"[^A-Za-z0-9_.-]", "", cfg.get("continue_suffix") or "") or "_cont"
  base = cfg["project_name"].strip() + suffix
  out = paths["output"]

  def taken(name):
    if not os.path.isdir(out):
      return False
    return any(fn == name + ".safetensors" or fn.startswith(name + "-") for fn in os.listdir(out))

  name, i = base, 1
  while taken(name):
    i += 1
    name = f"{base}{i}"
  return name


def apply_continue(cfg, paths):
  """Makes a 'continue' run safe: network shape comes from the Lora itself, output name never collides."""
  if not cfg.get("continue_mode"):
    return cfg
  cfg = dict(cfg)
  meta = read_lora_meta(os.path.expanduser(cfg["continue_from_lora"].strip())) if cfg["continue_from_lora"].strip() else None
  cfg["_continue_meta"] = meta
  if meta and meta.get("dim"):
    cfg["network_dim"] = meta["dim"]
    if meta.get("alpha"):
      cfg["network_alpha"] = meta["alpha"]
    cfg["lora_type"] = meta["lora_type"]
    if meta.get("conv_dim"):
      cfg["conv_dim"] = meta["conv_dim"]
      cfg["conv_alpha"] = meta.get("conv_alpha") or cfg["conv_alpha"]
  cfg["_output_name"] = continue_output_name(cfg, paths)
  return cfg


def merged(cfg):
  out = dict(DEFAULTS)
  if isinstance(cfg, dict):
    for k, v in cfg.items():
      if k in DEFAULTS:
        out[k] = v
  for k, dv in DEFAULTS.items():
    if isinstance(dv, bool):
      out[k] = bool(out[k])
    elif isinstance(dv, str):
      out[k] = "" if out[k] is None else str(out[k])
  return out


# --------------------------------------------------------------------------
# system sampler (CPU / RAM / GPU / disk / trainer process)
# --------------------------------------------------------------------------
NVSMI_FIELDS = ["index", "name", "utilization.gpu", "utilization.memory", "memory.used", "memory.total",
                "temperature.gpu", "power.draw", "power.limit", "clocks.sm", "clocks.max.sm", "fan.speed",
                "pstate"]


def _num(s):
  s = s.strip()
  if not s or s.startswith("[") or s.lower() in ("n/a", "none"):
    return None
  try:
    return float(s)
  except ValueError:
    return s


def read_gpus():
  exe = shutil.which("nvidia-smi")
  if not exe:
    return None
  try:
    out = subprocess.run([exe, f"--query-gpu={','.join(NVSMI_FIELDS)}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=4).stdout
  except Exception:
    return None
  gpus = []
  for line in out.strip().splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < len(NVSMI_FIELDS):
      continue
    g = dict(zip(NVSMI_FIELDS, parts))
    gpus.append({
      "index": to_int(g["index"]), "name": g["name"], "util": _num(g["utilization.gpu"]),
      "mem_util": _num(g["utilization.memory"]), "mem_used": _num(g["memory.used"]),
      "mem_total": _num(g["memory.total"]), "temp": _num(g["temperature.gpu"]),
      "power": _num(g["power.draw"]), "power_limit": _num(g["power.limit"]),
      "clock": _num(g["clocks.sm"]), "clock_max": _num(g["clocks.max.sm"]),
      "fan": _num(g["fan.speed"]), "pstate": g["pstate"],
    })
  return gpus


class ProcStat:
  """/proc fallback when psutil is missing."""
  def __init__(self):
    self.prev = None

  def cpu_percent(self):
    try:
      with open("/proc/stat") as f:
        vals = [int(x) for x in f.readline().split()[1:8]]
    except Exception:
      return 0.0
    idle = vals[3] + vals[4]
    total = sum(vals)
    if self.prev is None:
      self.prev = (idle, total)
      return 0.0
    di, dt = idle - self.prev[0], total - self.prev[1]
    self.prev = (idle, total)
    return 0.0 if dt <= 0 else max(0.0, min(100.0, 100.0 * (1 - di / dt)))

  @staticmethod
  def meminfo():
    d = {}
    try:
      with open("/proc/meminfo") as f:
        for line in f:
          k, v = line.split(":", 1)
          d[k] = int(v.split()[0]) * 1024
    except Exception:
      return 0, 0
    total = d.get("MemTotal", 0)
    return total - d.get("MemAvailable", d.get("MemFree", 0)), total


class Sampler(threading.Thread):
  def __init__(self, trainer, interval=1.0, keep=900):
    super().__init__(daemon=True)
    self.trainer = trainer
    self.interval = interval
    self.lock = threading.Lock()
    self.history = collections.deque(maxlen=keep)
    self.seq = 0
    self.procstat = ProcStat()
    self.disk_path = os.path.expanduser("~")
    self._pcache = {}
    self._demo_phase = random.random() * 10
    if psutil:
      psutil.cpu_percent(percpu=True)  # prime

  def _trainer_proc(self):
    pid = self.trainer.pid()
    if not pid or not psutil:
      self._pcache = {}
      return None
    try:
      root = self._pcache.get(pid) or psutil.Process(pid)
      procs = [root] + root.children(recursive=True)
      cache, cpu, rss = {}, 0.0, 0.0
      for p in procs:
        p = self._pcache.get(p.pid, p)      # reuse objects so cpu_percent() has a baseline
        cache[p.pid] = p
        try:
          cpu += p.cpu_percent(None)
          rss += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
          pass
      self._pcache = cache
      return {"cpu": round(cpu, 1), "rss": rss, "procs": len(procs)}
    except Exception:
      self._pcache = {}
      return None

  def _demo_gpu(self):
    training = self.trainer.status() == "training"
    t = now() + self._demo_phase
    util = (91 + 6 * math.sin(t / 3) + random.random() * 3) if training else random.random() * 2
    used = (10850 + 120 * math.sin(t / 9)) if training else 310
    return [{"index": 0, "name": "NVIDIA Tesla T4 (demo)", "util": round(util, 1), "mem_util": round(util * 0.55, 1),
             "mem_used": round(used), "mem_total": 15360, "temp": round(58 + (14 if training else 0) + 2 * math.sin(t / 7)),
             "power": round((66 if training else 11) + 3 * math.sin(t / 2), 1), "power_limit": 70.0,
             "clock": 1590 if training else 300, "clock_max": 1590, "fan": None, "pstate": "P0" if training else "P8"}]

  def sample(self):
    t = now()
    if psutil:
      cores = psutil.cpu_percent(percpu=True)
      cpu = sum(cores) / len(cores) if cores else 0.0
      vm = psutil.virtual_memory()
      ram_used, ram_total = vm.total - vm.available, vm.total
      sw = psutil.swap_memory()
      swap = {"used": sw.used, "total": sw.total}
      try:
        freq = psutil.cpu_freq()
        freq = freq.current if freq else None
      except Exception:
        freq = None
    else:
      cpu, cores, freq = self.procstat.cpu_percent(), [], None
      ram_used, ram_total = self.procstat.meminfo()
      swap = {"used": 0, "total": 0}
    try:
      load = os.getloadavg()[0]
    except (OSError, AttributeError):
      load = None
    try:
      du = shutil.disk_usage(self.disk_path if os.path.exists(self.disk_path) else "/")
      disk = {"used": du.used, "total": du.total, "free": du.free, "path": self.disk_path}
    except OSError:
      disk = None
    gpus = read_gpus()
    if gpus is None and CFG["demo"]:
      gpus = self._demo_gpu()
    return {"t": t, "cpu": round(cpu, 1), "cores": [round(c) for c in cores], "freq": freq, "load": load,
            "ram_used": ram_used, "ram_total": ram_total, "swap": swap, "disk": disk,
            "gpus": gpus, "proc": self._trainer_proc()}

  def run(self):
    while True:
      try:
        s = self.sample()
        with self.lock:
          self.seq += 1
          s["seq"] = self.seq
          self.history.append(s)
      except Exception as e:  # never let monitoring kill the server
        sys.stderr.write(f"[sampler] {e}\n")
      time.sleep(self.interval)

  def since(self, cursor, limit=900):
    with self.lock:
      items = [s for s in self.history if s["seq"] > cursor]
      return items[-limit:], self.seq

  def latest(self):
    with self.lock:
      return self.history[-1] if self.history else None


def static_info():
  info = {"python": platform.python_version(), "platform": platform.platform(), "colab": CFG["colab"],
          "cpu_model": None, "cpu_count": os.cpu_count(), "host": platform.node()}
  try:
    with open("/proc/cpuinfo") as f:
      for line in f:
        if line.lower().startswith("model name"):
          info["cpu_model"] = line.split(":", 1)[1].strip()
          break
  except Exception:
    info["cpu_model"] = platform.processor() or platform.machine()
  if psutil:
    info["cpu_physical"] = psutil.cpu_count(logical=False)
    info["ram_total"] = psutil.virtual_memory().total
  else:
    info["ram_total"] = ProcStat.meminfo()[1]
  exe = shutil.which("nvidia-smi")
  if exe:
    try:
      out = subprocess.run([exe], capture_output=True, text=True, timeout=4).stdout
      m = re.search(r"CUDA Version:\s*([\d.]+)", out)
      info["cuda"] = m.group(1) if m else None
      m = re.search(r"Driver Version:\s*([\d.]+)", out)
      info["driver"] = m.group(1) if m else None
    except Exception:
      pass
  info["packages"] = venv_packages()
  return info


def venv_packages():
  """Versions inside the trainer's own venv (None until it is installed)."""
  names = ("torch", "xformers", "accelerate", "transformers", "diffusers", "bitsandbytes", "safetensors")
  py = os.path.join(work_dir(), "trainer", "sd_scripts", "venv", "bin", "python")
  if not os.path.exists(py):
    return {n: None for n in names}
  code = ("import importlib.metadata as m, json\nr={}\nfor n in %r:\n try: r[n]=m.version(n)\n except Exception: r[n]=None\nprint(json.dumps(r))" % (names,))
  try:
    out = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=20).stdout
    return json.loads(out.strip().splitlines()[-1])
  except Exception:
    return {n: None for n in names}


# --------------------------------------------------------------------------
# the trainer: runs the whole pipeline and parses the output
# --------------------------------------------------------------------------
DEMO_SCRIPT = r'''
import math, random, sys, time
total, epochs, spe, speed = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
out = sys.stdout
def w(s): out.write(s); out.flush()
w("prepare tokenizer\n"); time.sleep(.4)
w("update token length: 225\n")
w("[Dataset 0]\n  batch_size: 2\n  resolution: (512, 512)\n  enable_bucket: True\n"); time.sleep(.4)
w("WARNING: xFormers is enabled (demo)\n")
w("caching latents.\n"); time.sleep(.3)
for i in range(0, 101, 10):
  w(f"\r100%|{'#'*(i//10):<10}| {i}/100 [00:0{i//20}<00:00, 40.2it/s]"); time.sleep(.05)
w("\n")
w("running training / \u5b66\u7fd2\u958b\u59cb\n")
w(f"  num train images * repeats / \u5b66\u7fd2\u753b\u50cf\u306e\u6570\u00d7\u7e70\u308a\u8fd4\u3057\u56de\u6570: {int(spe*2)}\n")
w(f"  num epochs / epoch\u6570: {epochs}\n  batch size per device / \u30d0\u30c3\u30c1\u30b5\u30a4\u30ba: 2\n")
w(f"  total optimization steps / \u5b66\u7fd2\u30b9\u30c6\u30c3\u30d7\u6570: {total}\n")
step, t0, hist = 0, time.time(), []
spent = lambda s: "%02d:%02d" % divmod(int(s), 60)
last_epoch = 0
while step < total:
  ep = min(epochs, int(step / spe) + 1)
  if ep != last_epoch:
    if last_epoch:
      w("\nsaving checkpoint: /demo/output/demo-%02d.safetensors\n" % last_epoch)
    w(f"\nepoch {ep}/{epochs}\n"); last_epoch = ep
  time.sleep(1.0 / speed)
  step += 1
  loss = 0.10 + 0.32 * math.exp(-step / (0.22 * total)) + random.gauss(0, 0.03) * (1.4 - step / total)
  if random.random() < 0.01: loss += 0.25
  hist.append(max(loss, 0.01)); avg = sum(hist[-100:]) / len(hist[-100:])
  el = time.time() - t0; rate = step / el; rem = (total - step) / rate
  pct = int(100 * step / total)
  bar = ("\u2588" * (pct // 10)).ljust(10)
  w(f"\rsteps: {pct:3d}%|{bar}| {step}/{total} [{spent(el)}<{spent(rem)}, {rate:5.2f}it/s, avr_loss={avg:.4f}]")
  if step == total // 2: w("\nWARNING: demo warning in the middle of training\n")
w("\nsaving checkpoint: /demo/output/demo-%02d.safetensors\n" % epochs)
w("\nmodel saved.\n")
'''


class Trainer:
  def __init__(self):
    self.lock = threading.RLock()
    self.run_id = 0
    self._status = "idle"
    self.phase = ""
    self.error = None
    self.proc = None
    self.cfg = None
    self.paths = None
    self.started = None
    self.ended = None
    self.train_started = None
    self.stop_requested = False
    self.rex_total = None
    self.reset_run_state()

  # -- state helpers ------------------------------------------------------
  def reset_run_state(self):
    self.progress = {"step": 0, "total": 0, "epoch": 0, "epochs": 0, "pct": 0.0, "speed": None,
                     "speed_unit": "it/s", "elapsed": None, "remaining": None, "loss": None}
    self.metrics = []         # [step, loss, t]
    self.metrics_base = 0
    self.epoch_marks = []     # steps at which an epoch started
    self.log = []             # [t, level, text]
    self.log_base = 0
    self.live = ""            # latest tqdm line (not appended to the log on every tick)
    self.last_prog_log = 0
    self.last_prog_text = ""
    self.last_step = -1
    self.artifacts = []

  def status(self):
    with self.lock:
      return self._status

  def set_status(self, status, phase=None):
    with self.lock:
      self._status = status
      if phase is not None:
        self.phase = phase

  def pid(self):
    with self.lock:
      p = self.proc
      return p.pid if p and p.poll() is None else None

  def active(self):
    return self.status() in ("validating", "installing", "downloading", "configuring", "training", "stopping")

  # -- log / metrics ------------------------------------------------------
  def add_log(self, text, level=None):
    if level is None:
      low = text.lower()
      if "traceback" in low or "error" in low or "💥" in text or "exception" in low:
        level = "error"
      elif "warn" in low:
        level = "warn"
      elif text.startswith(("✅", "⭐", "📄", "🔄", "🏭", "💿", "📁", "📈", "🔮")):
        level = "ok"
      else:
        level = "info"
    with self.lock:
      self.log.append([now(), level, text])
      if len(self.log) > 20000:
        drop = len(self.log) - 15000
        del self.log[:drop]
        self.log_base += drop

  def handle_line(self, raw):
    line = ANSI.sub("", raw).rstrip()
    if not line.strip():
      return
    if TQDM_ANY.search(line):
      self.handle_progress(line)
      return
    m = EPOCH_RE.match(line)
    if m:
      with self.lock:
        self.progress["epoch"], self.progress["epochs"] = int(m.group(1)), int(m.group(2))
        step = self.progress["step"]
        if not self.epoch_marks or self.epoch_marks[-1] != step:
          if step > 0:
            self.epoch_marks.append(step)
    m = re.match(r"REX_TOTAL (\d+)", line)
    if m:
      self.rex_total = int(m.group(1))
      return
    m = TOTAL_STEPS_RE.search(line)
    if m:
      with self.lock:
        self.progress["total"] = int(m.group(1))
    m = NUM_EPOCHS_RE.search(line)
    if m:
      with self.lock:
        self.progress["epochs"] = int(m.group(1))
    if line.lstrip().lower().startswith("saving checkpoint"):
      self.refresh_artifacts()
    self.add_log(line)

  def handle_progress(self, line):
    with self.lock:
      self.live = line
      m = TQDM_STEPS.search(line)
      is_training = line.lstrip().lower().startswith("steps")
      if m and is_training:
        step, total = int(m.group(1)), int(m.group(2))
        p = self.progress
        p["step"], p["total"] = step, total
        p["pct"] = 100.0 * step / total if total else 0.0
        p["elapsed"], p["remaining"] = m.group(3), m.group(4)
        p["speed"] = None if m.group(5) == "?" else float(m.group(5))
        p["speed_unit"] = m.group(6)
        lm = AVR_LOSS.search(line)
        if lm and step != self.last_step:
          loss = float(lm.group(1))
          p["loss"] = loss
          self.metrics.append([step, loss, now()])
          self.last_step = step
          if len(self.metrics) > 60000:
            drop = len(self.metrics) - 50000
            del self.metrics[:drop]
            self.metrics_base += drop
        if self.train_started is None:
          self.train_started = now()
      # keep the log readable: a progress line at most every 20 s (+ final)
      t = now()
      final = m and m.group(1) == m.group(2)
      if (t - self.last_prog_log > 20 or final) and line.strip() != self.last_prog_text:
        self.last_prog_log = t
        self.last_prog_text = line.strip()
        self.log.append([t, "prog", line.strip()])

  def refresh_artifacts(self):
    pass  # the UI lists the output folder on demand

  # -- subprocess plumbing --------------------------------------------------
  def stream(self, cmd, cwd=None, env=None, shell=False):
    """Runs a command, feeding every output line into handle_line(). Returns the exit code."""
    if self.stop_requested:
      return 130
    shown = cmd if isinstance(cmd, str) else " ".join(
      (cmd[:cmd.index("-c") + 1] + ["<inline script>"] + cmd[cmd.index("-c") + 2:]) if "-c" in cmd else cmd)
    self.add_log("$ " + shown, "cmd")
    e = dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="3",
             BITSANDBYTES_NOWELCOME="1", SAFETENSORS_FAST_GPU="1")
    if env:
      e.update(env)
    try:
      proc = subprocess.Popen(cmd, cwd=cwd, env=e, shell=shell, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, bufsize=0, start_new_session=True)
    except OSError as err:
      self.add_log(f"💥 Could not start process: {err}", "error")
      return 127
    with self.lock:
      self.proc = proc
    fd = proc.stdout.fileno()
    buf = b""
    while True:
      try:
        chunk = os.read(fd, 8192)
      except OSError:
        break
      if not chunk:
        break
      buf += chunk
      parts = re.split(rb"[\r\n]", buf)
      buf = parts.pop()
      for part in parts:
        if part.strip():
          self.handle_line(part.decode("utf-8", "replace"))
      if buf and b"%|" in buf:       # show the in-flight tqdm line immediately
        self.handle_line(buf.decode("utf-8", "replace"))
    if buf.strip():
      self.handle_line(buf.decode("utf-8", "replace"))
    code = proc.wait()
    with self.lock:
      self.proc = None
    return code

  def stop(self):
    with self.lock:
      if not self.active():
        return False
      self.stop_requested = True
      proc = self.proc
      self._status = "stopping"
      self.phase = "Stopping…"
    self.add_log("🛑 Stop requested by user.", "warn")
    if proc and proc.poll() is None:
      def killer():
        try:
          os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
          return
        for _ in range(40):
          if proc.poll() is not None:
            return
          time.sleep(0.25)
        try:
          os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
          pass
      threading.Thread(target=killer, daemon=True).start()
    return True

  # -- pipeline -------------------------------------------------------------
  def start(self, cfg):
    cfg = merged(cfg)
    paths = paths_for(cfg)
    cfg = apply_continue(cfg, paths)
    with self.lock:
      if self.active():
        return {"ok": False, "error": "A run is already in progress."}
    if TAGGER.active():
      return {"ok": False, "error": "The auto tagger is running. Wait for it to finish or stop it first."}
    info = dataset_info(cfg, paths)
    if CFG["demo"] and not info["images"]:
      info.update({"images": 24, "total_steps": info.get("total_steps") or 0})
      steps_per_epoch = 24 * to_int(cfg["num_repeats"], 1) / max(1, to_int(cfg["train_batch_size"], 1))
      total = int(to_int(cfg["how_many"]) * steps_per_epoch) if cfg["preferred_unit"] == "epochs" else to_int(cfg["how_many"])
      info.update({"steps_per_epoch": steps_per_epoch, "total_steps": total,
                   "epochs": to_int(cfg["how_many"]) if cfg["preferred_unit"] == "epochs" else int(total / steps_per_epoch)})
    err = validate_config(cfg, paths, info)
    if err:
      return {"ok": False, "error": err}
    with self.lock:
      self.run_id += 1
      self.reset_run_state()
      self.cfg, self.paths = cfg, paths
      self.error = None
      self.started, self.ended, self.train_started = now(), None, None
      self.stop_requested = False
      self._status, self.phase = "validating", "Checking dataset…"
      self.progress["total"] = info["total_steps"]
      self.progress["epochs"] = info["epochs"]
    SAMPLER.disk_path = paths["output"] if os.path.isdir(paths["output"]) else os.path.dirname(paths["output"])
    if cfg.get("continue_mode"):
      m = cfg.get("_continue_meta") or {}
      shape = f"dim {m['dim']}, alpha {m['alpha']:g}" if m.get("dim") and m.get("alpha") else "shape not found in file, using your settings"
      at = f" (saved at epoch {m['epoch']})" if m.get("epoch") else ""
      self.add_log(f"⏩ Continuing from {os.path.basename(cfg['continue_from_lora'])}{at} · {shape}", "ok")
      self.add_log(f"⏩ New checkpoints will be saved as {cfg['_output_name']}-… so the originals stay untouched", "ok")
    threading.Thread(target=self._pipeline, args=(cfg, paths, info), daemon=True).start()
    return {"ok": True, "run_id": self.run_id}

  def _fail(self, message):
    self.add_log("💥 " + message, "error")
    with self.lock:
      self.error = message
      self._status, self.phase = "failed", message
      self.ended = now()

  def _pipeline(self, cfg, paths, info):
    try:
      self._pipeline_inner(cfg, paths, info)
    except Exception as e:  # noqa
      import traceback
      self.add_log(traceback.format_exc(), "error")
      self._fail(f"Unexpected error: {e}")

  def _check_stopped(self):
    if self.stop_requested:
      with self.lock:
        self._status, self.phase = "stopped", "Stopped by user"
        self.ended = now()
      return True
    return False

  def _pipeline_inner(self, cfg, paths, info):
    self.add_log(f"💿 Project “{cfg['project_name']}”  ·  {info['images']} images  ·  "
                 f"{int(info['steps_per_epoch'])} steps/epoch  ·  {info['total_steps']} total steps", "ok")
    for f in info["folders"]:
      self.add_log(f"📁 {f['path']}  ({f['images']} images × {f['repeats']} repeats"
                   f"{', regularization' if f['reg'] else ''})", "ok")

    if CFG["demo"]:
      return self._demo_pipeline(cfg, paths, info)

    for d in (paths["main"], paths["trainer"], paths["config"], paths["output"], paths["logs"], paths["models"]):
      os.makedirs(d, exist_ok=True)

    # 1) install the trainer (same as the XL notebook; this needs internet) -----------
    marker = os.path.join(paths["trainer"], ".gui_installed")
    if cfg["install_mode"] == "online":
      if os.path.exists(marker) and os.path.exists(paths["venv_python"]):
        self.add_log("✅ Trainer already installed in this session.", "ok")
      else:
        self.set_status("installing", "Installing trainer…")
        self.add_log("🏭 Installing trainer…", "ok")
        t0 = now()
        ok = self._install(cfg, paths)
        if self._check_stopped() or not ok:
          return
        open(marker, "w").write("ok")
        self.add_log(f"✅ Installation finished in {int(now() - t0)} seconds.", "ok")
    elif not os.path.exists(paths["venv_python"]):
      return self._fail(f"Install mode is 'Installed' but {paths['repo']} has no venv.")
    self.patch_wandb(paths, cfg)
    if self._check_stopped():
      return

    # 2) model + VAE ------------------------------------------------------------------
    self.set_status("downloading", "Preparing base model…")
    model_file = self._ensure_model(cfg, paths)
    if self._check_stopped() or not model_file:
      return

    # 3) configs (+ Rex scheduler needs the real step count) ------------------------------
    self.set_status("configuring", "Writing config files…")
    rex_args = None
    train_toml, ds_toml = build_configs(cfg, paths, model_file, info)
    with open(paths["dataset_toml"], "w") as f:
      f.write(ds_toml)
    if effective_learning(cfg)["lr_scheduler_type"]:
      rex_args = self._rex_args(cfg, paths, info)
      if rex_args is None:
        return
      train_toml, ds_toml = build_configs(cfg, paths, model_file, info, rex_args)
    with open(paths["training_toml"], "w") as f:
      f.write(train_toml)
    self.add_log(f"📄 Config saved to {paths['training_toml']}", "ok")
    self.add_log(f"📄 Dataset config saved to {paths['dataset_toml']}", "ok")
    if self._check_stopped():
      return

    # 4) train ----------------------------------------------------------------------------
    self.set_status("training", "Training…")
    self.add_log("⭐ Starting trainer…", "ok")
    cmd = [paths["venv_python"], paths["train_script"], f"--config_file={paths['training_toml']}",
           f"--dataset_config={paths['dataset_toml']}"]
    code = self.stream(cmd, cwd=paths["repo"], env={"PYTHONWARNINGS": "ignore"})
    self._finish(code)

  def _finish(self, code):
    with self.lock:
      self.ended = now()
      if self.stop_requested:
        self._status, self.phase = "stopped", "Stopped by user"
        self.add_log("🛑 Training stopped.", "warn")
      elif code == 0:
        self._status, self.phase = "finished", "Training finished"
        self.progress["pct"] = 100.0
        self.add_log("✅ Done! Your Lora is in the output folder (use the latest epoch).", "ok")
      else:
        self._status, self.phase = "failed", f"Trainer exited with code {code}"
        self.error = self.phase
        self.add_log(f"💥 Trainer exited with code {code}.", "error")

  def _demo_pipeline(self, cfg, paths, info):
    self.set_status("installing", "Demo: pretending to install…")
    time.sleep(1.2)
    self.add_log("✅ (demo) Trainer ready.", "ok")
    self.set_status("downloading", "Demo: pretending to fetch the model…")
    time.sleep(1.2)
    self.add_log("🔄 (demo) Model ready.", "ok")
    self.set_status("configuring", "Writing config files…")
    build_configs(cfg, paths, "/demo/model.safetensors", info)
    self.add_log("📄 (demo) Config generated, not written.", "ok")
    if self._check_stopped():
      return
    self.set_status("training", "Training (demo)…")
    self.add_log("⭐ Starting trainer…", "ok")
    code = self.stream([sys.executable, "-u", "-c", DEMO_SCRIPT, str(info["total_steps"]), str(max(1, info["epochs"])),
                        str(max(1, info["steps_per_epoch"])), str(CFG.get("demo_speed", 6.0))])
    self._finish(code)

  # -- installation (port of install_trainer in Lora_Trainer_XL.ipynb) -------------------
  def _install(self, cfg, paths):
    trainer, kohya = paths["trainer"], paths["repo"]
    can_apt = CFG["colab"] or (hasattr(os, "geteuid") and os.geteuid() == 0)
    steps = []
    if can_apt:
      steps += [["apt-get", "-y", "update", "-qq"], ["apt-get", "install", "-y", "python3.10-venv", "aria2", "-qq"]]
    else:
      self.add_log("Not root: skipping apt. Make sure python3-venv and aria2 are installed.", "warn")
    for c in steps:
      if self.stream(c) != 0 and not self.stop_requested:
        self._fail("apt install failed: " + " ".join(c[:4]))
        return False
    if os.path.isdir(trainer):
      shutil.rmtree(trainer, ignore_errors=True)
    if self.stream(["git", "clone", REPO, trainer]) != 0:
      if not self.stop_requested:
        self._fail("git clone failed (no internet?).")
      return False
    script = os.path.join(trainer, "colab_install.sh")
    os.chmod(script, 0o755)
    if self.stream(["bash", script], cwd=trainer) != 0:
      if not self.stop_requested:
        self._fail("colab_install.sh failed. See the log.")
      return False
    for c in ([paths["venv_pip"], "uninstall", "-y", "rich"], [paths["venv_pip"], "install", "tqdm==4.67.1"]):  # fix logging
      self.stream(c)
    self.patch_sources(kohya)
    return not self.stop_requested

  def patch_sources(self, kohya):
    """Same patches the notebook applies with sed (all idempotent)."""
    def edit(rel, fn):
      p = rel if os.path.isabs(rel) else os.path.join(kohya, rel)
      if not os.path.exists(p):
        return
      s = open(p, encoding="utf-8").read()
      n = fn(s)
      if n != s:
        open(p, "w", encoding="utf-8").write(n)
    edit("library/train_util.py", lambda s: s if "LOAD_TRUNCATED_IMAGES" in s else s.replace(
      "from PIL import Image", "from PIL import Image, ImageFile\nImageFile.LOAD_TRUNCATED_IMAGES=True"))
    edit("library/train_util.py", lambda s: s.replace("{:06d}", "{:02d}"))
    edit("train_network.py", lambda s: s.replace(
      '"." + args.save_model_as)', '"-{:02d}.".format(num_train_epochs) + args.save_model_as)'))
    import glob
    for p in glob.glob(os.path.join(kohya, "venv/lib/python3*/site-packages/diffusers/utils/deprecation_utils.py")):
      edit(p, lambda s: s.replace("if version.parse", "if False:#"))
    for rel in ("library/lpw_stable_diffusion.py", "library/sdxl_lpw_stable_diffusion.py"):
      edit(rel, lambda s: s.replace("from transformers import CLIPFeatureExtractor",
                                    "from transformers import CLIPImageProcessor as CLIPFeatureExtractor"))

  def patch_wandb(self, paths, cfg):
    if not cfg["wandb_key"].strip():
      return
    for rel in ("train_network.py", "sdxl_train.py"):
      p = os.path.join(paths["repo"], rel)
      if os.path.exists(p):
        s = open(p, encoding="utf-8").read()
        n = s.replace("accelerator.log(logs, step=epoch + 1)", "")
        if n != s:
          open(p, "w", encoding="utf-8").write(n)

  def _rex_args(self, cfg, paths, info):
    """Port of calculate_rex_steps(); the bucket maths needs sd-scripts, so it runs in its venv."""
    self.add_log("🤔 Calculating Rex steps", "ok")
    e = effective_learning(cfg)
    cycles = e["cycles"] or 1
    if cfg["preferred_unit"] == "steps":
      total = to_int(cfg["how_many"])
    else:
      self.rex_total = None
      code = self.stream([paths["venv_python"], "-c", REX_SCRIPT, paths["dataset_toml"], str(to_int(cfg["resolution"], 1024)),
                          str(to_int(cfg["train_batch_size"], 1)), str(to_int(cfg["how_many"]))], cwd=paths["repo"])
      if code != 0 or not self.rex_total:
        if not self.stop_requested:
          self._fail("Could not calculate the Rex scheduler steps. See the log.")
        return None
      total = self.rex_total
    args = [f"first_cycle_max_steps={total // cycles}"]
    warm = round(total * e["lr_warmup_ratio"]) // cycles
    if warm > 0:
      args.append(f"warmup_steps={warm}")
    self.add_log(f"  cycle steps: {total // cycles}", "info")
    return args

  # -- model download (port of download_model) -----------------------------------------------
  def _download(self, url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if shutil.which("aria2c"):
      return self.stream(["aria2c", url, "--console-log-level=warn", "-c", "-s", "16", "-x", "16", "-k", "10M",
                          "-d", os.path.dirname(dest), "-o", os.path.basename(dest)])
    return self._urllib_download(url, dest)

  def _ensure_model(self, cfg, paths):
    spec = model_spec(cfg)
    url = spec["url"]
    if not url:
      self._fail("No model selected. Paste a link or path, or pick a preset.")
      return None
    vae_path = os.path.join(paths["models"], "sdxl_vae.safetensors")
    if not spec["diffusers"] and not os.path.exists(vae_path):
      self.add_log("🔄 Downloading SDXL VAE…", "ok")
      if self._download(VAE_URL, vae_path) != 0 or not os.path.exists(vae_path):
        if not self.stop_requested:
          self._fail("The SDXL VAE couldn't be downloaded.")
        return None
    local = os.path.expanduser(url)
    if os.path.isfile(local):
      self.add_log(f"📁 Using local model file: {local}", "ok")
      return local
    if spec["diffusers"]:
      m = re.search(r"huggingface\.co/([^/]+)/([^/?#]+)", url)
      if not m:
        self._fail("Diffusers models must be Hugging Face links (huggingface.co/user/model).")
        return None
      self.add_log("🍃 Diffusers model (validated by sd-scripts).", "ok")
      return f"{m.group(1)}/{m.group(2)}"
    real, model_dir = url, paths["models"]
    path_only = real.lower().split("?")[0]
    if spec["file"]:
      model_file = os.path.join(model_dir, spec["file"])
    elif path_only.endswith((".ckpt", ".safetensors")):
      model_file = os.path.join(model_dir, os.path.basename(real.split("?")[0]))
    else:
      model_file = os.path.join(model_dir, "downloaded_model.safetensors")
    if re.search(r"(?:https?://)?(?:www\.)?huggingface\.co/[^/]+/[^/]+/blob", real):
      real = real.replace("/blob/", "/resolve/")
    m = re.search(r"(?:https?://)?(?:www\.)?civitai\.com/models/([0-9]+)(/[A-Za-z0-9-_]+)?", url)
    if m:
      if m.group(2):
        model_file = os.path.join(model_dir, m.group(2).lstrip("/") + ".safetensors")
      mv = re.search(r"modelVersionId=([0-9]+)", url)
      if not mv:
        self._fail("That civitai link doesn't include a modelVersionId. Right-click the download button and copy the direct link.")
        return None
      real = f"https://civitai.com/api/download/models/{mv.group(1)}"
    if os.path.exists(model_file) and os.path.getsize(model_file) > (50 << 20) and MODEL_CACHE.get("url") == url:
      self.add_log("🔄 Model already downloaded.", "ok")
      return model_file
    self.add_log("🔄 Downloading model…", "ok")
    code = self._download(real, model_file)
    if self.stop_requested:
      return None
    if code != 0 or not os.path.exists(model_file) or os.path.getsize(model_file) < (1 << 20):
      self._fail("The model couldn't be downloaded. Check that the link works without logging in, or use a path "
                 "in your Drive.")
      return None
    if model_file.lower().endswith(".safetensors"):   # a "safetensors" that isn't one is a ckpt
      try:
        with open(model_file, "rb") as f:
          n = int.from_bytes(f.read(8), "little")
        if n > os.path.getsize(model_file) or n > (100 << 20):
          new = os.path.splitext(model_file)[0] + ".ckpt"
          os.replace(model_file, new)
          model_file = new
          self.add_log(f"Renamed model to {os.path.basename(new)}", "info")
      except OSError:
        pass
    MODEL_CACHE["url"] = url
    return model_file

  def _urllib_download(self, url, dest):
    try:
      req = urllib.request.Request(url, headers={"User-Agent": "kohya-gui"})
      with urllib.request.urlopen(req, timeout=30) as r, open(dest, "wb") as out:
        total = int(r.headers.get("Content-Length") or 0)
        got, last = 0, 0
        while True:
          if self.stop_requested:
            return 1
          b = r.read(1 << 20)
          if not b:
            break
          out.write(b)
          got += len(b)
          if now() - last > 3:
            last = now()
            self.add_log(f"  download: {got >> 20} / {total >> 20 if total else '?'} MB", "prog")
      return 0
    except Exception as e:
      self.add_log(f"download error: {e}", "error")
      return 1

  # -- snapshot for the API ---------------------------------------------------------
  def snapshot(self):
    with self.lock:
      p = dict(self.progress)
      s = {
        "run_id": self.run_id, "status": self._status, "phase": self.phase, "error": self.error,
        "started": self.started, "ended": self.ended, "progress": p, "live": self.live,
        "epoch_marks": list(self.epoch_marks), "project": self.cfg["project_name"] if self.cfg else None,
        "pid": self.proc.pid if self.proc and self.proc.poll() is None else None,
        "output_dir": self.paths["output"] if self.paths else None,
      }
      if self.train_started and self._status == "training":
        s["train_elapsed"] = now() - self.train_started
      return s

  def metrics_since(self, cursor):
    with self.lock:
      idx = max(0, cursor - self.metrics_base)
      pts = self.metrics[idx:]
      return pts, self.metrics_base + len(self.metrics)

  def log_since(self, cursor, limit=1500):
    with self.lock:
      idx = max(0, cursor - self.log_base)
      lines = self.log[idx:idx + limit]
      return lines, self.log_base + idx + len(lines)


# Runs inside the sd-scripts venv: same bucket maths as calculate_rex_steps() in the notebook.
REX_SCRIPT = r'''
import math, sys, toml
from pathlib import Path
from PIL import Image
from library.train_util import BucketManager
ds_file, res, bs, epochs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
conf = toml.load(ds_file)
g = conf["general"]
subsets = conf["datasets"][0]["subsets"]
types = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
bm = BucketManager(False, (res, res), g["min_bucket_reso"], g["max_bucket_reso"], g["bucket_reso_steps"])
bm.make_buckets()
for s in subsets:
  for image in Path(s["image_dir"]).iterdir():
    if image.suffix not in types:
      continue
    with Image.open(image) as img:
      bucket_reso, _, _ = bm.select_bucket(img.width, img.height)
      for _ in range(s["num_repeats"]):
        bm.add_image(bucket_reso, image)
steps = sum(math.ceil(len(b) / bs) for b in bm.buckets)
print("REX_TOTAL", steps * epochs)
'''

MODEL_CACHE = {}
TRAINER = Trainer()
SAMPLER = Sampler(TRAINER)

# --------------------------------------------------------------------------
# dataset maker: auto tagger, activation tag, tag statistics
# (port of steps 4, 5 and "Analyze Tags" of Dataset_Maker.ipynb)
# --------------------------------------------------------------------------
TAGGER_REPO = "https://github.com/kohya-ss/sd-scripts"
TAGGER_COMMIT = "e89653975ddf429cdf0c0fd268da0a5a3e8dba1f"   # same pin as Dataset_Maker.ipynb
TAGGER_BOTH = ["SmilingWolf/wd-eva02-large-tagger-v3", "SmilingWolf/wd-vit-large-tagger-v3"]
TAGGER_MODELS = {
  "both": TAGGER_BOTH,
  "SmilingWolf/wd-eva02-large-tagger-v3": ["SmilingWolf/wd-eva02-large-tagger-v3"],
  "SmilingWolf/wd-vit-large-tagger-v3": ["SmilingWolf/wd-vit-large-tagger-v3"],
  "SmilingWolf/wd-vit-tagger-v3": ["SmilingWolf/wd-vit-tagger-v3"],
  "SmilingWolf/wd-swinv2-tagger-v3": ["SmilingWolf/wd-swinv2-tagger-v3"],
  "SmilingWolf/wd-convnext-tagger-v3": ["SmilingWolf/wd-convnext-tagger-v3"],
}
REPO_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
TQDM_COUNT = re.compile(r"(\d+)/(\d+)\s*\[")


def split_tags(text):
  return [t.strip() for t in str(text or "").split(",") if t.strip()]


def dm_check(cfg, need_images=True):
  """Returns (paths, error). Shared by tagging, activation tag and statistics."""
  name = cfg["project_name"].strip()
  if not cfg["dataset_folder"].strip():
    if not name or any(c in name for c in " .()\"'\\/"):
      return None, "Please enter a valid project name (no spaces or . ( ) \" ' \\ /)."
  paths = paths_for(cfg)
  if CFG["colab"] and not os.path.isdir("/content/drive/MyDrive"):
    return paths, "Google Drive is not mounted. Run the first cell of GUI_Trainer.ipynb."
  return paths, None


def dm_files(folder, exts):
  out = []
  for root, dirs, files in os.walk(folder):
    dirs.sort()
    for fn in sorted(files):
      if fn.lower().endswith(exts):
        out.append(os.path.join(root, fn))
  return out


def caption_path(img, ext):
  return os.path.splitext(img)[0] + ext


def dm_stats(cfg, top=50):
  paths, err = dm_check(cfg)
  res = {"ok": False, "error": err, "folder": paths["images"] if paths else None, "images": 0, "captioned": 0,
         "unique": 0, "total_tags": 0, "avg_per_image": 0, "top": []}
  if err:
    return res
  folder = paths["images"]
  if not os.path.isdir(folder):
    res["error"] = f"Dataset folder not found: {folder}"
    return res
  ext = cfg["caption_extension"] if cfg["caption_extension"] in (".txt", ".caption") else ".txt"
  imgs = dm_files(folder, SUPPORTED_IMAGES)
  counter = collections.Counter()
  captioned = 0
  for img in imgs:
    cp = caption_path(img, ext)
    if not os.path.isfile(cp):
      continue
    try:
      with open(cp, "r", encoding="utf-8", errors="replace") as f:
        tags = split_tags(f.read())
    except OSError:
      continue
    captioned += 1
    counter.update(tags)
  total = sum(counter.values())
  res.update({"ok": True, "images": len(imgs), "captioned": captioned, "unique": len(counter), "total_tags": total,
              "avg_per_image": round(total / captioned, 1) if captioned else 0,
              "top": counter.most_common(max(1, min(to_int(top, 50), 200)))})
  return res


def dm_add_tags(cfg, tags):
  """Puts the given tag(s) at the start of every image's caption file (creates missing ones)."""
  paths, err = dm_check(cfg)
  if err:
    return {"ok": False, "error": err}
  tags = list(dict.fromkeys(tags))
  if not tags:
    return {"ok": False, "error": "Enter a tag to add."}
  if any(("\n" in t or "\r" in t) for t in tags):
    return {"ok": False, "error": "A tag can't contain line breaks."}
  folder = paths["images"]
  if not os.path.isdir(folder):
    return {"ok": False, "error": f"Dataset folder not found: {folder}"}
  if TRAINER.active() or TAGGER.active():
    return {"ok": False, "error": "Wait for the current training / tagging job to finish."}
  ext = cfg["caption_extension"] if cfg["caption_extension"] in (".txt", ".caption") else ".txt"
  imgs = dm_files(folder, SUPPORTED_IMAGES)
  if not imgs:
    return {"ok": False, "error": f"No images found in {folder}"}
  modified = created = unchanged = 0
  for img in imgs:
    cp = caption_path(img, ext)
    existing = os.path.isfile(cp)
    cur = []
    if existing:
      with open(cp, "r", encoding="utf-8", errors="replace") as f:
        cur = split_tags(f.read())
    new = list(tags) + [t for t in cur if t not in tags]     # same result as the notebook: remove, then insert at the front
    if existing and new == cur:
      unchanged += 1
      continue
    with open(cp, "w", encoding="utf-8") as f:
      f.write(", ".join(new))
    if existing:
      modified += 1
    else:
      created += 1
  return {"ok": True, "images": len(imgs), "modified": modified, "created": created, "unchanged": unchanged,
          "tags": tags}


def _tiny_png(seed):
  import struct, zlib
  def chunk(t, d):
    c = struct.pack(">I", len(d)) + t + d
    return c + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
  rows = b"".join(b"\x00" + bytes((seed * 37 + x * 9 + y * 5) % 256 for x in range(8)) for y in range(8))
  return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 0, 0, 0, 0)) + \
    chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


class Tagger:
  """Runs finetune/tag_images_by_wd14_tagger.py (installed once into its own venv) like step 4 of the notebook."""
  def __init__(self):
    self.lock = threading.RLock()
    self.run_id = 0
    self._status = "idle"
    self.phase = ""
    self.error = None
    self.proc = None
    self.stop_requested = False
    self.log = []
    self.log_base = 0
    self.live = ""
    self.progress = {"pct": 0.0, "model": 0, "models": 0}
    self.started = self.ended = None
    self.result = None

  def status(self):
    with self.lock:
      return self._status

  def active(self):
    return self.status() in ("installing", "tagging", "stopping")

  def dirs(self):
    base = os.path.join(work_dir(), "tagger")
    kohya = os.path.join(base, "kohya")
    return {"base": base, "kohya": kohya, "python": os.path.join(kohya, "venv", "bin", "python"),
            "pip": os.path.join(kohya, "venv", "bin", "pip"),
            "marker": os.path.join(kohya, f".easytrainer_installed_{TAGGER_COMMIT[:8]}")}

  def add_log(self, text, level=None):
    if level is None:
      low = text.lower()
      if "traceback" in low or "error" in low or "💥" in text or "exception" in low:
        level = "error"
      elif "warn" in low:
        level = "warn"
      elif text.startswith(("✅", "📊", "🏭", "🏷", "📎")):
        level = "ok"
      else:
        level = "info"
    with self.lock:
      self.log.append([now(), level, text])
      if len(self.log) > 5000:
        drop = len(self.log) - 4000
        del self.log[:drop]
        self.log_base += drop

  def set_status(self, status, phase=None):
    with self.lock:
      self._status = status
      if phase is not None:
        self.phase = phase

  def _handle_line(self, raw):
    line = ANSI.sub("", raw).rstrip()
    if not line.strip():
      return
    if TQDM_ANY.search(line):
      m = TQDM_COUNT.search(line)
      with self.lock:
        self.live = line.strip()
        if m and int(m.group(2)) > 0 and self._status == "tagging":
          frac = min(1.0, int(m.group(1)) / int(m.group(2)))
          models = max(1, self.progress["models"])
          self.progress["pct"] = round(100.0 * (self.progress["model"] + frac) / models, 1)
      return
    self.add_log(line)

  def stream(self, cmd, cwd=None, env=None):
    if self.stop_requested:
      return 130
    self.add_log("$ " + " ".join(cmd), "cmd")
    e = dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="3")
    if env:
      e.update(env)
    try:
      proc = subprocess.Popen(cmd, cwd=cwd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              bufsize=0, start_new_session=True)
    except OSError as err:
      self.add_log(f"💥 Could not start process: {err}", "error")
      return 127
    with self.lock:
      self.proc = proc
    fd = proc.stdout.fileno()
    buf = b""
    while True:
      try:
        chunk = os.read(fd, 8192)
      except OSError:
        break
      if not chunk:
        break
      buf += chunk
      parts = re.split(rb"[\r\n]", buf)
      buf = parts.pop()
      for part in parts:
        if part.strip():
          self._handle_line(part.decode("utf-8", "replace"))
      if buf and b"%|" in buf:
        self._handle_line(buf.decode("utf-8", "replace"))
    if buf.strip():
      self._handle_line(buf.decode("utf-8", "replace"))
    code = proc.wait()
    with self.lock:
      self.proc = None
    return code

  def stop(self):
    with self.lock:
      if not self.active():
        return False
      self.stop_requested = True
      proc = self.proc
      self._status, self.phase = "stopping", "Stopping…"
    self.add_log("🛑 Stop requested by user.", "warn")
    if proc and proc.poll() is None:
      def killer():
        try:
          os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
          return
        for _ in range(40):
          if proc.poll() is not None:
            return
          time.sleep(0.25)
        try:
          os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
          pass
      threading.Thread(target=killer, daemon=True).start()
    return True

  # -- start ---------------------------------------------------------------
  def start(self, cfg):
    cfg = merged(cfg)
    paths, err = dm_check(cfg)
    if err:
      return {"ok": False, "error": err}
    if self.active():
      return {"ok": False, "error": "Tagging is already running."}
    if TRAINER.active():
      return {"ok": False, "error": "Training is running. Wait for it to finish or stop it first."}
    model = cfg["dm_model"]
    if model == "custom":
      repo = cfg["dm_custom_model"].strip()
      if not REPO_ID_RE.match(repo):
        return {"ok": False, "error": "Custom model must be a Hugging Face repo id like user/model-name."}
      models = [repo]
    elif model in TAGGER_MODELS:
      models = list(TAGGER_MODELS[model])
    else:
      return {"ok": False, "error": "Unknown tagger model."}
    thr = to_float(cfg["dm_threshold"], 0.25)
    if not 0.0 <= thr <= 1.0:
      return {"ok": False, "error": "Threshold must be between 0 and 1."}
    folder = paths["images"]
    if not CFG["demo"]:
      if not os.path.isdir(folder):
        return {"ok": False, "error": f"Dataset folder not found: {folder}"}
      if not dm_files(folder, SUPPORTED_IMAGES):
        return {"ok": False, "error": f"No images found in {folder}"}
    params = {"models": models, "threshold": thr, "characters": bool(cfg["dm_include_characters"]),
              "blacklist": split_tags(cfg["dm_blacklist"]), "ext": cfg["caption_extension"]
              if cfg["caption_extension"] in (".txt", ".caption") else ".txt"}
    with self.lock:
      self.run_id += 1
      self.log, self.log_base, self.live, self.error, self.result = [], 0, "", None, None
      self.progress = {"pct": 0.0, "model": 0, "models": len(models)}
      self.stop_requested = False
      self.started, self.ended = now(), None
      self._status, self.phase = "installing" if not CFG["demo"] else "tagging", "Preparing…"
    threading.Thread(target=self._run, args=(paths, params), daemon=True).start()
    return {"ok": True, "run_id": self.run_id}

  def _fail(self, message):
    self.add_log("💥 " + message, "error")
    with self.lock:
      self.error = message
      self._status, self.phase = "failed", message
      self.ended = now()

  def _run(self, paths, p):
    try:
      if CFG["demo"]:
        return self._demo(paths, p)
      if not self._install():
        return
      self.set_status("tagging", "Tagging images…")
      d = self.dirs()
      folder = paths["images"]
      for i, repo in enumerate(p["models"]):
        with self.lock:
          self.progress["model"] = i
          self.progress["pct"] = round(100.0 * i / len(p["models"]), 1)
          self.phase = f"Tagging with {repo} ({i + 1}/{len(p['models'])})"
        self.add_log(f"🏷️ Tagging with {repo}", "ok")
        cmd = [d["python"], "finetune/tag_images_by_wd14_tagger.py", folder, f"--repo_id={repo}",
               f"--general_threshold={p['threshold']}",
               f"--character_threshold={p['threshold'] if p['characters'] else 1.1}",
               "--batch_size=8", "--max_data_loader_n_workers=2", f"--caption_extension={p['ext']}",
               "--onnx", "--recursive", "--remove_underscore"]
        if p["blacklist"]:
          cmd += ["--undesired_tags", ",".join(p["blacklist"])]
        if i > 0:
          cmd.append("--append_tags")
        code = self.stream(cmd, cwd=d["kohya"])
        if self.stop_requested:
          return self._stopped()
        if code != 0:
          return self._fail(f"The tagger exited with code {code}. See the log.")
      with self.lock:
        self.progress["pct"] = 100.0
      self._done(paths)
    except Exception as e:            # never leave the UI stuck in "tagging"
      self._fail(f"Unexpected error: {e}")

  def _stopped(self):
    self.add_log("🛑 Tagging stopped. Captions written so far are kept.", "warn")
    with self.lock:
      self._status, self.phase, self.ended = "stopped", "Stopped", now()

  def _done(self, paths):
    cfg = dict(DEFAULTS, **{"project_name": "", "dataset_folder": paths["images"]})
    st = dm_stats(cfg, 5)
    self.add_log(f"✅ Tagging complete: {st.get('captioned', 0)} captioned images, {st.get('unique', 0)} unique tags.", "ok")
    with self.lock:
      self.result = {"captioned": st.get("captioned", 0), "images": st.get("images", 0), "unique": st.get("unique", 0)}
      self._status, self.phase, self.ended = "finished", "Finished", now()

  def _install(self):
    d = self.dirs()
    if os.path.isfile(d["marker"]) and os.path.isfile(d["python"]):
      self.add_log("✅ Tagger already installed in this session.", "ok")
      return True
    self.set_status("installing", "Installing the tagger (first time only, a few minutes)…")
    self.add_log("🏭 Installing the tagger (same pinned sd-scripts as the Dataset Maker notebook)…", "ok")
    can_apt = CFG["colab"] or (hasattr(os, "geteuid") and os.geteuid() == 0)
    if can_apt:
      for c in (["apt-get", "-y", "update", "-qq"], ["apt-get", "install", "-y", "python3.10-venv", "-qq"]):
        if self.stream(c) != 0 and not self.stop_requested:
          self._fail("apt install failed: " + " ".join(c[:4]))
          return False
    else:
      self.add_log("Not root: skipping apt. Make sure python3-venv is installed.", "warn")
    if self.stop_requested:
      return self._stopped() or False
    os.makedirs(d["base"], exist_ok=True)
    shutil.rmtree(d["kohya"], ignore_errors=True)
    py = shutil.which("python3.10") or sys.executable
    steps = [
      (["git", "clone", TAGGER_REPO, d["kohya"]], d["base"]),
      (["git", "reset", "--hard", TAGGER_COMMIT], d["kohya"]),
      ([py, "-m", "venv", "venv"], d["kohya"]),
      ([d["pip"], "install", "-r", "requirements.txt"], d["kohya"]),
      ([d["pip"], "install", "fairscale==0.4.13", "timm==0.6.12"], d["kohya"]),
      ([d["pip"], "install", "onnx", "onnxruntime-gpu==1.20.2", "--extra-index-url",
        "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"], d["kohya"]),
      ([d["pip"], "install", "numpy==1.26.4"], d["kohya"]),
    ]
    for cmd, cwd in steps:
      if self.stream(cmd, cwd=cwd) != 0:
        if self.stop_requested:
          return self._stopped() or False
        self._fail("Install step failed: " + " ".join(cmd[:4]))
        return False
    self.stream([d["pip"], "uninstall", "-y", "rich"], cwd=d["kohya"])
    open(d["marker"], "w").close()
    self.add_log("✅ Tagger installed.", "ok")
    return True

  def _demo(self, paths, p):
    folder = paths["images"]
    real = os.path.realpath(folder)
    if not real.startswith(os.path.realpath(CFG["root"]) + os.sep):
      return self._fail("Demo mode only works inside the demo folder.")
    os.makedirs(folder, exist_ok=True)
    imgs = dm_files(folder, SUPPORTED_IMAGES)
    if not imgs:
      self.add_log("Demo: creating 24 tiny placeholder images.", "warn")
      for i in range(24):
        with open(os.path.join(folder, f"demo_{i:02d}.png"), "wb") as f:
          f.write(_tiny_png(i))
      imgs = dm_files(folder, SUPPORTED_IMAGES)
    vocab = ["1girl", "solo", "long hair", "looking at viewer", "smile", "blush", "simple background", "white background",
             "open mouth", "blue eyes", "brown hair", "school uniform", "skirt", "shirt", "standing", "upper body",
             "bangs", "short hair", "outdoors", "sky", "cloud", "day", "tree", "hair ornament", "ribbon", "jacket",
             "holding", "closed mouth", "black hair", "full body", "hat", "dress", "from side", "red eyes", "water",
             "flower", "sitting", "bow", "gloves", "night", "virtual youtuber", "parody"]
    weights = [1.0 / (i + 1) ** 0.8 for i in range(len(vocab))]
    rnd = random.Random(7)
    for mi, repo in enumerate(p["models"]):
      with self.lock:
        self.progress["model"] = mi
        self.phase = f"Tagging with {repo} ({mi + 1}/{len(p['models'])})"
      self.add_log(f"🏷️ Tagging with {repo} (demo)", "ok")
      for k, img in enumerate(imgs):
        if self.stop_requested:
          return self._stopped()
        picks = []
        for t in rnd.choices(vocab, weights, k=rnd.randint(8, 18)):
          if t not in picks and t not in p["blacklist"] and rnd.random() < 0.5 + p["threshold"] / 2 * 0.0 + (1 - p["threshold"]) * 0.5:
            picks.append(t)
        cp = caption_path(img, p["ext"])
        old = []
        if mi > 0 and os.path.isfile(cp):
          with open(cp, encoding="utf-8") as f:
            old = split_tags(f.read())
        with open(cp, "w", encoding="utf-8") as f:
          f.write(", ".join(old + [t for t in picks if t not in old]))
        with self.lock:
          frac = (k + 1) / len(imgs)
          self.progress["pct"] = round(100.0 * (mi + frac) / len(p["models"]), 1)
          self.live = f"{int(frac * 100)}%| {k + 1}/{len(imgs)} [demo]"
        time.sleep(0.08)
    self._done(paths)

  # -- API -----------------------------------------------------------------
  def poll(self, run, cursor):
    with self.lock:
      same = run == self.run_id
      idx = max(0, (cursor if same else 0) - self.log_base)
      lines = self.log[idx:idx + 800]
      return {"run_id": self.run_id, "status": self._status, "phase": self.phase, "error": self.error,
              "progress": dict(self.progress), "live": self.live, "started": self.started, "ended": self.ended,
              "result": self.result, "reset": not same,
              "lines": lines, "next": self.log_base + idx + len(lines)}


TAGGER = Tagger()


# --------------------------------------------------------------------------
# settings persistence
# --------------------------------------------------------------------------
def settings_path():
  base = "/content/drive/MyDrive/Loras" if CFG["colab"] and os.path.isdir("/content/drive/MyDrive") else (
    CFG["root"] or os.path.expanduser("~"))
  return os.path.join(base, "_gui_settings.json")


def load_settings():
  try:
    with open(settings_path()) as f:
      saved = json.load(f)
    return merged(saved) if saved.get("settings_version") == DEFAULTS["settings_version"] else dict(DEFAULTS)
  except Exception:
    return dict(DEFAULTS)


def save_settings(cfg):
  p = settings_path()
  try:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
      json.dump(merged(cfg), f, indent=1)
    return True
  except OSError:
    return False


ALLOWED_OUT = set()


def list_outputs(cfg):
  paths = paths_for(merged(cfg))
  out_dir = paths["output"]
  ALLOWED_OUT.add(os.path.realpath(out_dir))
  items = []
  if os.path.isdir(out_dir):
    for n in sorted(os.listdir(out_dir)):
      p = os.path.join(out_dir, n)
      if os.path.isfile(p):
        st = os.stat(p)
        item = {"name": n, "size": st.st_size, "mtime": st.st_mtime, "path": p}
        if n.endswith(".safetensors"):
          item["meta"] = read_lora_meta(p)
        items.append(item)
  items.sort(key=lambda x: x["mtime"], reverse=True)
  return {"dir": out_dir, "files": items}


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
  server_version = "KohyaGUI/1.0"

  def log_message(self, fmt, *args):  # keep the console quiet
    pass

  # helpers
  def _send(self, code, body, ctype="application/json", extra=None):
    if isinstance(body, (dict, list)):
      body = json.dumps(body).encode()
    elif isinstance(body, str):
      body = body.encode()
    self.send_response(code)
    self.send_header("Content-Type", ctype)
    self.send_header("Content-Length", str(len(body)))
    self.send_header("Cache-Control", "no-store")
    for k, v in (extra or {}).items():
      self.send_header(k, v)
    self.end_headers()
    if self.command != "HEAD":
      self.wfile.write(body)

  def _json_body(self):
    n = int(self.headers.get("Content-Length") or 0)
    if n > 2_000_000:
      raise ValueError("Request too large")
    raw = self.rfile.read(n) if n else b"{}"
    return json.loads(raw or b"{}")

  def _authed(self, u):
    """With a public tunnel, everything needs the secret token (via ?token= once, then a cookie)."""
    tok = CFG["token"]
    if not tok:
      return True
    cookies = dict(c.strip().split("=", 1) for c in (self.headers.get("Cookie") or "").split(";") if "=" in c)
    if secrets.compare_digest(cookies.get("kgui", ""), tok):
      return True
    q = urllib.parse.parse_qs(u.query).get("token", [""])[0]
    if q and secrets.compare_digest(q, tok):
      self._send(302, b"", "text/plain", {"Location": u.path or "/",
                 "Set-Cookie": f"kgui={tok}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=2592000"})
      return False
    self._send(401, "Unauthorized. Open the full link (with ?token=...) printed by the notebook.", "text/plain")
    return False

  def _host_ok(self):
    if CFG["colab"] or CFG["host"] not in ("127.0.0.1", "localhost"):
      return True
    host = (self.headers.get("Host") or "").split(":")[0]
    return host in ("127.0.0.1", "localhost", "[::1]")

  # routes
  def do_GET(self):
    if not self._host_ok():
      return self._send(403, {"error": "bad host"})
    u = urllib.parse.urlparse(self.path)
    if not self._authed(u):
      return
    q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
    try:
      if u.path in ("/", "/index.html"):
        with open(os.path.join(WEB_DIR, "index.html"), "rb") as f:
          return self._send(200, f.read(), "text/html; charset=utf-8")
      if u.path == "/api/defaults":
        return self._send(200, {"defaults": DEFAULTS, "saved": load_settings(), "colab": CFG["colab"],
                                "demo": CFG["demo"], "bundle_root": CFG["bundle_root"],
                                "info": static_info(), "settings_path": settings_path(),
                                "tunnel_url": CFG["tunnel_url"]})
      if u.path == "/api/poll":
        snap = TRAINER.snapshot()
        same_run = to_int(q.get("run"), -1) == snap["run_id"]
        mc = to_int(q.get("m"), 0) if same_run else 0
        lc = to_int(q.get("l"), 0) if same_run else 0
        pts, mnext = TRAINER.metrics_since(mc)
        lines, lnext = TRAINER.log_since(lc)
        samples, snext = SAMPLER.since(to_int(q.get("s"), 0))
        snap["tunnel_url"] = CFG["tunnel_url"]
        snap_dm = TAGGER.poll(to_int(q.get("dr"), -1), to_int(q.get("dl"), 0))
        return self._send(200, {"status": snap, "reset": not same_run,
                                "metrics": {"points": pts, "next": mnext},
                                "log": {"lines": lines, "next": lnext},
                                "system": {"samples": samples, "next": snext}, "dm": snap_dm})
      if u.path == "/api/info":
        return self._send(200, static_info())
      if u.path == "/api/download":
        return self._download(q.get("path", ""))
      return self._send(404, {"error": "not found"})
    except Exception as e:
      return self._send(500, {"error": str(e)})

  def _download(self, path):
    path = os.path.realpath(path)
    allowed = list(ALLOWED_OUT)
    if TRAINER.paths:
      allowed.append(os.path.realpath(TRAINER.paths["output"]))
    if not any(path.startswith(a + os.sep) for a in allowed) or not os.path.isfile(path):
      return self._send(403, {"error": "not an output file"})
    size = os.path.getsize(path)
    self.send_response(200)
    self.send_header("Content-Type", mimetypes.guess_type(path)[0] or "application/octet-stream")
    self.send_header("Content-Length", str(size))
    self.send_header("Content-Disposition", f'attachment; filename="{os.path.basename(path)}"')
    self.end_headers()
    with open(path, "rb") as f:
      shutil.copyfileobj(f, self.wfile)

  def do_POST(self):
    if not self._host_ok():
      return self._send(403, {"error": "bad host"})
    # Custom header => browsers must preflight cross-site requests, which we never answer.
    if self.headers.get("X-Requested-With") != "kohya-gui":
      return self._send(403, {"error": "missing header"})
    u = urllib.parse.urlparse(self.path)
    if not self._authed(u):
      return
    try:
      body = self._json_body()
      if u.path == "/api/start":
        res = TRAINER.start(body.get("config", {}))
        if res["ok"]:
          save_settings(body.get("config", {}))
        return self._send(200 if res["ok"] else 400, res)
      if u.path == "/api/stop":
        return self._send(200, {"ok": TRAINER.stop()})
      if u.path == "/api/settings":
        return self._send(200, {"ok": save_settings(body.get("config", {}))})
      if u.path == "/api/dataset":
        cfg = merged(body.get("config", {}))
        paths = paths_for(cfg)
        info = dataset_info(cfg, paths) if cfg["project_name"].strip() or cfg["dataset_folder"].strip() \
          or cfg["custom_dataset"].strip() else {"ok": False, "folders": [], "images": 0, "captions": 0,
                                                  "error": None, "empty": True}
        info["paths"] = paths
        info["effective"] = effective_learning(cfg)
        return self._send(200, info)
      if u.path == "/api/dm/stats":
        return self._send(200, dm_stats(merged(body.get("config", {})), body.get("top", 50)))
      if u.path == "/api/dm/tag":
        res = TAGGER.start(body.get("config", {}))
        if res["ok"]:
          save_settings(body.get("config", {}))
        return self._send(200 if res["ok"] else 400, res)
      if u.path == "/api/dm/stop":
        return self._send(200, {"ok": TAGGER.stop()})
      if u.path == "/api/dm/activation":
        cfg = merged(body.get("config", {}))
        res = dm_add_tags(cfg, split_tags(cfg["dm_activation_tag"]))
        if res["ok"]:
          save_settings(body.get("config", {}))
        return self._send(200 if res["ok"] else 400, res)
      if u.path == "/api/lora_info":
        lp = os.path.expanduser(str(body.get("path", "")).strip())
        if not (lp.endswith(".safetensors") and os.path.isfile(lp)):
          return self._send(200, {"ok": False, "error": "File not found (must be an existing .safetensors)."})
        meta = read_lora_meta(lp)
        return self._send(200, {"ok": meta is not None, "meta": meta, "error": None if meta else "Not a valid safetensors file."})
      if u.path == "/api/outputs":
        return self._send(200, list_outputs(body.get("config", {})))
      if u.path == "/api/preview":
        cfg = merged(body.get("config", {}))
        paths = paths_for(cfg)
        cfg = apply_continue(cfg, paths)
        info = dataset_info(cfg, paths)
        info.setdefault("total_steps", 0)
        spec = model_spec(cfg)
        model_file = spec["url"]
        if spec["diffusers"]:
          model_file = "/".join(model_file.rstrip("/").split("/")[-2:])
        elif model_file and not os.path.isfile(os.path.expanduser(model_file)):
          model_file = os.path.join(paths["models"], spec["file"] or os.path.basename(model_file.split("?")[0]) or "model.safetensors")
        t, d = build_configs(cfg, paths, model_file, {**info, "total_steps": info.get("total_steps", 0)})
        return self._send(200, {"training": t, "dataset": d, "paths": paths})
      return self._send(404, {"error": "not found"})
    except Exception as e:
      return self._send(400, {"error": str(e)})


def main():
  ap = argparse.ArgumentParser(description="EasyTrainer GUI")
  ap.add_argument("--host", default="127.0.0.1")
  ap.add_argument("--port", type=int, default=7860)
  ap.add_argument("--root", default=None, help="local projects root (non-Colab). Default ~/Loras")
  ap.add_argument("--bundle-root", default=None, help="Drive folder used to cache tools (cloudflared)")
  ap.add_argument("--demo", action="store_true", help="simulate a training run (no GPU or dataset needed)")
  ap.add_argument("--demo-speed", type=float, default=6.0, help="demo steps per second")
  ap.add_argument("--no-browser", action="store_true")
  ap.add_argument("--tunnel", action="store_true", help="expose the GUI through a free Cloudflare quick tunnel (token protected)")
  ap.add_argument("--url-file", default=None, help="write the public link to this file")
  ap.add_argument("--colab", dest="colab", action="store_true", default=None)
  ap.add_argument("--no-colab", dest="colab", action="store_false")
  a = ap.parse_args()

  CFG.update(host=a.host, port=a.port, demo=a.demo, demo_speed=a.demo_speed)
  if a.colab is not None:
    CFG["colab"] = a.colab
  CFG["root"] = os.path.abspath(os.path.expanduser(a.root or "~/Loras"))
  if a.bundle_root:
    CFG["bundle_root"] = a.bundle_root
  if a.demo and not a.root:
    CFG["root"] = os.path.join(os.path.expanduser("~"), "Loras_demo")
  SAMPLER.disk_path = CFG["root"] if os.path.isdir(CFG["root"]) else os.path.expanduser("~")
  SAMPLER.start()

  if a.tunnel:
    import tunnel
    CFG["token"] = secrets.token_urlsafe(18)

    def got_url(url):
      CFG["tunnel_url"] = url
      full = f"{url}/?token={CFG['token']}"
      print(f"PUBLIC URL: {full}", flush=True)
      if a.url_file:
        with open(a.url_file, "w") as f:
          f.write(full)

    def tunnel_err(msg):
      CFG["tunnel_error"] = str(msg)
      print(f"[tunnel] {msg}", flush=True)

    tunnel.start(a.port, os.path.join(CFG["bundle_root"], "tools"), got_url, tunnel_err)

  server = ThreadingHTTPServer((a.host, a.port), Handler)
  server.daemon_threads = True
  url = f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '') else a.host}:{a.port}"
  print(f"EasyTrainer running at {url}  (colab={CFG['colab']}, demo={CFG['demo']})", flush=True)
  if a.host not in ("127.0.0.1", "localhost"):
    print("WARNING: listening on a non-loopback address. Anyone who can reach it can start training jobs.", flush=True)
  if not a.no_browser and not CFG["colab"]:
    try:
      import webbrowser
      threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    except Exception:
      pass
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    print("\nShutting down…")
    TRAINER.stop()


if __name__ == "__main__":
  main()
