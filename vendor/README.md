# vendor/

The trainer code EasyTrainer runs is stored in this folder, so nothing is cloned from GitHub at run time.
The GUI copies `vendor/trainer` into its work folder (`/content/trainer` on Colab) and installs the
Python packages there with pip (pip is the only thing that still needs the internet).

```
vendor/trainer/
  installer.py          creates sd_scripts/venv and pip-installs everything (no git)
  requirements.txt      extra packages, installed after sd_scripts/requirements.txt
  custom_scheduler/     LoraEasyCustomOptimizer: the Came optimizer and the Rex scheduler
  sd_scripts/           the part of sd-scripts the GUI uses (training, LoRA network, WD14 tagger)
  LICENSE               GPL-3.0 (backend)
vendor/check_vendor.py  checks that the bundle is self-contained
```

## Where it comes from

| Folder | Source | Version |
|---|---|---|
| `trainer/sd_scripts/` | kohya-ss/sd-scripts | v0.12.3 (2026-10-26), Apache-2.0 (`LICENSE.md`) |
| `trainer/custom_scheduler/`, `trainer/installer.py`, `trainer/requirements.txt` | LoRA_Easy_Training_scripts_Backend | `main` snapshot, GPL-3.0 (`LICENSE`) |

Only the files the GUI needs were kept: the 50-file import closure of `sdxl_train_network.py`,
`networks/lora.py` and `finetune/tag_images_by_wd14_tagger.py` (1.1 MB of the 20 MB repo). The backend's web
server (`main.py`, `utils/`, tunnels), the Windows files and the other models/scripts of sd-scripts are not here.

## Changes made to the original files

Edits to sd-scripts are marked `EasyTrainer patch` in the code (`grep -rn "EasyTrainer patch" vendor`).
They replace the sed-style patches the GUI used to apply after cloning:

1. `library/checkpoint_io.py`: epoch files are numbered with 2 digits (`name-01.safetensors`, not `name-000001`).
2. `train_network.py`: the final model is saved with its epoch number (`name-10.safetensors`, not `name.safetensors`).
3. `library/dataset.py`: `ImageFile.LOAD_TRUNCATED_IMAGES = True`, so a truncated image does not stop a run.

Other changes: `installer.py` was rewritten (same pip steps, same order; no git/submodules, no server config,
no Windows branches) and `requirements.txt` lost the packages only the backend server used
(`starlette`, `uvicorn`, `pyngrok`, `pycloudflared`, `requests`).

## Things to know

* **PyTorch version.** This sd-scripts documents *PyTorch 2.6.0 or later* as required, but `installer.py`
  still installs the stack it always did (torch 2.5.1 / CUDA 12.4, `TORCH_STACK` block at the top). This has not
  been run on a GPU yet. If training fails on import or on a missing torch API, bump that block (it is
  installed with `--no-deps`, so the `nvidia-*` pins must match the torch build).
* `requirements.txt` is installed last and pins `pytorch_optimizer==3.1.2` (needed by `custom_scheduler`),
  overriding the `3.10.0` pinned by sd-scripts.
* Editing files under `vendor/trainer` takes effect on the next training run. Changing `installer.py`,
  a `requirements.txt`, `setup.py` or `custom_scheduler/` rebuilds the venv.
* The WD14 tagger now runs from this same sd-scripts (it used a separate, older pinned commit before).

## Updating sd-scripts

Copy the new files over `vendor/trainer/sd_scripts`, re-apply the three patches above, then run
`python vendor/check_vendor.py`. A new import that is missing from the bundle is reported there instead of
failing in the middle of a training run.

`Lora_Trainer_XL.ipynb`, `Dataset_Maker.ipynb` and the offline-bundle files are standalone Colab notebooks
that are not part of the GUI; they still clone their own copies.
