# Kohya Colabs

Accessible Google Colab notebooks for Stable Diffusion Lora training, based on the work of [kohya-ss](https://github.com/kohya-ss/sd-scripts) and [Linaqruf](https://github.com/Linaqruf/kohya-trainer).

If you need support I now have a public [Discord server](https://discord.com/invite/hGHnfda)

You can support me on [Buy me a coffee](https://buymeacoffee.com/holostrawberry)

| |🇬🇧 English|🇪🇸 Spanish|
|:--|:-:|:-:|
| 📊 **Dataset Maker** | [![Open in Colab](https://raw.githubusercontent.com/AlamandaCathartica/EasyTrainer/main/assets/colab-badge.svg)](https://colab.research.google.com/github/AlamandaCathartica/EasyTrainer/blob/main/Dataset_Maker.ipynb) | [![Abrir en Colab](https://raw.githubusercontent.com/AlamandaCathartica/EasyTrainer/main/assets/colab-badge-spanish.svg)](https://colab.research.google.com/github/AlamandaCathartica/EasyTrainer/blob/main/Spanish_Dataset_Maker.ipynb) |
| 🌟 **XL Lora Trainer** | [![Open in Colab](https://raw.githubusercontent.com/AlamandaCathartica/EasyTrainer/main/assets/colab-badge.svg)](https://colab.research.google.com/github/AlamandaCathartica/EasyTrainer/blob/main/Lora_Trainer_XL.ipynb) | - |
| 🖥️ **EasyTrainer (GUI)** | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlamandaCathartica/EasyTrainer/blob/main/GUI_Trainer.ipynb) | - |
| **Lora making Guide** | [Click Here](https://civitai.com/models/22530) | - |
| **Stable Diffusion guide** | [Click Here](https://huggingface.co/hollowstrawberry/stable-diffusion-guide/blob/main/README.md#index) | [Click Aquí](https://huggingface.co/hollowstrawberry/stable-diffusion-guide/blob/main/spanish.md#index) |

### 📊 Dataset Maker - Features

* Able to scrape hundreds of images from the popular anime gallery [Gelbooru](https://gelbooru.com/index.php?page=wiki&s=view&id=18780), that match the conditions set by the user.
* Finds duplicate images using the [FiftyOne](https://docs.voxel51.com/) open-source software.
* Displays the user's dataset back to them through the FiftyOne interface so that they may manually curate their images.
* Able to generate tags for all your anime images using the [Waifu Diffusion 1.4 Tagger](https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2) model.
* Able to generate captions for all your images using the [BLIP](https://huggingface.co/spaces/Salesforce/BLIP) model.
* Gives you the ability to edit hundreds of text files at once, to add/remove/replace tags inside them dynamically.
* Works inside your Google Drive by default.
* Connects easily with Lora Trainer.

### 🌟 XL Lora Trainer

* Can train LoRA and LoCon for Stable Diffusion XL, includes several model options for anime.
* One click to install and start training.
* Can work with multiple colab configurations, including T4 (free) and A100.
* Offers most parameters while setting useful values behind the scenes to keep it simple.
* Allows you to optionally define multiple folders for training.
* Uses the latest technologies to train quickly.
* Works inside your Google Drive by default and connects easily with Dataset Maker.

&nbsp;

![Dataset Maker screenshot](assets/datasetmaker.png)


### 🧊 Offline bundle (not used by the GUI)

`offline_bundle.py`, `Build_Offline_Bundle.ipynb` and the `train_network*_wrapper.py` files froze the install of the SD1.5 / legacy XL trainers, which were removed from this copy. They are kept only for reference. The XL trainer installs online (`colab_install.sh`), exactly like `Lora_Trainer_XL.ipynb`.

### 🖥️ GUI (loss / GPU / CPU / log monitor)

`gui/` is a Material 3 web interface for the Lora trainer (same pipeline and settings as `Lora_Trainer_XL.ipynb`). It needs **no extra packages** (Python standard library only; uses `nvidia-smi` and `psutil` if present) and no internet for the page itself.

* **Setup**: all trainer options as forms; live dataset check (images, steps/epoch, total steps); Prodigy/DAdapt overrides shown.
* **Overview**: progress, step/epoch/speed/ETA, interactive loss chart (smoothing, log scale, epoch markers, CSV export), GPU utilisation / VRAM / temperature / power / clocks, CPU (per core), RAM, trainer process and disk, with history sparklines.
* **Log**: live trainer output with filter, warning/error filters, follow, wrap, download.
* **Outputs**: saved epochs with download. **System & config**: hardware/software versions, paths, and a preview of the generated TOML files.

**Colab:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlamandaCathartica/EasyTrainer/blob/main/GUI_Trainer.ipynb) Pick a GPU runtime and run the cell. Nothing needs to be uploaded: the notebook clones this repo into the Colab session, mounts Drive for your datasets/outputs, and opens a free Cloudflare tunnel link.
**Dataset maker mode:** switch to *Dataset maker* at the top of the GUI to tag a dataset without leaving the page: pick a project name, choose the WD14 tagger model(s) and threshold (like step 4 of `Dataset_Maker.ipynb`), put a tag at the front of every `.txt`, and see a bar graph of the most common tags.

**Continue mode:** switch to *Continue* to train an existing Lora further. Pick a saved `.safetensors` from the project's output folder (or paste a path); its rank (dim/alpha) and Lora type are read from the file's metadata so the shapes always match. Choose how many more epochs/steps and adjust the learning rates (there is a *Halve both LRs* shortcut). New checkpoints are saved as `<project>_cont-…` (a number is added if that name is taken), so your original files are never overwritten. Dataset, base model and optimizer are shared with the Trainer tab, and the same live monitor is used.
**Local:** `python gui/kohya_gui.py` and open http://127.0.0.1:7860 (`--root`, `--port`, `--tunnel` available).
**Preview without a GPU:** `python gui/kohya_gui.py --demo` simulates a training run.

Install modes: *Install (online)* (default, like the notebook) or *Already installed*. The loss curve comes from the trainer's `avr_loss` output; the server only listens on localhost unless you pass `--host`.
