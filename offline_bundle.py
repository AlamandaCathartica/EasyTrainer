"""
Offline / frozen installation helper for kohya-colab.

Two phases:

  build(profile)    -> run ONCE while online. Downloads every wheel (full
                       transitive closure, exact versions), the sd-scripts
                       source at a pinned commit, and the .deb files for aria2
                       into a bundle folder (default: Google Drive).
                       Writes MANIFEST.json with SHA-256 of every file.

  install(profile)  -> run on every session. Uses ONLY the bundle:
                       pip --no-index --find-links, local source tarball,
                       dpkg -i for aria2. Never touches the network, so
                       upstream changes can no longer break the install.

Profiles describe what to freeze. See PROFILES below.
"""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request

DEFAULT_BUNDLE_ROOT = "/content/drive/MyDrive/Loras/_offline_bundle"
HERE = os.path.dirname(os.path.abspath(__file__))

# Exact pins. Anything not listed here is resolved ONCE at build time and then
# frozen into requirements.lock, so it can never drift afterwards.
PROFILES = {
  "sd15": {
    "python": "3.10",   # Colab python at time of freezing; checked at install
    "source_repo": "uYouUs/sd-scripts",
    "source_commit": "",  # FILL IN (build() resolves HEAD and records it if empty)
    "torch_index": "https://download.pytorch.org/whl/cu121",
    "torch_requirements": ["torch==2.4.0", "torchvision==0.19.0", "xformers==0.0.27.post2", "triton==3.0.0"],
    "requirements": [
      "accelerate==0.25.0", "transformers==4.36.2", "diffusers[torch]==0.25.0",
      "ftfy==6.1.1", "opencv-python==4.8.1.78", "einops==0.7.0",
      "pytorch-lightning==1.9.0", "bitsandbytes==0.43.0", "prodigyopt==1.0",
      "lion-pytorch==0.0.6", "tensorboard", "safetensors==0.4.2",
      "altair==4.2.2", "easygui==0.98.3", "toml==0.10.2", "voluptuous==0.13.1",
      "huggingface-hub==0.20.1", "imagesize==1.4.1", "rich==13.7.1",
      "numpy==1.26.4", "pygments",
    ],
    "apt": ["aria2"],
    "wrappers": ["train_network_wrapper.py", "dracula.py"],
  },
  "sd15_es": {  # Spanish_Lora_Trainer keeps its own torch/xformers pair (torch 2.4.1 <-> xformers 0.0.28.post1)
    "python": "3.10",
    "source_repo": "uYouUs/sd-scripts",
    "source_commit": "",
    "torch_index": "https://download.pytorch.org/whl/cu121",
    "torch_requirements": ["torch==2.4.1", "torchvision==0.19.1", "xformers==0.0.28.post1", "triton==3.0.0"],
    "requirements": None,  # filled below from sd15
    "apt": ["aria2"],
    "wrappers": ["train_network_wrapper.py", "dracula.py"],
  },
  "xl_legacy": {
    "python": "3.10",
    "source_repo": "qaneel/kohya-trainer",
    "source_commit": "",
    "torch_index": "https://download.pytorch.org/whl/cu121",
    "torch_requirements": ["torch==2.5.1+cu121", "xformers==0.0.28.post3"],
    "requirements": [
      "accelerate==0.19.0", "transformers==4.30.2", "diffusers==0.18.2",
      "bitsandbytes==0.45.1", "flax==0.7.5", "opencv-python",
      "jax==0.4.23", "jaxlib==0.4.23", "pytorch-lightning==1.9.0",
      "voluptuous==0.13.1", "toml==0.10.2", "ftfy==6.1.1", "einops==0.6.0",
      "safetensors", "pygments", "huggingface-hub==0.22.0",
      "wandb", "invisible-watermark==0.2.0", "open-clip-torch==2.20.0",
      "dadaptation==3.1", "prodigyopt==1.0", "lion-pytorch==0.1.2",
      "came-pytorch", "numpy<2",
    ],
    "apt": ["aria2"],
    "wrappers": ["train_network_xl_wrapper.py", "dracula.py"],
  },
}

PROFILES["sd15_es"]["requirements"] = list(PROFILES["sd15"]["requirements"])


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _run(cmd, **kw):
  print("$", " ".join(cmd) if isinstance(cmd, (list, tuple)) else cmd)
  subprocess.run(cmd, check=True, **kw)


def _sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def _bundle_dir(profile, root):
  return os.path.join(root or DEFAULT_BUNDLE_ROOT, profile)


def _py_tag():
  return f"{sys.version_info.major}.{sys.version_info.minor}"


def _write_manifest(bdir):
  files = {}
  for dp, _, fs in os.walk(bdir):
    for fn in fs:
      if fn == "MANIFEST.json":
        continue
      p = os.path.join(dp, fn)
      files[os.path.relpath(p, bdir)] = _sha256(p)
  manifest = {
    "python": _py_tag(),
    "platform": platform.platform(),
    "machine": platform.machine(),
    "files": files,
  }
  with open(os.path.join(bdir, "MANIFEST.json"), "w") as f:
    json.dump(manifest, f, indent=1, sort_keys=True)
  return manifest


def verify(profile, root=None):
  """Checks every file in the bundle against MANIFEST.json."""
  bdir = _bundle_dir(profile, root)
  mpath = os.path.join(bdir, "MANIFEST.json")
  if not os.path.exists(mpath):
    raise FileNotFoundError(f"No bundle at {bdir}. Run build('{profile}') once while online.")
  manifest = json.load(open(mpath))
  if manifest["python"] != _py_tag():
    raise RuntimeError(
      f"Bundle was built for Python {manifest['python']} but this runtime is {_py_tag()}. "
      "Colab upgraded its Python; rebuild the bundle or use a runtime with the matching Python.")
  bad = [rel for rel, digest in manifest["files"].items()
         if not os.path.exists(os.path.join(bdir, rel)) or _sha256(os.path.join(bdir, rel)) != digest]
  if bad:
    raise RuntimeError("Bundle is corrupted or incomplete: " + ", ".join(bad[:10]))
  return manifest


# ----------------------------------------------------------------------------
# phase 1: BUILD (online, run once)
# ----------------------------------------------------------------------------
def build(profile, root=None, force=False):
  cfg = PROFILES[profile]
  bdir = _bundle_dir(profile, root)
  wheels = os.path.join(bdir, "wheels")
  debs = os.path.join(bdir, "debs")
  if os.path.exists(os.path.join(bdir, "MANIFEST.json")) and not force:
    print(f"Bundle already exists at {bdir}. Pass force=True to rebuild.")
    return
  for d in (wheels, debs):
    os.makedirs(d, exist_ok=True)

  # 1) source code at an exact commit (tarball, no git needed later)
  repo = cfg["source_repo"]
  commit = cfg["source_commit"]
  if not commit:
    out = subprocess.run(["git", "ls-remote", f"https://github.com/{repo}", "HEAD"],
                         check=True, capture_output=True, text=True).stdout
    commit = out.split()[0]
    print(f"source_commit was empty; freezing {repo} at current HEAD {commit}")
  tar_path = os.path.join(bdir, "source.tar.gz")
  urllib.request.urlretrieve(f"https://github.com/{repo}/archive/{commit}.tar.gz", tar_path)

  # 2) wrapper scripts: use the copies shipped in THIS repo, not raw GitHub main
  wdir = os.path.join(bdir, "wrappers")
  os.makedirs(wdir, exist_ok=True)
  for w in cfg["wrappers"]:
    shutil.copy(os.path.join(HERE, w), os.path.join(wdir, w))

  # 3) python wheels, full transitive closure
  req_file = os.path.join(bdir, "requirements.in")
  with open(req_file, "w") as f:
    f.write("\n".join(cfg["requirements"]) + "\n")
  torch_req = os.path.join(bdir, "torch-requirements.in")
  with open(torch_req, "w") as f:
    f.write("\n".join(cfg["torch_requirements"]) + "\n")

  _run([sys.executable, "-m", "pip", "download", "-d", wheels,
        "-r", torch_req, "--extra-index-url", cfg["torch_index"]])
  _run([sys.executable, "-m", "pip", "download", "-d", wheels,
        "-r", req_file, "-r", torch_req,
        "--extra-index-url", cfg["torch_index"]])

  # sd-scripts' own requirements.txt can be unpinned: freeze its closure too
  src_tmp = os.path.join(bdir, "_src")
  shutil.rmtree(src_tmp, ignore_errors=True)
  os.makedirs(src_tmp)
  with tarfile.open(tar_path) as t:
    t.extractall(src_tmp)
  inner = os.path.join(src_tmp, os.listdir(src_tmp)[0])
  sd_req = os.path.join(inner, "requirements.txt")
  if os.path.exists(sd_req):
    # drop the "-e ." line and anything we already pinned above
    lines = [l.strip() for l in open(sd_req) if l.strip() and not l.startswith(("-e", "#"))]
    pinned = {r.split("==")[0].split("[")[0].lower() for r in cfg["requirements"] + cfg["torch_requirements"]}
    extra = [l for l in lines if l.split("==")[0].split(">")[0].split("<")[0].split("[")[0].lower() not in pinned]
    if extra:
      extra_file = os.path.join(bdir, "sdscripts-extra.in")
      open(extra_file, "w").write("\n".join(extra) + "\n")
      _run([sys.executable, "-m", "pip", "download", "-d", wheels, "-r", extra_file,
            "-c", req_file, "--extra-index-url", cfg["torch_index"]])
  shutil.rmtree(src_tmp, ignore_errors=True)

  # 4) resolve once in a throw-away venv and write the exact lock file
  venv = os.path.join("/tmp", f"lockvenv_{profile}")
  shutil.rmtree(venv, ignore_errors=True)
  _run([sys.executable, "-m", "venv", venv])
  vpip = os.path.join(venv, "bin", "pip")
  inputs = ["-r", req_file, "-r", torch_req]
  if os.path.exists(os.path.join(bdir, "sdscripts-extra.in")):
    inputs += ["-r", os.path.join(bdir, "sdscripts-extra.in")]
  _run([vpip, "install", "--no-index", "--find-links", wheels] + inputs)
  frozen = subprocess.run([vpip, "freeze", "--exclude-editable"], check=True,
                          capture_output=True, text=True).stdout
  with open(os.path.join(bdir, "requirements.lock"), "w") as f:
    f.write(frozen)
  shutil.rmtree(venv, ignore_errors=True)

  # 5) apt packages (+ dependencies) as .deb files
  if cfg["apt"]:
    _run(["apt-get", "update", "-qq"])
    deps = subprocess.run(
      ["apt-get", "install", "--print-uris", "-y", "-qq", "--reinstall"] + cfg["apt"],
      check=True, capture_output=True, text=True).stdout
    urls = [l.split()[0].strip("'") for l in deps.splitlines() if l.startswith("'")]
    for u in urls:
      dest = os.path.join(debs, os.path.basename(u))
      if not os.path.exists(dest):
        urllib.request.urlretrieve(u, dest)

  info = {"profile": profile, "source_repo": repo, "source_commit": commit}
  json.dump(info, open(os.path.join(bdir, "source.json"), "w"), indent=1)
  manifest = _write_manifest(bdir)
  size = sum(os.path.getsize(os.path.join(bdir, k)) for k in manifest["files"]) / 1e9
  print(f"\nBundle built at {bdir}: {len(manifest['files'])} files, {size:.2f} GB. "
        f"Frozen for Python {_py_tag()}. You can now install fully offline.")


# ----------------------------------------------------------------------------
# phase 2: INSTALL (offline, every session)
# ----------------------------------------------------------------------------
def install(profile, repo_dir, root=None, extra_pip_flags=()):
  """Installs sd-scripts + all dependencies from the bundle only.

  Blocks network access for pip (PIP_NO_INDEX) so a missing wheel fails loudly
  instead of silently pulling a newer version from PyPI.
  """
  bdir = _bundle_dir(profile, root)
  verify(profile, root)
  wheels = os.path.join(bdir, "wheels")

  # source
  shutil.rmtree(repo_dir, ignore_errors=True)
  os.makedirs(repo_dir, exist_ok=True)
  with tarfile.open(os.path.join(bdir, "source.tar.gz")) as t:
    t.extractall(repo_dir + "_unpack")
  inner = os.path.join(repo_dir + "_unpack", os.listdir(repo_dir + "_unpack")[0])
  shutil.rmtree(repo_dir)
  shutil.move(inner, repo_dir)
  shutil.rmtree(repo_dir + "_unpack", ignore_errors=True)

  # wrappers
  for w in os.listdir(os.path.join(bdir, "wrappers")):
    shutil.copy(os.path.join(bdir, "wrappers", w), os.path.join(repo_dir, w))

  # apt packages
  debs = os.path.join(bdir, "debs")
  if os.path.isdir(debs) and os.listdir(debs):
    deb_files = [os.path.join(debs, f) for f in sorted(os.listdir(debs)) if f.endswith(".deb")]
    _run(["dpkg", "-i", "--force-depends"] + deb_files)

  # pip, offline
  env = dict(os.environ, PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
  _run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", wheels,
        "--no-deps", "-r", os.path.join(bdir, "requirements.lock"), *extra_pip_flags], env=env)
  # --no-deps is safe because the lock already contains the full closure; check it:
  subprocess.run([sys.executable, "-m", "pip", "check"], env=env)  # informational only (Colab preinstalls unrelated packages)

  # sd-scripts itself, without letting pip resolve anything
  _run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
        "--no-build-isolation", "-e", repo_dir], env=env, cwd=repo_dir)
  print("\nOffline install finished. Nothing was downloaded.")


if __name__ == "__main__":
  import argparse
  ap = argparse.ArgumentParser()
  ap.add_argument("action", choices=["build", "verify"])
  ap.add_argument("profile", choices=list(PROFILES))
  ap.add_argument("--root", default=None)
  ap.add_argument("--force", action="store_true")
  a = ap.parse_args()
  if a.action == "build":
    build(a.profile, a.root, a.force)
  else:
    verify(a.profile, a.root)
    print("OK")
