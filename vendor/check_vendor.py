#!/usr/bin/env python3
"""
Checks that the bundled trainer code in vendor/trainer is self-contained.

    python vendor/check_vendor.py

Run it after copying new files into vendor/trainer/sd_scripts (for example when updating sd-scripts).
It needs no GPU and no packages: it only reads the source files. It verifies that
  * every .py file is valid Python,
  * every import of the bundled namespaces (library, networks, finetune, relative imports) points at a
    file that is actually in the bundle,
  * the files the GUI starts or the installer needs are there.
A missing file would otherwise only show up as an ImportError in the middle of a training run.
"""
import ast
import os
import sys
import warnings

warnings.simplefilter("ignore", SyntaxWarning)   # upstream docstrings contain harmless "\\(" escapes

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINER = os.path.join(HERE, "trainer")
SD = os.path.join(TRAINER, "sd_scripts")
LOCAL = ("library", "networks", "finetune")          # namespaces that live inside sd_scripts/

REQUIRED = [                                         # started by the GUI / needed by the installer
  "installer.py", "requirements.txt", "LICENSE",
  "custom_scheduler/setup.py", "custom_scheduler/LoraEasyCustomOptimizer/__init__.py",
  "sd_scripts/requirements.txt", "sd_scripts/setup.py", "sd_scripts/LICENSE.md",
  "sd_scripts/sdxl_train_network.py", "sd_scripts/train_network.py", "sd_scripts/networks/lora.py",
  "sd_scripts/finetune/tag_images_by_wd14_tagger.py", "sd_scripts/library/train_util.py",
  "sd_scripts/finetune/make_captions.py", "sd_scripts/finetune/blip/blip.py", "sd_scripts/finetune/blip/med.py",
  "sd_scripts/finetune/blip/vit.py", "sd_scripts/finetune/blip/med_config.json",   # BLIP photo captions (Dataset_Maker)
]


def py_files(root):
  for dp, dn, fn in os.walk(root):
    dn[:] = [d for d in dn if d not in ("__pycache__", "venv")]
    for f in fn:
      if f.endswith(".py"):
        yield os.path.join(dp, f)


def exists_module(root, dotted):
  p = os.path.join(root, *dotted.split("."))
  return os.path.isfile(p + ".py") or os.path.isfile(os.path.join(p, "__init__.py"))


def check_file(path, problems):
  rel = os.path.relpath(path, SD)
  try:
    tree = ast.parse(open(path, encoding="utf-8-sig").read(), path)
  except SyntaxError as e:
    problems.append(f"{rel}: syntax error: {e}")
    return
  pkg = os.path.dirname(rel).replace(os.sep, ".")
  for n in ast.walk(tree):
    if isinstance(n, ast.Import):
      names = [(a.name, []) for a in n.names]
    elif isinstance(n, ast.ImportFrom):
      mod = n.module or ""
      if n.level:                                    # relative import: resolve against this file's package
        base = pkg.split(".") if pkg else []
        base = base[:len(base) - (n.level - 1)]
        mod = ".".join(base + ([mod] if mod else []))
      names = [(mod, [a.name for a in n.names])]
    else:
      continue
    for mod, subs in names:
      if not mod or (mod.split(".")[0] not in LOCAL and not (isinstance(n, ast.ImportFrom) and n.level)):
        continue
      if not exists_module(SD, mod):
        problems.append(f"{rel}:{n.lineno}: imports '{mod}', which is not in the bundle")
        continue
      init = os.path.join(SD, *mod.split("."), "__init__.py")
      if not os.path.isfile(init):                   # `mod` is a plain module file: its names can't be checked here
        continue
      for s in subs:                                 # `from library import x`: x must be a submodule or named in __init__
        if s != "*" and not exists_module(SD, mod + "." + s):
          if s not in open(init, encoding="utf-8-sig").read():
            problems.append(f"{rel}:{n.lineno}: 'from {mod} import {s}' finds neither a module nor a name")


def main():
  problems = [f"missing: {r}" for r in REQUIRED if not os.path.exists(os.path.join(TRAINER, r))]
  files = list(py_files(TRAINER))
  for f in files:
    if f.startswith(SD + os.sep):
      check_file(f, problems)
    else:
      try:
        ast.parse(open(f, encoding="utf-8-sig").read(), f)
      except SyntaxError as e:
        problems.append(f"{os.path.relpath(f, TRAINER)}: syntax error: {e}")
  if problems:
    print(f"vendor check FAILED ({len(problems)} problem(s)):")
    for p in problems:
      print("  -", p)
    return 1
  print(f"vendor check OK: {len(files)} Python files, all imports of library/networks/finetune resolve inside the bundle.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
