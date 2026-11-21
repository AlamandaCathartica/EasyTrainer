"""Cloudflare quick tunnel (trycloudflare.com, free, no account) for running the GUI on Colab."""
import os, platform, re, shutil, stat, subprocess, tempfile, threading, urllib.request

URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def _local_dir():
  """A real local disk folder (Google Drive's FUSE mount can't chmod/execute files)."""
  d = os.path.join(tempfile.gettempdir(), "easytrainer_tools")
  os.makedirs(d, exist_ok=True)
  return d


def _make_runnable(src, log):
  """Copies a binary to local disk, marks it executable and returns the local path (or None)."""
  try:
    dst = os.path.join(_local_dir(), "cloudflared")
    if os.path.abspath(src) != os.path.abspath(dst):
      shutil.copyfile(src, dst)
    os.chmod(dst, os.stat(dst).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return dst if os.access(dst, os.X_OK) else None
  except Exception as e:
    log(f"Could not prepare cached cloudflared ({e}); downloading a fresh copy.")
    return None


def ensure_binary(cache_dir, log=print):
  found = shutil.which("cloudflared")
  if found:
    return found
  # 1) a copy cached in Drive: only used as a download cache, always executed from local disk
  cached = os.path.join(cache_dir, "cloudflared")
  try:
    if os.path.isfile(cached) and os.path.getsize(cached) > 1_000_000:
      ready = _make_runnable(cached, log)
      if ready:
        return ready
  except OSError:
    pass
  # 2) download to local disk (never fails because of Drive permissions)
  arch = "arm64" if platform.machine().lower() in ("aarch64", "arm64") else "amd64"
  url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{arch}"
  local = os.path.join(_local_dir(), "cloudflared")
  log(f"Downloading cloudflared ({arch})...")
  tmp = local + ".part"
  urllib.request.urlretrieve(url, tmp)
  os.chmod(tmp, os.stat(tmp).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
  os.replace(tmp, local)
  # 3) best-effort cache in Drive for the next session; a failure here is harmless
  try:
    os.makedirs(cache_dir, exist_ok=True)
    shutil.copyfile(local, cached + ".part")
    os.replace(cached + ".part", cached)
    log(f"Cached cloudflared in {cache_dir}")
  except Exception as e:
    log(f"(Could not cache cloudflared in Drive: {e}. Not a problem, it will be downloaded again next time.)")
  return local


def start(port, cache_dir, on_url, on_error=print):
  """Starts the tunnel in the background; calls on_url(public_url) once it is up."""
  def run():
    try:
      exe = ensure_binary(cache_dir, on_error)
      proc = subprocess.Popen([exe, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
      for line in proc.stdout:
        m = URL_RE.search(line)
        if m:
          on_url(m.group(0))
      on_error("cloudflared exited.")
    except Exception as e:
      on_error(f"Tunnel failed: {e}")
  threading.Thread(target=run, daemon=True).start()
