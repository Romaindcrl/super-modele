"""La CLI Kaggle, avec la clé d'API de .env : le code commun à kaggle_watch.py et au dashboard."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OWNER = "romaindl"
KAGGLE = ["uvx", "--from", "kaggle==2.2.4", "kaggle"]


def env() -> dict:
    """L'environnement, plus les variables de .env (KAGGLE_API_TOKEN), jamais affichées."""
    extra = dict(line.split("=", 1) for line in (ROOT / ".env").read_text().splitlines()
                 if "=" in line and not line.startswith("#")) if (ROOT / ".env").exists() else {}
    return {**os.environ, **{k.strip(): v.strip().strip('"').strip("'") for k, v in extra.items()}}


def kaggle(*args: str, check: bool = True, timeout: float = 1800) -> str:
    """Une commande de la CLI Kaggle ; renvoie sa sortie. Une commande bloquée (réseau) s'arrête au bout de
    `timeout` secondes : elle ne doit jamais figer l'appelant."""
    try:
        done = subprocess.run([*KAGGLE, *args], capture_output=True, text=True, env=env(), timeout=timeout)
    except subprocess.TimeoutExpired:
        if check:
            raise RuntimeError(f"kaggle {' '.join(args)} : délai dépassé")
        return "timeout"
    out = (done.stdout + done.stderr).strip()
    if check and done.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} : {out}")
    return out


def kernel_status(slug: str) -> str:
    """queued, running, complete, error, cancel, absent (session jamais lancée), d'après `kernels status`,
    ou unknown : Kaggle injoignable (réseau coupé, Mac en veille…). Les appelants gardent alors l'état
    d'avant, au lieu de croire la session disparue."""
    out = kaggle("kernels", "status", f"{OWNER}/{slug}", check=False, timeout=60).lower()
    for word in ("complete", "error", "cancel", "running", "queued"):
        if f"kernelworkerstatus.{word}" in out:
            return word
    return "absent" if "cannot access kernel" in out or "404" in out else "unknown"
