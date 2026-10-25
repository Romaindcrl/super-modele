"""Entraîne le GPT taille GPT-2 sur Kaggle, session après session, et rapatrie chaque modèle dans checkpoints/.

    caffeinate -is uv run kaggle_watch.py          # le Mac reste éveillé tant que le suivi tourne

1. envoie les données (`data/kaggle/upload/`, `gpt.py`, `kaggle/*.py`) en dataset Kaggle privé ;
2. lance en parallèle la préparation du corpus (CPU) et l'essai sur TPU (vitesse de chaque taille) ;
3. choisit la taille d'après l'essai (GPT-2 medium si le TPU va assez vite), lance la session 1, puis la 2 ;
   chaque session livre un modèle affiné, téléchargé dans checkpoints/.

L'état est écrit dans `data/kaggle/status.json`, que l'interface (`app.py`) affiche. On peut arrêter et
relancer ce script à tout moment : il reprend où il en était. Demande la clé d'API Kaggle dans
KAGGLE_API_TOKEN, dans .env (kaggle.com → Settings → API Tokens).
"""

import json
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # comme app.py : le paquet, sans dépendre de l'installation

from super_modele.kaggle_cli import KAGGLE, OWNER as USER, env, kaggle  # noqa: E402,F401

DATASET = f"{USER}/super-modele-donnees"
STEPS = ["dialogue", "tpusmoke", "gpumod", "tpumod"]  # corpus (CPU), essai TPU, puis la recette modded sur GPU et TPU
HOURS = {"dialogue": 1.0, "tpusmoke": 0.5, "gpumod": 11.0, "tpumod": 8.6}  # durées attendues
NEEDS = {"dialogue": [], "tpusmoke": [], "gpumod": ["dialogue"], "tpumod": ["dialogue", "tpusmoke"]}
UPLOAD = ROOT / "data" / "kaggle" / "upload"
STATUS = ROOT / "data" / "kaggle" / "status.json"
POLL = 180  # secondes entre deux nouvelles de Kaggle


def load() -> dict:
    return json.loads(STATUS.read_text()) if STATUS.exists() else {"sessions": {}}


def save(state: dict, message: str) -> None:
    state["updated"] = datetime.now().isoformat(timespec="seconds")
    state["message"] = message
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    print(f"{datetime.now():%H:%M:%S} {message}", flush=True)


def upload_dataset(state: dict) -> None:
    """Le dataset privé : le code (à jour) et les données. Créé la première fois, nouvelle version ensuite."""
    shutil.copy(ROOT / "src" / "super_modele" / "gpt.py", UPLOAD / "gpt.py")
    for name in ("train_gpt2.py", "train_tpu.py", "prepare_data.py", "prepare_dialogue.py"):
        shutil.copy(ROOT / "kaggle" / name, UPLOAD / name)
    (UPLOAD / "dataset-metadata.json").write_text(json.dumps(
        {"title": "super-modele-donnees", "id": DATASET, "licenses": [{"name": "unknown"}]}, indent=2))
    if DATASET in kaggle("datasets", "list", "--mine", "--csv", check=False):
        kaggle("datasets", "version", "-p", str(UPLOAD), "-m", "code à jour", "-q")
    else:
        kaggle("datasets", "create", "-p", str(UPLOAD), "-q")  # privé par défaut
    for _ in range(60):  # Kaggle traite les fichiers avant de les rendre disponibles
        if "ready" in kaggle("datasets", "status", DATASET, check=False).lower():
            break
        time.sleep(10)
    state["dataset"] = True
    save(state, f"données envoyées sur Kaggle ({DATASET}, privé)")


def kernel_status(step: str) -> str:
    """queued, running, complete, error, cancel, absent ou unknown (Kaggle injoignable), d'après `kaggle kernels status`."""
    from super_modele.kaggle_cli import kernel_status as status

    return status(f"super-modele-{step}")


def no_tpu(step: str) -> bool:
    """Vrai si la session a échoué faute de TPU (la CLI lance parfois sur la machine CPU/GPU)."""
    dest = ROOT / "data" / "kaggle" / f"{step}_log"
    kaggle("kernels", "output", f"{USER}/super-modele-{step}", "-p", str(dest), "-o", "-q", "--file-pattern", r".*\.log$",
           check=False)
    log = " ".join(f.read_text(errors="ignore") for f in dest.glob("*.log"))
    return "torch_xla" in log and ("No module named" in log or "assert xr.device_type" in log or "AssertionError" in log)


def download(step: str) -> Path:
    """Les petits fichiers de sortie (modèle affiné, score, exemples, mesures, journal) ; pas le corpus."""
    dest = ROOT / "data" / "kaggle" / step
    dest.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{USER}/super-modele-{step}", "-p", str(dest), "-o", "-q", "--file-pattern",
           r"(checkpoints/dialogues.*|.*resultat\.json|.*exemples\.txt|bench\.json|data/stats\.json|.*\.log)$")
    for f in (dest / "checkpoints").glob("dialogues_*"):
        shutil.copy(f, ROOT / "checkpoints" / f.name)  # l'interface les trouve là
    return dest


def choose_size(state: dict) -> None:
    """GPT-2 medium si l'essai montre qu'il apprend assez de tokens en 2 sessions, sinon GPT-2 small."""
    bench = json.loads((ROOT / "data" / "kaggle" / "essai" / "bench.json").read_text())["bench"]
    ok = {(b["model"], b["micro_batch"]): b["tokens_per_s_per_core"] * 8 for b in bench if b.get("ok")}
    medium = max(((m, v) for (name, m), v in ok.items() if name == "medium"), key=lambda x: x[1], default=None)
    # 15 h × tokens/s : medium (320 M) mérite ~6 G tokens (Chinchilla) ; il les voit dès ~110 k tokens/s
    model, micro = ("medium", medium[0]) if medium and medium[1] >= 110_000 else ("small", 16)
    state["model"] = {"name": model, "micro_batch": micro, "tokens_per_s": ok.get((model, micro))}
    for step in ("s1", "s2"):
        nb = ROOT / "kaggle" / "kernels" / step / "notebook.ipynb"
        text = re.sub(r"MODEL=\S+ MICRO_BATCH=\S+", f"MODEL={model} MICRO_BATCH={micro}", nb.read_text())
        nb.write_text(text.replace("__MODEL__", model).replace("__MICRO__", str(micro)))
    save(state, f"essai TPU : {', '.join(f'{k[0]}×{k[1]} {v:,.0f} tokens/s' for k, v in ok.items())} → GPT-2 {model}")


def main() -> None:
    state = load()
    if not state.get("dataset"):
        upload_dataset(state)
    while True:
        for step in STEPS:
            info = state["sessions"].setdefault(step, {"status": "en attente"})
            ready = all(state["sessions"].get(n, {}).get("status") == "complete" for n in NEEDS[step])
            if info["status"] == "en attente" and ready:
                out = kaggle("kernels", "push", "-p", str(ROOT / "kaggle" / "kernels" / step))
                info.update(status="queued", pushed=datetime.now().isoformat(timespec="seconds"), hours=HOURS[step],
                            url=next(iter(re.findall(r"https://\S+", out)), ""))
                save(state, f"{step} lancé sur Kaggle")
            elif info["status"] == "clic":  # attend une relance depuis l'éditeur Kaggle, avec le TPU
                if kernel_status(step) in ("queued", "running"):
                    info["status"] = "queued"
                    save(state, f"{step} relancé depuis l'éditeur")
            elif info["status"] in ("queued", "running"):
                status = kernel_status(step)
                if status == "unknown":  # Kaggle injoignable : on réessaiera au tour suivant
                    continue
                if status == "error" and step.startswith(("s", "essai", "tpu")) and no_tpu(step):
                    info["status"] = "clic"
                    save(state, f"{step} : Kaggle l'a lancé sans TPU — ouvrir https://www.kaggle.com/code/{USER}/"
                                f"super-modele-{step}/edit puis Save Version → Save & Run All")
                    continue
                if status != info["status"]:
                    info["status"] = status
                    if status == "running":
                        info["started"] = datetime.now().isoformat(timespec="seconds")
                    if status in ("complete", "error", "cancel"):
                        info["finished"] = datetime.now().isoformat(timespec="seconds")
                        try:
                            dest = download(step)
                            result = next(dest.glob("**/resultat.json"), None)
                            if result is not None and step not in ("prep", "dialogue"):
                                info["result"] = json.loads(result.read_text())
                        except RuntimeError as error:
                            info["download_error"] = str(error)[-300:]
                    save(state, f"{step} : {status}")
        statuses = {step: state["sessions"][step]["status"] for step in STEPS}

        def blocked(step: str) -> bool:  # une étape dont un prérequis a échoué ne partira jamais
            return any(statuses[n] in ("error", "cancel") or blocked(n) for n in NEEDS[step])

        alive = [s for s in STEPS if statuses[s] in ("queued", "running", "clic") or (statuses[s] == "en attente" and not blocked(s))]
        if not alive:  # plus rien ne tourne ni ne peut partir : fin du suivi
            save(state, "arrêt : " + ", ".join(f"{k} {v}" for k, v in statuses.items()))
            return
        state["checked"] = datetime.now().isoformat(timespec="seconds")
        STATUS.write_text(json.dumps(state, indent=2, ensure_ascii=False))
        time.sleep(POLL)


if __name__ == "__main__":
    sys.exit(main())
