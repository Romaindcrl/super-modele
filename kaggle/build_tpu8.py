"""Construit kaggle/kernels/tpu8 : le pré-entraînement du run v6-591M sur le TPU v5e-8 de Kaggle (8 puces, un
processus par puce), repris de la sauvegarde du dataset super-modele-etat.

    python kaggle/build_tpu8.py --limit-h 8.3 --target 12e9

Entrées : super-modele-code (le code figé du run, le tokenizer, les Simpson), super-modele-etat (l'état), les sorties
super-modele-prep et super-modele-dialogue (le corpus). Sortie : run/pretrain_state.pt et run/info.json (pas,
tokens, date), que la machine suivante (Kaggle ou Colab) reprend ; run/train.log.

Un fichier STOP est posé après --limit-h heures (la session TPU de Kaggle s'arrête à 9 h) : l'entraînement s'arrête
proprement au pas multiple de 10 suivant et sauvegarde son état. L'entraînement repart avec la cible de la sauvegarde
(--resume-target, là où en est le planning du learning rate), puis la nouvelle (--target) est écrite dans le fichier
de budget qu'il relit tous les 10 pas : le learning rate remonte par paliers (LR_RISE) au lieu de sauter.
"""

import argparse
import json
from pathlib import Path

KERNELS = Path(__file__).resolve().parent / "kernels"


def notebook(limit_h: float, target: float, resume_target: float) -> dict:
    cells = []

    def code(text: str) -> None:
        cells.append({"cell_type": "code", "id": f"c{len(cells)}", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": text.strip()})

    cells.append({"cell_type": "markdown", "id": "m0", "metadata": {},
                  "source": "# Super modèle v6-591M · pré-entraînement sur le TPU v5e-8 de Kaggle"})
    code(f'''
import glob, json, os, re, subprocess, threading, time
T0 = time.time()
LIMIT_H, TARGET, RESUME_TARGET = {limit_h}, {target:.0f}, {resume_target:.0f}


def sh(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"$ {{cmd[:160]}}\\n{{(p.stdout + p.stderr)[-1500:]}}", flush=True)
    if p.returncode:
        raise RuntimeError(f"échec : {{cmd}}")


print({{k: v for k, v in os.environ.items() if k.startswith(("TPU", "XLA", "PJRT", "CLOUD_TPU"))}}, flush=True)
''')
    code('''
# le venv du run, comme sur Colab : jax 0.7.2 et sa libtpu 0.0.23 (noyaux Pallas de l'attention flash), torch 2.9 et
# torch_xla 2.9 sans l'extra [tpu] ; hors de /kaggle/working (pas dans la sortie)
VENV = "/root/jaxenv"
PY = f"{VENV}/bin/python"
sh("pip install -q uv")
sh(f"python3 -m uv venv --system-site-packages {VENV}")
UV = f"python3 -m uv pip install -q --python {PY}"
sh(UV + " 'jax[tpu]==0.7.2'")
sh(UV + " torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu")
sh(UV + " torch_xla==2.9.0 tokenizers safetensors")
# la topologie que Kaggle fixe pour un seul processus (tpu_process_addresses="local") : xmp.spawn pose la sienne pour
# 8 processus, avec setdefault ; gardées, elles font échouer le démarrage (« Expected 8 worker addresses, got 1 »)
CLEAN = ("TPU_PROCESS_ADDRESSES", "TPU_PROCESS_PORT", "TPU_VISIBLE_CHIPS", "TPU_CHIPS_PER_PROCESS_BOUNDS",
         "TPU_PROCESS_BOUNDS", "CLOUD_TPU_TASK_ID")
ENV = {k: v for k, v in os.environ.items() if k not in CLEAN}
ENV["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
p = subprocess.run([PY, "-c", "import torch, torch_xla, jax, torch_xla.runtime as xr; print('ENV_OK', torch.__version__, "
                    "torch_xla.__version__, jax.__version__, xr.device_type(), xr.global_runtime_device_count())"],
                   capture_output=True, text=True, env=ENV)
print(p.stdout[-400:], p.stderr[-1500:], flush=True)
assert "ENV_OK" in p.stdout and p.stdout.strip().endswith("TPU 8"), "pas de TPU v5e-8 : session sans TPU"
''')
    code('''
CODE = next(os.path.dirname(f) for f in glob.glob("/kaggle/input/**/train_tpu.py", recursive=True)
            if "LR_RISE" in open(f).read())
print("code :", CODE)
print("corpus :", sorted(glob.glob("/kaggle/input/**/*_train.bin", recursive=True)), flush=True)
OUT = "/kaggle/working/run"
os.makedirs(OUT, exist_ok=True)
# l'état le plus récent des entrées (la sauvegarde de Colab, ou la sortie de la session Kaggle précédente), d'après
# l'info.json écrit à côté ; lié dans OUT, que train_tpu.py préfère (RESUME_DIR, trié après /kaggle/input)
states = []
for f in glob.glob("/kaggle/input/**/pretrain_state.pt", recursive=True):
    try:
        states.append((json.load(open(os.path.join(os.path.dirname(f), "info.json"))), f))
    except (OSError, ValueError):
        print("sans info.json :", f)
info, state = max(states, key=lambda s: s[0]["mtime"])
print("reprise :", state, info, flush=True)
os.symlink(state, f"{OUT}/pretrain_state.pt")
# là où en est le planning du learning rate (WSD : descente linéaire sur les 40 % de la fin) : la cible pour laquelle
# il donne le dernier learning rate de la session précédente, sinon celle donnée au notebook
if info.get("lr") is not None and info.get("tokens"):
    RESUME_TARGET = info["target"] if info["lr"] >= 1 else info["tokens"] / (1 - 0.4 * info["lr"])
# une sauvegarde déjà à sa cible (pré-entraînement fini) : 0 pas prévu, et la nouvelle cible, relue tous les 10 pas
# dans la boucle, jamais lue ; 20 pas de marge pour y entrer
RESUME_TARGET = max(RESUME_TARGET, info.get("tokens", 0) + 20 * 524288)
print(f"cible à la reprise {RESUME_TARGET / 1e9:.2f} G, puis {TARGET / 1e9:.2f} G tokens", flush=True)
with open(f"{OUT}/pretrain_gpt2_modded.tokens", "w") as f:  # relu tous les 10 pas : la nouvelle cible
    f.write(str(int(TARGET)))


def watchdog():
    while time.time() - T0 < LIMIT_H * 3600:
        time.sleep(20)
    open(f"{OUT}/STOP", "w").close()
    print(f"STOP posé après {LIMIT_H} h", flush=True)


threading.Thread(target=watchdog, daemon=True).start()
RUN = {**ENV, "INPUT_DIR": CODE, "OUT": OUT, "RESUME_DIR": OUT, "ARCH": "modern", "OPTIM": "normuon", "MODEL": "w1536",
       "ATTN_KERNEL": "flash", "QKV_LAYOUT": "1", "ACCUM_MODE": "loop", "FT_ACCUM_MODE": "loop", "FT_OPTIM": "normuon",
       "TOKENS_PER_STEP": "524288", "MIX": "subs:0.5,web:0.3,wiki:0.2", "SESSION": "1", "EVAL_EVERY_S": "600",
       "SAVE_EVERY_S": "1800", "FT_TOKENS": "100000000", "HOURS_PRETRAIN": "1000", "TOTAL_PRETRAIN_HOURS": "1000",
       "HOURS_FINETUNE": "100", "MICRO_BATCH": "8", "PRETRAIN_TOKENS": str(int(RESUME_TARGET)), "CORPUS_CAP": "0",
       "MALLOC_MMAP_THRESHOLD_": "1048576"}
log = open(f"{OUT}/train.log", "a")
p = subprocess.Popen([PY, "-u", f"{CODE}/train_tpu.py", "train"], cwd=OUT, env=RUN, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    log.write(line)
    log.flush()
    if not any(s in line for s in ("Warning", "warn(", "SliceBuilder", "mark_step", "DeprecationWarning")):
        print(line, end="", flush=True)
print("code de sortie :", p.wait(), "· après", round((time.time() - T0) / 3600, 2), "h", flush=True)
''')
    code('''
st = f"{OUT}/pretrain_state.pt"
if os.path.exists(st) and not os.path.islink(st):  # un état écrit ici (pas le lien vers l'entrée) : pour la suite,
    # sa version, son pas, ses tokens, la cible et le dernier learning rate (le planning reprend là)
    read = ("import json, sys, numpy as np, torch\\n"
            "allowed = [np._core.multiarray.scalar, np.dtype] + [type(np.dtype(t)) for t in ('float32', 'float64', 'int64')]\\n"
            "with torch.serialization.safe_globals(allowed):\\n"
            "    s = torch.load(sys.argv[1], map_location='cpu', weights_only=True, mmap=True)\\n"
            "print(json.dumps({'step': int(s['step']), 'tokens': int(s['tokens'])}))")
    q = subprocess.run([PY, "-c", read, st], capture_output=True, text=True, env=ENV)
    lrs = re.findall(r"pretrain_gpt2_modded .*?lr ×([\\d.]+)", open(f"{OUT}/train.log").read())
    info = {"mtime": os.path.getmtime(st), "size": os.path.getsize(st), **json.loads(q.stdout.strip().splitlines()[-1]),
            "target": int(open(f"{OUT}/pretrain_gpt2_modded.tokens").read()), "lr": float(lrs[-1]) if lrs else None}
    json.dump(info, open(f"{OUT}/info.json", "w"))
    print(info)
elif os.path.islink(st):
    os.remove(st)  # pas d'état écrit : le lien vers l'entrée n'a rien à faire dans la sortie
for f in (f"{OUT}/STOP",):
    if os.path.exists(f):
        os.remove(f)
sh("ls -la /kaggle/working/run /kaggle/working/run/checkpoints 2>/dev/null | head -30; du -sh /kaggle/working")
''')
    return {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit-h", type=float, default=8.3, help="STOP posé après ce temps (session TPU : 9 h)")
    parser.add_argument("--target", type=float, default=12e9, help="tokens visés pour le pré-entraînement")
    parser.add_argument("--resume-target", type=float, default=6.62e9, help="la cible de la sauvegarde reprise")
    parser.add_argument("--slug", default="super-modele-tpu8", help="le noyau Kaggle (deux, en alternance)")
    parser.add_argument("--after", help="le noyau de la session précédente : sa sortie en entrée (son état)")
    args = parser.parse_args()
    OUT = KERNELS / args.slug.removeprefix("super-modele-")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "notebook.ipynb").write_text(json.dumps(notebook(args.limit_h, args.target, args.resume_target), indent=1,
                                                   ensure_ascii=False))
    (OUT / "kernel-metadata.json").write_text(json.dumps({
        "id": f"romaindl/{args.slug}", "title": args.slug, "code_file": "notebook.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": True,
        "machine_shape": "TpuV5E8", "enable_internet": True,
        "dataset_sources": ["romaindl/super-modele-code", "romaindl/super-modele-etat"],
        "kernel_sources": ["romaindl/super-modele-prep", "romaindl/super-modele-dialogue"]
                          + ([f"romaindl/{args.after}"] if args.after else []), "competition_sources": []},
        indent=2))
    print(f"{OUT} : STOP après {args.limit_h} h, cible {args.target / 1e9:.2f} G tokens (reprise à {args.resume_target / 1e9:.2f})")


if __name__ == "__main__":
    main()
