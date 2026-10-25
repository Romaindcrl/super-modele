"""Pré-entraîne un GPT sur un TPU de Colab, puis l'affine sur les Simpson, et le garde en vie jusqu'au bout (CLI
`colab`, crédits Colab). Moteur : ENGINE (jax : train_jax.py, torch : train_tpu.py).

    caffeinate -is uv run colab_watch.py v6     # reprend la machine `v6` déjà réservée

Toutes les 4 minutes : recopie le journal de la machine dans data/colab/run.log (le dashboard le lit), vérifie
que l'entraînement tourne et que la dépense reste sous le plafond (le solde du départ moins MARGIN unités), et
rapatrie les poids de chaque nouvel état pour le chat (dernier checkpoint). Les sauvegardes de l'état (poids +
optimiseurs) partent de la machine vers Kaggle sans passer par le Mac : une boucle sur la machine envoie chaque nouvel
état, Mac allumé ou non. Toutes les heures : ajuste le nombre de tokens visé à ce que le budget permet, à la vitesse
mesurée (tant qu'on est sur le plateau du planning, ça ne change rien d'autre). Tant qu'on est sur un v5e : demande un
v6e-1 à chaque tour (~toutes les 2 minutes) ; s'il est accordé, l'entraînement s'arrête proprement (fichier STOP),
l'état passe sur le v6e par Kaggle et la v5e est libérée. Si la machine disparaît, on en reprend une et on repart de la dernière
sauvegarde. À la fin : le modèle affiné, son score et ses exemples sont rapatriés dans checkpoints/, et la machine est
libérée. Pour relancer le suivi (nouveau code) : créer data/colab/restart ; il s'arrête entre deux tours (jamais au
milieu d'une demande de machine) et run.sh le relance.

Le corpus (15 Go) est téléchargé par chaque machine depuis les sorties Kaggle, avec la clé de .env, envoyée dans
un fichier (jamais affichée). État du suivi : data/colab/state.json ; ce qu'il fait : data/colab/watch.log.
"""

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from super_modele.kaggle_cli import OWNER, env  # noqa: E402

DIR = ROOT / "data" / "colab"
LOG = DIR / "run.log"  # le journal de l'entraînement, recopié (pour le dashboard)
EVENTS = DIR / "watch.log"  # ce que fait le suivi
STATE = DIR / "state.json"
BACKUP = DIR / "state"  # la dernière sauvegarde de l'état, en morceaux
MARGIN = 5.0  # unités gardées en réserve : le plafond de dépense est le solde du départ moins cette marge
# secondes ; SYNC : le journal, pour le dashboard ; HUNT : la pause entre deux tours sur un v5e, où chaque tour demande
# un v6e (une demande refusée ne coûte rien ; avec l'attente d'une réponse tardive, ~1 demande toutes les 2 min)
SYNC, POLL, HUNT, BACKUP_EVERY = 60, 240, 10, 3600
# L'état complet (4,8 Go, poids + optimiseur) est écrit toutes les 30 min (SAVE_EVERY_S) ; une boucle sur la machine
# (KAGGLE_LOOP_SCRIPT) l'envoie aussitôt dans un dataset Kaggle privé, et la machine suivante l'y reprend (comme le
# corpus) : ni le Mac, ni sa connexion, ni le suivi ne sont sur le chemin. La copie sur le Mac : seulement en secours.
KAGGLE_STATE = f"{OWNER}/super-modele-etat"
CHUNK = 40 * 2**20
TRANSFERS = 4  # morceaux envoyés ou rapatriés en même temps (mesuré : 4,3 s par morceau à 6 contre 8,6 s un par un)
ENGINE = "torch"  # mesuré sur le v6e : PyTorch + têtes de 128 + attention flash, le plus rapide (JAX 103,4k au mieux en medium)
REMOTE = "/content/jrun" if ENGINE == "jax" else "/content/run"  # le dossier de l'entraînement sur la machine
JAXENV = "/content/jaxenv"  # jax 0.7.2 avec sa libtpu (celle de torch_xla est trop ancienne pour la splash attention)
SCRIPT, PYTHON, STATE_FILE, STAGE = (("train_jax.py", f"{JAXENV}/bin/python", "pretrain_state.safetensors", "pretrain_jax")
                                     if ENGINE == "jax" else
                                     ("train_tpu.py", f"{JAXENV}/bin/python", "pretrain_state.pt", "pretrain_gpt2_modded"))
# PyTorch passe aussi par le venv de JAX : sa libtpu 0.0.23 accepte les noyaux Pallas (attention flash) que JAX 0.7.2
# compile pour torch_xla ; celle livrée avec torch_xla 2.9 (0.0.21) les refuse.
# Le code du run, figé dans data/colab/frozen : une relance renvoie toujours la version lancée, même si
# kaggle/ ou src/ changent entre-temps (essais de vitesse).
FILES = ["data/colab/frozen/gpt.py", f"data/colab/frozen/{SCRIPT}", "data/kaggle/upload/tokenizer.json",
         "data/kaggle/upload/simpsons_train.txt", "data/kaggle/upload/simpsons_val.txt",
         "data/kaggle/upload/simpsons_test.txt"]
COMMON = {"TORCH_DEVICE_BACKEND_AUTOLOAD": "0",  # l'extension torch_tpu de l'image Colab : incompatible avec torch 2.9
          # les blocs de plus d'1 Mo toujours rendus au système une fois libérés : sinon glibc relève son seuil après la
          # première sauvegarde, et la copie de l'état (4,8 Go) des suivantes reste dans le tas (mesuré le 22/10 sur v5e :
          # tas de 4,08 Go après la 3e sauvegarde, 6,5 Go de mémoire en moins pour le corpus)
          "MALLOC_MMAP_THRESHOLD_": "1048576",
          "TOKENS_PER_STEP": "524288", "MIX": "subs:0.5,web:0.3,wiki:0.2", "SESSION": "1", "OUT": REMOTE,
          "RESUME_DIR": "/content", "EVAL_EVERY_S": "600", "SAVE_EVERY_S": "1800", "FT_TOKENS": "100000000",
          "HOURS_PRETRAIN": "1000", "TOTAL_PRETRAIN_HOURS": "1000", "HOURS_FINETUNE": "100"}
RUN = COMMON | ({"INPUT": "/content/sm", "JAX_CACHE": "/content/jax_cache", "N_LAYER": "24", "N_EMBD": "1024", "N_HEAD": "16",
                 "REMAT": "none", "FT_OPTIM": "adamw"}
                if ENGINE == "jax" else
                {"ARCH": "modern", "OPTIM": "normuon", "MODEL": "w1536", "ATTN_KERNEL": "flash", "QKV_LAYOUT": "1",  # +17 % (95,9k contre 81,9k au banc)
                 "ACCUM_MODE": "loop",
                 "FT_ACCUM_MODE": "loop",
                 "FT_OPTIM": "normuon"})  # affinage NorMuon (choix de l'utilisateur ; AdamW 1,229 contre 1,243 sur le 124 M)
# v5e autorisé (choix de l'utilisateur, 23/10 16:15) : un v6e s'il y en a un, sinon un v5e, avec le correctif (corpus
# de 4 Go, mémoire rendue après chaque sauvegarde) ; sur un v5e, le suivi demande un v6e à chaque tour et y passe.
# False : le suivi n'attend qu'un v6e, sans rien dépenser en attendant.
ALLOW_V5E = True
# Séquences par micro-lot, selon la mémoire de la puce : sur un v5e (16 Go), 16 dépasse de 0,2 Go pour GPT-2 medium.
# 20 × 1536 sur v6e : 82 000 tokens/s à 16 (pic de 11,1 Go de HBM), 73 000 à 32. Le v5e a 16 Go : 16 devrait tenir ;
# sinon (mémoire du TPU dépassée au lancement) le suivi relance avec la moitié (state["micro_batch"])
MICRO_BATCH = {"V5E1": "8", "V6E1": "16"}  # v5e à 16 : 9,96 Go demandés, 9,04 libres (22 octobre)
# Tokens du corpus gardés : une machine v5e (47 Go de RAM, dont ~36 pris par l'entraînement et le TPU) ne garde pas les
# 15,5 Go du corpus en cache, et l'entraînement attend le disque. Mesuré le 22/10 : ~12 Go de cache, où chaque page
# lue compte deux fois (/content sur un périphérique loop sans accès direct) ; 4 G tokens (8 Go) n'y restaient pas,
# 2 G tokens (4 Go, deux fois plus que ce qu'un v5e entraîne avec le budget restant) y tiennent. Un v6e (172 Go) garde tout.
CORPUS_CAP = {"V5E1": "2000000000"}
TOKENS_MIN, TOKENS_MAX = 2e9, 15e9  # le corpus fait 7,7 G tokens : deux passages au plus
VERSION = "v6-591M"  # le nom du modèle dans checkpoints/ (vX-YM : n° du pré-entraînement, taille)
RATE = {"V6E1": 4.08, "V5E1": 2.92}  # unités par heure de chaque machine (colab usage)
TOKENS_START = 11e9  # la première cible, ajustée chaque heure au solde et à la vitesse mesurée


def set_phase(state: dict, phase: str) -> None:
    """Ce que fait le suivi (le dashboard l'affiche) : attente d'un TPU, préparation, entraînement, terminé, arrêté."""
    if state.get("phase") != phase:
        state.update(phase=phase, phase_since=datetime.now().isoformat(timespec="seconds"))
        save(state)


def note(message: str) -> None:
    line = f"{datetime.now():%H:%M:%S} {message}"
    print(line, flush=True)
    DIR.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def colab(*args: str, timeout: float = 600, stdin: str | None = None) -> str:
    """Une commande `colab` ; sa sortie (texte), ou "" si elle dépasse `timeout`."""
    try:
        done = subprocess.run(["colab", *args], input=stdin, capture_output=True, text=True, timeout=timeout)
        return done.stdout + done.stderr
    except subprocess.TimeoutExpired:
        return ""


def vm(session: str, code: str, timeout: float = 600) -> str:
    """Exécute du Python sur la machine (dans son noyau Jupyter) et renvoie ce qu'il affiche."""
    return colab("exec", "-s", session, "--timeout", str(timeout), stdin=code, timeout=timeout + 60)


def vm_json(session: str, code: str, timeout: float = 600) -> dict | None:
    m = re.search(r"^\{.*\}$", vm(session, code, timeout), re.M)
    return json.loads(m[0]) if m else None


def load() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save(state: dict) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def machines() -> list[dict] | None:
    """Les machines du compte : [{name ("?" sans session locale), endpoint, hardware}], None si Colab ne répond pas."""
    out = colab("sessions", timeout=120)
    if "Hardware:" not in out and "No active sessions found on server" not in out:
        return None  # pas de réponse claire de Colab (réseau…) : on ne conclut rien
    return [{"name": m[1], "endpoint": m[2], "hardware": m[3]}
            for m in re.finditer(r"\[(\S+)\] (\S+) \| Hardware: (\S+)", out)]


def hardware_of(session: str) -> str:
    return next((m["hardware"] for m in machines() or [] if m["name"] == session), "")


def adopt(endpoints: set[str]) -> dict[str, str]:
    """Rattache à l'outil les machines de `endpoints` sans session locale : {nom donné : matériel}."""
    py = Path.home() / ".local/share/uv/tools/google-colab-cli/bin/python"
    script = DIR / "adopt.py"
    script.write_text(
        "import sys\nfrom colab_cli.common import state\nfrom colab_cli.state import SessionState\n"
        "wanted = set(sys.argv[1:])\nknown = {s.endpoint for s in state.store.list().values()}\n"
        "for a in state.client.list_assignments():\n"
        "    if a.endpoint in known or a.endpoint not in wanted: continue\n"
        "    name = a.accelerator.value.lower() + '-' + a.endpoint[-5:]\n"
        "    i = a.runtime_proxy_info\n"
        "    state.store.add(SessionState(name=name, token=i.token, url=i.url, endpoint=a.endpoint,"
        " token_expires_at=i.expires_at(), variant=a.variant.name, accelerator=a.accelerator.value,"
        " machine_shape='STANDARD'))\n"
        "    print(name, a.accelerator.value)\n")
    out = subprocess.run([str(py), str(script), *endpoints], capture_output=True, text=True).stdout
    return dict(line.split()[:2] for line in out.splitlines() if len(line.split()) >= 2)


def balance() -> tuple[float | None, float | None]:
    """(solde, consommation par heure) du compte Colab."""
    out = colab("usage", timeout=120)
    b, r = re.search(r"Current balance: ([\d.]+)", out), re.search(r"Usage rate: ([\d.]+)", out)
    return (float(b[1]) if b else None), (float(r[1]) if r else None)


def send_token(session: str) -> None:
    """La clé Kaggle de .env, dans un fichier de la machine (jamais affichée, ni passée en argument)."""
    with tempfile.NamedTemporaryFile("w", delete=False, prefix="k") as f:  # dossier temporaire du système
        f.write(env()["KAGGLE_API_TOKEN"])
        local = f.name
    try:
        vm(session, "import os; os.makedirs('/root/.kaggle', exist_ok=True)")
        colab("upload", "-s", session, local, "/root/.kaggle/token")
        vm(session, "import os; os.chmod('/root/.kaggle/token', 0o600)")
    finally:
        os.remove(local)


# Le corpus attendu sur la machine (Go) : un fichier plus petit est un téléchargement interrompu
CORPUS_SIZES = {"wiki_train.bin": 3.86, "web_train.bin": 7.0, "subs_train.bin": 4.57,
                "wiki_val.bin": 0.02, "web_val.bin": 0.04, "subs_val.bin": 0.02}


def setup(session: str, with_restore: bool = False, state: dict | None = None) -> bool:
    """L'environnement (vérifié sur le TPU avant tout le reste), le code, la clé Kaggle et le corpus (téléchargé par la
    machine depuis les sorties Kaggle, réessayé tant qu'il manque des octets). False : machine inutilisable."""
    # train_tpu.py cherche le corpus sous /content : aucun autre *_train.bin ne doit traîner (restes d'un affinage)
    vm(session, "import os, shutil; [shutil.rmtree(d, ignore_errors=True) for d in ('/content/sm/replay', '/content/sm/ft')]; "
                f"[os.makedirs(d, exist_ok=True) for d in ('/content/sm', '{REMOTE}', '/content/data')]")
    # Le venv du run : jax 0.7.2 et sa libtpu 0.0.23 (les noyaux Pallas), plus torch 2.9 et torch_xla 2.9 si l'image de
    # Colab ne les a pas (depuis le 20 octobre, ses TPU arrivent avec une autre pile PyTorch, sans torch_xla). torch_xla
    # sans l'extra [tpu] : il ne touche pas à la libtpu du venv.
    out = vm(session, f'''
import os, subprocess
def sh(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (p.stdout + p.stderr)[-600:]
# TORCH_DEVICE_BACKEND_AUTOLOAD=0 : notre torch 2.9 ne charge pas l'extension torch_tpu de l'image (incompatible)
py, uv = "TORCH_DEVICE_BACKEND_AUTOLOAD=0 {JAXENV}/bin/python", "python3 -m uv pip install -q --python {JAXENV}/bin/python"
if not os.path.exists("{JAXENV}/bin/python"):
    print(sh("pip install -q uv && python3 -m uv venv --system-site-packages {JAXENV} && " + uv + " 'jax[tpu]==0.7.2'"))
check = py + " -c 'import torch, torch_xla, tokenizers, safetensors, numpy; print(torch.__version__, torch_xla.__version__)'"
if subprocess.run(check, shell=True, capture_output=True).returncode:
    print(sh(uv + " torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu"))
    print(sh(uv + " torch_xla==2.9.0 tokenizers safetensors"))
print(sh(py + " -c 'import torch, torch_xla, jax, torch_xla.core.xla_model as xm; "
           "print(\\"ENV_OK\\", torch.__version__, torch_xla.__version__, jax.__version__, xm.xla_device())'"))
''', timeout=1800)
    if "ENV_OK" not in out:
        note(f"{session} : environnement inutilisable : {out.strip()[-400:]}")
        return False
    note(f"{session} : environnement prêt : {out[out.index('ENV_OK'):].splitlines()[0][:120]}")
    for f in FILES:
        colab("upload", "-s", session, str(ROOT / f), f"/content/sm/{Path(f).name}")
    send_token(session)
    note(f"{session} : code et clé envoyés, téléchargement du corpus")
    # En arrière-plan sur la machine (une coupure de la connexion de Colab ne l'arrête pas) : les trois parties en
    # parallèle, chacune relancée si rien n'arrive pendant 15 min ; le suivi lit l'avancement chaque minute.
    kb = (state or {}).get("kaggle_backup") or {}
    from_kaggle = with_restore and kb.get("state") == "ready"  # toujours plus récente que la copie du Mac (secours)
    restored = {"ok": True}
    sender = None
    if with_restore and not from_kaggle and (BACKUP / STATE_FILE).exists():  # secours : la copie du Mac
        def send_backup() -> None:
            try:
                restore(session)
            except Exception as error:  # empreinte différente, transfert coupé…
                restored["ok"] = False
                note(f"{session} : restauration ratée : {error}")
        sender = threading.Thread(target=send_backup, daemon=True)
        sender.start()
    if from_kaggle:
        note(f"{session} : l'état du pas {kb.get('step')} viendra de Kaggle ({KAGGLE_STATE}), avec le corpus")
        state["backup_step"] = kb.get("step") or state.get("backup_step")
    ok = fetch_data(session, kb if from_kaggle else None, state)
    if ok and sender is not None:  # corpus raté : inutile d'attendre la fin du transfert
        sender.join()
    return ok and restored["ok"]


def fetch_data(session: str, kaggle_state: dict | None = None, state: dict | None = None) -> bool:
    """Le corpus (et l'état sauvegardé sur Kaggle, si `kaggle_state` : cette version ou une plus récente) téléchargés
    par la machine elle-même, en arrière-plan (une coupure de la connexion de Colab ne l'arrête pas) : chaque partie
    dans son dossier, en parallèle, relancée si rien n'arrive pendant 15 min ; le suivi lit l'avancement chaque
    minute. Le pas de l'état repris est noté dans `state` (backup_step)."""
    expected = dict(CORPUS_SIZES)
    if kaggle_state:
        expected["_state"] = {"dataset": KAGGLE_STATE, "mtime": kaggle_state["mtime"], "size": kaggle_state["size"]}
    vm(session, "import os\nfor f in ('/content/corpus_status.json',):\n    os.path.exists(f) and os.remove(f)\n"
                "open('/content/kaggle_fetch.py', 'w').write(" + repr(KAGGLE_FETCH_SCRIPT) + ")\n"
                "open('/content/corpus.py', 'w').write(" + repr(CORPUS_SCRIPT.replace("EXPECTED", json.dumps(expected))) + ")\n"
                "import subprocess; subprocess.Popen(['python3', '/content/corpus.py'], start_new_session=True, "
                "stdout=open('/content/corpus.log', 'w'), stderr=subprocess.STDOUT)")
    t0, last = time.time(), ""
    # ~3 h de vérifications, comptées une à une : le temps où le Mac dort ne compte pas (la machine, elle, continue ;
    # le 23/10, 3 h d'horloge passées Mac endormi ont fait rendre une machine dont les données étaient peut-être prêtes)
    for _ in range(180):
        time.sleep(60)
        info = vm_json(session, "import json, os; p = '/content/corpus_status.json'; "
                                "print(open(p).read().strip() if os.path.exists(p) else json.dumps({'state': 'starting'}))", timeout=60)
        if not info:
            continue
        if info.get("state") in ("ok", "failed"):
            note(f"{session} : données {info['state']} en {round(time.time() - t0)} s : {info.get('sizes')} "
                 f"({info.get('restarts', 0)} relances)" + (f" · erreurs : {info['errors']}" if info.get("errors") else ""))
            if info["state"] == "ok" and kaggle_state and state is not None:
                # la version reprise : peut-être plus récente que la dernière connue du suivi (envoyée Mac éteint)
                got = vm_json(session, "print(open('/content/data/state/info.json').read().strip())", timeout=60) or {}
                if got.get("step"):
                    state["backup_step"] = got["step"]
                if got.get("mtime", 0) > kaggle_state["mtime"] + 1:
                    state["kaggle_backup"] = {"state": "ready", **got}
                save(state)
                note(f"{session} : état repris de Kaggle : pas {got.get('step', '?')}, "
                     f"écrit le {datetime.fromtimestamp(got.get('mtime', 0)):%d/%m à %H:%M}")
            return info["state"] == "ok"
        summary = f"{info.get('state')} {info.get('sizes')} relances {info.get('restarts', 0)}"
        if time.time() - t0 > 600 and summary != last and int(time.time() - t0) % 900 < 60:
            note(f"{session} : données en cours : {summary}")
        last = summary
    note(f"{session} : données toujours incomplètes après 180 vérifications (~3 h Mac réveillé)")
    return False


# Les sorties d'un notebook Kaggle (le corpus) : le client officiel lit le fichier entier en mémoire avant de l'écrire,
# sans reprise ; une connexion coupée à 97 % de web_train.bin (7 Go) faisait tout recommencer, et échouer encore.
# Ici : l'adresse de chaque fichier demandée à l'API Kaggle, le fichier écrit au fil de l'eau, et repris là où il
# s'était arrêté (en-tête Range) après une coupure, avec une adresse fraîche si l'ancienne a expiré.
KAGGLE_FETCH_SCRIPT = r'''
import os, re, sys, time
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
kernel, pattern, dest = sys.argv[1], re.compile(sys.argv[2]), sys.argv[3]
owner, slug = kernel.split("/")
api = KaggleApi()
api.authenticate()
def listing():
    with api.build_kaggle_client() as client:
        request = ApiListKernelSessionOutputRequest()
        request.user_name, request.kernel_slug = owner, slug
        response = client.kernels.kernels_api_client.list_kernel_session_output(request)
        return {f.file_name: f.url for f in response.files or [] if pattern.search(f.file_name)}
urls = listing()
if not urls:
    sys.exit(f"aucun fichier de {kernel} ne correspond à {pattern.pattern}")
os.makedirs(dest, exist_ok=True)
for name in sorted(urls):
    out = os.path.join(dest, os.path.basename(name))
    for attempt in range(40):
        have = os.path.getsize(out) if os.path.exists(out) else 0
        try:
            with requests.get(urls[name], headers={"Range": f"bytes={have}-"} if have else {}, stream=True, timeout=120) as r:
                if r.status_code == 416:  # déjà complet
                    break
                if r.status_code in (400, 401, 403, 404):  # adresse expirée : une nouvelle
                    urls = listing()
                    continue
                r.raise_for_status()
                if r.status_code == 206:
                    total = int(r.headers["Content-Range"].rsplit("/", 1)[1])
                    mode = "ab"
                else:  # 200 : le fichier entier depuis le début
                    total, mode, have = int(r.headers.get("Content-Length", 0)), "wb", 0
                with open(out, mode) as f:
                    for chunk in r.iter_content(8 << 20):
                        f.write(chunk)
            if os.path.getsize(out) >= total:
                print(f"{name} : {total} octets", flush=True)
                break
        except (requests.RequestException, OSError, ValueError, KeyError) as error:
            print(f"{name} : coupé à {os.path.getsize(out) if os.path.exists(out) else 0} octets ({type(error).__name__}), reprise", flush=True)
            time.sleep(5)
    else:
        sys.exit(f"{name} : toujours incomplet après 40 reprises")
with open(os.path.join(dest, ".complete"), "w") as f:  # tous les fichiers à la taille annoncée par le serveur
    f.write(" ".join(sorted(urls)))
'''


CORPUS_SCRIPT = r'''
import json, os, subprocess, time
import shutil
EXP = json.loads('EXPECTED')
STATE = EXP.pop("_state", None)  # l'état sauvegardé sur Kaggle : {dataset, mtime, size}
env = {**os.environ, "KAGGLE_API_TOKEN": open("/root/.kaggle/token").read().strip()}
subprocess.run("python3 -m pip install -q kaggle==2.2.4", shell=True, env=env)
PARTS = {"wiki": "romaindl/super-modele-prep", "web": "romaindl/super-modele-prep", "subs": "romaindl/super-modele-dialogue"}
if STATE:
    PARTS["state"] = STATE["dataset"]
def sizes():
    return {f: os.path.getsize(os.path.join(d, f)) / 1e9 for d, _, fs in os.walk("/content/data") for f in fs if f.endswith(".bin")}
def done(part):
    if part == "state":  # complet (la taille de son info.json), et la version attendue ou une plus récente
        try:
            info = json.load(open("/content/data/state/info.json"))
            return (os.path.getsize("/content/data/state/pretrain_state.pt") == info["size"]
                    and info["mtime"] >= STATE["mtime"] - 1)
        except (OSError, ValueError, KeyError):
            return False
    return os.path.exists(f"/content/data/{part}/.complete")  # écrite par le téléchargeur, tailles vérifiées
def start(part):
    """Chaque partie dans son propre dossier : deux téléchargements de la même sortie Kaggle (wiki et web) écrivent
    sinon les mêmes fichiers annexes et l'un des deux échoue. Sa sortie est gardée (corpus_<partie>.log)."""
    log = open(f"/content/corpus_{part}.log", "a")
    log.write(f"--- {time.strftime('%H:%M:%S')} départ\n"); log.flush()
    if part == "state":  # fichier par fichier (l'archive entière d'une version récente : 404) ; un état commencé est
        # repris, sauf si une autre version est arrivée entre-temps (son info.json a changé) : celui-là repart de zéro
        d = "/content/data/state"
        os.makedirs(f"{d}/check", exist_ok=True)
        cmd = (f"kaggle datasets download {PARTS[part]} -f info.json -p {d}/check -o && "
               f"{{ cmp -s {d}/check/info.json {d}/info.json || {{ rm -f {d}/pretrain_state.pt; cp {d}/check/info.json {d}/info.json; }}; }} && "
               f"kaggle datasets download {PARTS[part]} -f pretrain_state.pt -p {d}")
    else:  # un fichier commencé n'est pas effacé : le téléchargeur reprend là où il s'était arrêté
        os.makedirs(f"/content/data/{part}", exist_ok=True)
        cmd = f"python3 /content/kaggle_fetch.py {PARTS[part]} '{part}_(train|val)[.]bin$' /content/data/{part}"
    return subprocess.Popen(cmd, shell=True, env=env, start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
def received(pgid):
    """Octets lus par les processus du téléchargement (réseau compris) : le client Kaggle laisse le fichier à 0 octet
    jusqu'à la fin, sa taille ne dit rien de l'avancement."""
    total = 0
    for pid in (os.listdir("/proc") if os.path.isdir("/proc") else []):
        if not pid.isdigit():
            continue
        try:
            if int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[2]) != pgid:
                continue
            total += next(int(l.split()[1]) for l in open(f"/proc/{pid}/io") if l.startswith("rchar:"))
        except (OSError, ValueError, IndexError, StopIteration):
            pass
    return total
def status(state, restarts):
    tails = {p: open(f"/content/corpus_{p}.log").read()[-300:] for p in PARTS
             if state == "failed" and not done(p) and os.path.exists(f"/content/corpus_{p}.log")}
    json.dump({"state": state, "restarts": restarts, "sizes": {k: round(v, 2) for k, v in sizes().items()}, "errors": tails},
              open("/content/corpus_status.json.tmp", "w"))
    os.replace("/content/corpus_status.json.tmp", "/content/corpus_status.json")
os.makedirs("/content/data", exist_ok=True)
procs = {p: start(p) for p in PARTS if not done(p)}
grown, tries = {p: (0.0, time.time()) for p in procs}, {p: 0 for p in PARTS}
restarts = 0
while procs:
    time.sleep(30)
    have = sizes()
    for part, proc in list(procs.items()):
        size = received(proc.pid) + sum(v for k, v in have.items() if k.startswith(part + "_"))
        if proc.poll() is not None:
            if done(part):
                del procs[part]
                continue
        elif size > grown[part][0]:
            grown[part] = (size, time.time())
            continue
        elif time.time() - grown[part][1] < 900:
            continue
        if tries[part] >= 4:  # cette partie : finie 4 fois sans le fichier complet, ou rien reçu depuis 15 min
            status("failed", restarts)
            raise SystemExit
        try:
            os.killpg(proc.pid, 9)
        except ProcessLookupError:
            pass
        restarts += 1
        tries[part] += 1
        procs[part], grown[part] = start(part), (0.0, time.time())
    status("running", restarts)
status("ok" if all(done(p) for p in PARTS) else "failed", restarts)
'''


def halt(session: str, state: dict, reason: str) -> None:
    """Un problème qui se répéterait sur toute machine : la machine est rendue et le suivi s'arrête (code 0, le
    lanceur ne le relance pas), au lieu de payer machine après machine pour rien."""
    colab("stop", "-s", session, timeout=120)
    state["halted"] = f"{datetime.now():%d/%m %H:%M} : {reason}"
    set_phase(state, "arrêté")
    note(f"ARRÊT DU SUIVI : {reason} — machine {session} rendue ; à corriger avant de relancer")
    sys.exit(0)


def micro_batch_for(state: dict, hardware: str) -> str:
    """Le micro-lot de cette machine : celui réduit après un manque de mémoire, sinon celui de MICRO_BATCH."""
    return state.get("micro_batch", {}).get(hardware) or MICRO_BATCH.get(hardware, "8")


def launch(session: str, hardware: str, pretrain_tokens: float, micro_batch: str | None = None,
           state: dict | None = None) -> None:
    """L'entraînement, en arrière-plan sur la machine (il survit à la fin de la commande). state["log_mark"] : où
    commence son journal dans data/colab/run.log (adjust_budget ne mesure la vitesse qu'après)."""
    if state is not None:
        state["log_mark"] = len(LOG.read_text(encoding="utf-8", errors="replace")) if LOG.exists() else 0
    run = RUN | {"MICRO_BATCH": micro_batch or MICRO_BATCH.get(hardware, "8"), "PRETRAIN_TOKENS": str(int(pretrain_tokens)),
                 "CORPUS_CAP": CORPUS_CAP.get(hardware, "0")}
    out = vm(session, f'''
import os, subprocess
os.makedirs("{REMOTE}", exist_ok=True)
for f in ("{REMOTE}/STOP", "{REMOTE}/exit", "{REMOTE}/{STAGE}.tokens"):
    if os.path.exists(f): os.remove(f)
env = {{**os.environ, **{json.dumps(run)}}}
cmd = "{PYTHON} /content/sm/{SCRIPT} train >> {REMOTE}/train.log 2>&1; echo $? > {REMOTE}/exit"
p = subprocess.Popen(["bash", "-c", cmd], cwd="/content/sm", env=env, start_new_session=True)
print("lancé", p.pid)
''')
    note(f"{session} ({hardware}) : {out.strip()[-60:]} · {pretrain_tokens / 1e9:.2f} G tokens visés · micro-lots de {run['MICRO_BATCH']}")


def status(session: str, offset: int) -> tuple[str, str, int]:
    """(état, nouveau texte du journal, nouvelle position) ; état : running, done, stopped, crashed, unreachable."""
    info = vm_json(session, f'''
import os, json
p = "{REMOTE}/train.log"
log = open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else ""
code = open("{REMOTE}/exit").read().strip() if os.path.exists("{REMOTE}/exit") else None
print(json.dumps({{"new": log[{offset}:], "size": len(log), "exit": code}}))
''', timeout=120)
    if info is None:
        return "unreachable", "", offset
    state = {None: "running", "0": "done", "3": "stopped"}.get(info["exit"], "crashed")
    return state, info["new"], info["size"]


def fetch(session: str, remote: str, local: Path) -> bool:
    """Un fichier de la machine, rapatrié par morceaux de 40 Mo et vérifié (empreinte). Sur la machine, le fichier
    est découpé au fil de la lecture (jamais entier en mémoire), et ses pages comme les morceaux en sortent ensuite :
    la mémoire sert au corpus de l'entraînement (voir KAGGLE_LOOP_SCRIPT)."""
    info = vm_json(session, f'''
import glob, hashlib, json, os
src = "{remote}"
if not os.path.exists(src): print(json.dumps({{"n": 0}})); raise SystemExit
for f in glob.glob("/content/parts/*"): os.remove(f)
os.makedirs("/content/parts", exist_ok=True)
digest, n, size = hashlib.sha256(), 0, 0
with open(src, "rb") as f:
    while chunk := f.read({CHUNK}):
        open(f"/content/parts/{{n:04d}}", "wb").write(chunk)
        digest.update(chunk)
        n, size = n + 1, size + len(chunk)
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
print(json.dumps({{"n": n, "sha": digest.hexdigest(), "size": size}}))
''', timeout=1200)
    if not info or not info["n"]:
        return False
    tmp = Path(tempfile.mkdtemp(dir=DIR))
    for attempt in range(3):  # en parallèle (deux fois plus vite qu'un par un), les morceaux manquants réessayés
        todo = [f"{i:04d}" for i in range(info["n"]) if not (tmp / f"{i:04d}").exists()
                or (tmp / f"{i:04d}").stat().st_size != min(CHUNK, info["size"] - i * CHUNK)]
        with ThreadPoolExecutor(TRANSFERS) as pool:
            list(pool.map(lambda name: colab("download", "-s", session, f"/content/parts/{name}", str(tmp / name),
                                             timeout=900), todo))
    vm(session, "import glob, os\nfor f in glob.glob('/content/parts/*'): os.remove(f)", timeout=120)
    parts = sorted(tmp.glob("*"))
    digest = hashlib.sha256()
    for f in parts:
        digest.update(f.read_bytes())
    ok = len(parts) == info["n"] and digest.hexdigest() == info["sha"]
    if ok:
        local.parent.mkdir(parents=True, exist_ok=True)
        with open(local, "wb") as out:
            for f in parts:
                out.write(f.read_bytes())
    for f in parts:
        f.unlink()
    tmp.rmdir()
    note(f"{Path(remote).name} : {info['size'] / 1e9:.2f} Go rapatriés" if ok else f"{Path(remote).name} : rapatriement incomplet")
    return ok


def backup(session: str, state: dict | None = None) -> bool:
    """L'état de la machine (poids + optimiseurs + avancement) dans data/colab/state/STATE_FILE ; son pas est noté
    (state["backup_step"]) quand c'est l'état que export_last a déjà lu."""
    tmp = BACKUP / f"{STATE_FILE}.part"
    info = vm_json(session, f"import json, os; print(json.dumps({{'mtime': os.path.getmtime('{REMOTE}/{STATE_FILE}')}}))", timeout=60)
    if not fetch(session, f"{REMOTE}/{STATE_FILE}", tmp):
        return False
    tmp.replace(BACKUP / STATE_FILE)
    if state is not None:
        same = info and abs(info["mtime"] - state.get("last_export_mtime", -1)) < 1
        state["backup_step"] = state.get("last_export_step") if same else None
        save(state)
    return True


# Tourne sur la machine, en arrière-plan, indépendamment du Mac (éteint ou en veille, rien ne change) : chaque nouvel
# état écrit par l'entraînement (toutes les 30 min, et à l'arrêt demandé avant un changement de machine) devient la
# version du dataset Kaggle privé KAGGLE_STATE ; un envoi raté est réessayé 5 min plus tard, puis 10, … 30 au plus.
# Où elle en est : /content/kaggle_backup.json ({state, ready : la dernière version prête, current, error}).
# Elle garde aussi le corpus de l'entraînement en mémoire : le disque de la machine (PersistentDisk, ~90 lectures au
# hasard par seconde) fait tomber l'entraînement, qui tire ses exemples au hasard dans le corpus, de ~22 800 tokens/s
# à ~3 500 dès qu'une sauvegarde de 4,8 Go chasse le corpus de la mémoire (mesuré le 22/10 sur v5e : 0,62 Go en
# mémoire sur 8 après une sauvegarde ; il mettait ~30 min à revenir). Les pages des gros fichiers devenus inutiles
# sont retirées de la mémoire, et les morceaux du corpus qui en sont sortis relus d'un trait (lecture séquentielle).
KAGGLE_LOOP_SCRIPT = r'''
import ctypes, json, os, subprocess, sys, threading, time
src, ds, py = sys.argv[1], sys.argv[2], sys.argv[3]
out, statf = "/content/kstate", "/content/kaggle_backup.json"
env = {**os.environ, "KAGGLE_API_TOKEN": open("/root/.kaggle/token").read().strip()}
try:
    now = json.load(open(statf))  # relancée : elle repart de la dernière version prête
except (OSError, ValueError):
    now = {}
def status(**kw):
    now.update(kw)
    json.dump(now, open(statf + ".tmp", "w")); os.replace(statf + ".tmp", statf)
def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)
def sh(cmd, timeout=3600):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)[-400:]
    except subprocess.TimeoutExpired:
        return 1, "délai dépassé"
def progress(path):
    """Le pas et les tokens de l'état, lus sans charger ses tenseurs (le torch du venv du run, en mmap)."""
    code = ("import json, sys, numpy as np, torch\n"
            "allowed = [np._core.multiarray.scalar, np.dtype] + [type(np.dtype(t)) for t in ('float32', 'float64', 'int64')]\n"
            "with torch.serialization.safe_globals(allowed):\n"
            "    st = torch.load(sys.argv[1], map_location='cpu', weights_only=True, mmap=True)\n"
            "print(json.dumps({'step': int(st['step']), 'tokens': int(st['tokens'])}))")
    try:
        p = subprocess.run([py, "-c", code, path], capture_output=True, text=True, timeout=600,
                           env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"})
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        return {}
try:  # la mémoire des fichiers (mincore, posix_fadvise) : seulement sous Linux
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
except (OSError, AttributeError):
    libc = None
PAGE, CHUNK = os.sysconf("SC_PAGE_SIZE"), 64 << 20
def resident(fd, offset, length):
    """Octets de [offset, offset + length) en mémoire (mincore : rien n'est lu)."""
    addr = libc.mmap(None, length, 1, 1, fd, offset)  # PROT_READ, MAP_SHARED
    if addr in (None, 2**64 - 1):  # MAP_FAILED : on ne relit rien
        return length
    vec = (ctypes.c_ubyte * ((length + PAGE - 1) // PAGE))()
    libc.mincore(addr, length, vec)
    libc.munmap(addr, length)
    b = bytes(vec)
    return (len(b) - b.count(0)) * PAGE
def drop(path):
    """Les pages de `path` retirées de la mémoire (le fichier reste sur le disque)."""
    if not hasattr(os, "posix_fadvise"):  # pas Linux
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
def corpus():
    """[(fichier, octets lus par l'entraînement)] : les .bin qu'il a ouverts, coupés à CORPUS_CAP réparti selon MIX
    (son environnement), comme dans train_tpu.py ; [] s'il ne tourne pas."""
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            if b"train_tpu.py" not in open(f"/proc/{pid}/cmdline", "rb").read():
                continue
            files = sorted({l.split(None, 5)[5].strip() for l in open(f"/proc/{pid}/maps") if l.rstrip().endswith("_train.bin")})
            e = dict(kv.split("=", 1) for kv in open(f"/proc/{pid}/environ", "rb").read().decode(errors="replace").split("\0") if "=" in kv)
        except (OSError, IndexError):
            continue
        if not files:  # le bash qui l'a lancé
            continue
        mix = dict((k, float(v)) for k, v in (item.split(":") for item in e["MIX"].split(","))) if e.get("MIX") else {}
        cap, parts = float(e.get("CORPUS_CAP") or 0), []
        for f in files:
            name, n = os.path.basename(f)[: -len("_train.bin")], os.path.getsize(f)
            if cap and mix.get(name):
                n = min(n, 2 * int(cap * mix[name] / sum(mix.values())))  # des tokens uint16
            parts.append((f, n))
        return parts
    return []
def available():
    try:
        return next(int(l.split()[1]) * 1024 for l in open("/proc/meminfo") if l.startswith("MemAvailable:"))
    except (OSError, StopIteration):
        return 0
# Seulement après une grosse perte (une sauvegarde), et pas plus d'une fois par quart d'heure : /content est sur un
# périphérique loop sans accès direct (dio 0), chaque page lue y est gardée deux fois en mémoire, et les accès de
# l'entraînement (une page tous les ~9 h en moyenne) ne la protègent pas ; relire tout ce qui manque à chaque tour
# chassait le corpus par le corpus (mesuré le 22/10 : 20 000 → 8 000 tokens/s).
WARM_BELOW, WARM_EVERY = 0.75, 900
last_warm = 0.0
def keep_warm():
    """Les morceaux du corpus en grande partie sortis de la mémoire, relus d'un trait ; pas pendant que l'entraînement
    écrit un état, ni s'il n'y a pas la place de tout garder."""
    global last_warm
    if libc is None or os.path.exists(src + ".tmp") or time.time() - last_warm < WARM_EVERY:
        return
    parts = corpus()
    total = sum(n for _, n in parts)
    if not parts or total > available() - (2 << 30):
        return
    have, todo = 0, []
    for path, n in parts:
        fd = os.open(path, os.O_RDONLY)
        try:
            for off in range(0, n, CHUNK):
                length = min(CHUNK, n - off)
                r = resident(fd, off, length)
                have += r
                if r < 0.5 * length:
                    todo.append((path, off, length))
        finally:
            os.close(fd)
    if have >= WARM_BELOW * total:
        return
    last_warm = time.time()
    t0, read, buf = time.time(), 0, bytearray(8 << 20)
    for path, off, length in todo:
        fd = os.open(path, os.O_RDONLY)
        try:
            pos = off
            while pos < off + length:
                k = os.preadv(fd, [memoryview(buf)[: min(len(buf), off + length - pos)]], pos)
                if k <= 0:
                    break
                pos += k
            read += pos - off
        finally:
            os.close(fd)
        if os.path.exists(src + ".tmp"):  # une sauvegarde commence : on s'arrête là
            break
    log(f"corpus remis en mémoire : {read / 1e9:.2f} Go relus en {time.time() - t0:.0f} s "
        f"({have / total:.0%} du corpus en mémoire avant)")
def upload():
    """Le dernier état → une nouvelle version du dataset (la seule gardée). True : prête à être téléchargée."""
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        os.remove(os.path.join(out, f))
    path = os.path.join(out, "pretrain_state.pt")
    os.link(src, path)  # le même fichier, sans copie : l'entraînement écrit le suivant à côté puis le renomme
    info = {"mtime": os.path.getmtime(path), "size": os.path.getsize(path), **progress(path)}
    json.dump(info, open(os.path.join(out, "info.json"), "w"))  # la machine qui reprend vérifie qu'elle a cette version
    json.dump({"title": "super-modele-etat", "id": ds, "licenses": [{"name": "CC0-1.0"}]},
              open(os.path.join(out, "dataset-metadata.json"), "w"))
    status(state="running", current=info, since=t0, error="")
    log(f"envoi de l'état du pas {info.get('step')}")
    stop = threading.Event()
    def dropper():  # l'envoi lit 4,8 Go : leurs pages retirées au fur et à mesure, le corpus reste en mémoire
        while not stop.wait(2):
            drop(path)
    threading.Thread(target=dropper, daemon=True).start()
    try:
        exists = sh(f"kaggle datasets status {ds}", 120)[0] == 0
        # une seule version gardée : la nouvelle remplace les anciennes (4,8 Go chacune)
        code, text = sh(f"kaggle datasets version -p {out} -m 'etat {int(info['mtime'])}' --delete-old-versions" if exists
                        else f"kaggle datasets create -p {out}")
        if code:
            status(state="failed", error=text)
            return False
        uploaded = time.time()
        for _ in range(120):  # Kaggle prépare la nouvelle version : « ready » quand elle est téléchargeable
            time.sleep(30)
            code, text = sh(f"kaggle datasets status {ds}", 120)
            if "ready" in text.lower() and time.time() - uploaded > 90:
                status(state="idle", ready={**info, "at": time.time(), "upload_s": round(uploaded - t0),
                                            "total_s": round(time.time() - t0)})
                log(f"pas {info.get('step')} prêt sur Kaggle (envoi {round(uploaded - t0)} s, total {round(time.time() - t0)} s)")
                return True
            if "error" in text.lower() or "fail" in text.lower():
                status(state="failed", error=text)
                return False
        status(state="failed", error="version pas prête après 1 h")
        return False
    finally:
        stop.set()
        drop(path)
sh("python3 -m pip install -q kaggle==2.2.4")
status(state="idle", pid=os.getpid())  # un envoi interrompu par l'arrêt de la boucle précédente est refait
log("boucle démarrée")
fails = 0
while True:
    done = (now.get("ready") or {}).get("mtime", 0)
    fresh = os.path.exists(src) and os.path.getmtime(src) > done + 1
    try:
        keep_warm()
    except Exception as error:  # le cache n'est jamais une raison d'arrêter les sauvegardes
        log(f"cache : {error!r}")
    if fresh and time.time() >= now.get("retry_at", 0):
        try:
            ok = upload()
        except Exception as error:  # état disparu entre-temps, disque plein…
            status(state="failed", error=repr(error)[-300:])
            ok = False
        fails = 0 if ok else fails + 1
        if not ok:
            status(retry_at=time.time() + 60 * min(30, 5 * fails))
            log(f"envoi raté : {now['error'].strip()[-200:]} ; nouvel essai dans {min(30, 5 * fails)} min")
    time.sleep(30)
'''


def kaggle_sync(session: str, state: dict) -> dict | None:
    """La boucle de la machine (KAGGLE_LOOP_SCRIPT) démarrée si elle n'y tourne pas, remplacée entre deux envois si
    son code a changé, et sa dernière version prête notée dans state["kaggle_backup"] : {state: ready, mtime, size,
    step, tokens}."""
    info = vm_json(session, f'''
import json, os, signal, subprocess, time
script, p = {KAGGLE_LOOP_SCRIPT!r}, "/content/kaggle_backup.json"
def alive():
    try:
        pid = open("/content/kaggle_loop.pid").read().strip()
        return "kaggle_loop.py" in subprocess.run(["ps", "-p", pid, "-o", "args="], capture_output=True, text=True).stdout
    except OSError:
        return False
now = json.load(open(p)) if os.path.exists(p) else {{}}
old = open("/content/kaggle_loop.py").read() if os.path.exists("/content/kaggle_loop.py") else ""
running, replaced = alive(), False
if running and old != script and now.get("state") != "running":  # nouvelle version : pas au milieu d'un envoi
    os.kill(int(open("/content/kaggle_loop.pid").read()), signal.SIGTERM)
    time.sleep(1)
    running, replaced = alive(), True
started = not running
if started:
    open("/content/kaggle_loop.py", "w").write(script)
    pr = subprocess.Popen(["python3", "/content/kaggle_loop.py", "{REMOTE}/{STATE_FILE}", "{KAGGLE_STATE}", "{PYTHON}"],
                          start_new_session=True, stdout=open("/content/kaggle_loop.log", "a"), stderr=subprocess.STDOUT)
    open("/content/kaggle_loop.pid", "w").write(str(pr.pid))
print(json.dumps({{**now, "started": started, "replaced": replaced}}))
''', timeout=120)
    if not info:
        return None
    if info["started"]:
        note(f"{session} : boucle de la machine {'remplacée par la nouvelle version' if info['replaced'] else 'démarrée'} "
             "(sauvegardes sur Kaggle, corpus gardé en mémoire)")
    ready, kb = info.get("ready") or {}, state.get("kaggle_backup") or {}
    if ready.get("mtime", 0) > (kb.get("mtime") or 0) + 1:
        state["kaggle_backup"] = {"state": "ready", **{k: ready[k] for k in ("mtime", "size", "step", "tokens") if k in ready}}
        save(state)
        note(f"sauvegarde sur Kaggle prête : pas {ready.get('step')}, {ready['size'] / 1e9:.2f} Go "
             f"(envoi {ready.get('upload_s')} s, total {ready.get('total_s')} s)")
    error = (info.get("error") or "") if info.get("state") == "failed" else ""
    if error[:80] != state.get("kaggle_error", ""):
        state["kaggle_error"] = error[:80]
        save(state)
        if error:
            note(f"sauvegarde sur Kaggle ratée, la machine réessaiera : {error.strip()[-200:]}")
    return info


def kaggle_wait(session: str, state: dict, wait: float = 3600) -> bool:
    """Attend que le dernier état écrit sur la machine (celui de l'arrêt demandé) soit la version prête sur Kaggle."""
    written = vm_json(session, f"import json, os; p = '{REMOTE}/{STATE_FILE}'; "
                               "print(json.dumps({'mtime': os.path.getmtime(p) if os.path.exists(p) else 0}))", timeout=60)
    if not written or not written["mtime"]:
        return False
    for _ in range(max(1, int(wait // 30))):  # comptées une à une, comme dans fetch_data (Mac en veille)
        kaggle_sync(session, state)
        if (state.get("kaggle_backup") or {}).get("mtime", 0) >= written["mtime"] - 1:
            return True
        time.sleep(30)
    note(f"{session} : l'état de {datetime.fromtimestamp(written['mtime']):%H:%M} toujours pas prêt sur Kaggle "
         f"après {wait / 60:.0f} min")
    return False


EXPORT_SCRIPT = r'''
import gc, json, os, sys
import numpy as np, torch
from safetensors.torch import save_file
src, out, mtime = sys.argv[1], sys.argv[2], float(sys.argv[3])
allowed = [np._core.multiarray.scalar, np.dtype] + [type(np.dtype(t)) for t in ("float32", "float64", "int64")]
with torch.serialization.safe_globals(allowed):  # mmap : seuls les poids sont lus (pas l'optimiseur), sans copie
    st = torch.load(src, map_location="cpu", weights_only=True, mmap=True)
w = {k: (v.to(torch.bfloat16) if v.dim() == 2 and not k.startswith("embed") else v).contiguous() for k, v in st["model"].items()}
vocab, n_embd = w["embed.weight"].shape
config = {"vocab_size": vocab, "block_size": 512, "n_embd": n_embd, "n_head": n_embd // 128,
          "n_layer": len({k.split(".")[1] for k in w if k.startswith("blocks.")}), "dropout": 0.0, "arch": "modern"}
info = {"step": int(st["step"]), "tokens": int(st["tokens"]), "val_loss": float(st["best"])}
save_file(w, out + ".tmp", metadata={"config": json.dumps(config), "tokenizer": open("/content/sm/tokenizer.json").read(),
                                     "info": json.dumps(info)})
os.replace(out + ".tmp", out)
params = sum(t.numel() for t in w.values())
del st, w
gc.collect()
if hasattr(os, "posix_fadvise"):  # les pages de l'état hors de la mémoire : le corpus de l'entraînement d'abord
    fd = os.open(src, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
print(json.dumps({"fresh": True, "mtime": mtime, "params": params, **info}))
'''


def export_last(session: str, state: dict) -> None:
    """Dès que l'entraînement a écrit un nouvel état : ses poids seuls (matrices en bf16, ~1,2 Go au lieu des 4,8 Go
    de l'état complet), extraits sur la machine puis rapatriés dans checkpoints/VERSION-last.safetensors, au format
    que le chat sait charger (« dernier checkpoint »)."""
    out = f"/content/last/{VERSION}-last.safetensors"
    # l'extraction tourne dans le venv du run (torch 2.9) : le torch de l'image Colab ne charge pas forcément
    info = vm_json(session, f'''
import json, os, subprocess
src = "{REMOTE}/{STATE_FILE}"
mtime = os.path.getmtime(src) if os.path.exists(src) else 0
if mtime <= {state.get("last_export_mtime", 0)}:
    print(json.dumps({{"fresh": False}}))
else:
    open("/content/export_last.py", "w").write({EXPORT_SCRIPT!r})
    os.makedirs("/content/last", exist_ok=True)
    p = subprocess.run(["{JAXENV}/bin/python", "/content/export_last.py", src, "{out}", str(mtime)], capture_output=True,
                       text=True, env={{**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}})
    lines = [l for l in p.stdout.splitlines() if l.startswith("{{")]
    print(lines[-1] if lines else json.dumps({{"fresh": False, "error": p.stderr[-300:]}}))
''', timeout=600)
    if info and info.get("error") and state.get("last_export_error") != info["error"][:80]:
        state["last_export_error"] = info["error"][:80]
        note(f"dernier checkpoint : extraction ratée : {info['error'][-200:]}")
    if not info or not info.get("fresh"):
        return
    local = ROOT / "checkpoints" / f"{VERSION}-last.safetensors"
    tmp = local.with_name(local.stem + ".tmp.safetensors")
    if not fetch(session, out, tmp):
        return
    tmp.replace(local)
    saved = datetime.fromtimestamp(info["mtime"])
    local.with_suffix(".json").write_text(json.dumps(
        {"version": local.stem, "last": True, "params": info["params"], "step": info["step"], "tokens": info["tokens"],
         "val_loss": info["val_loss"], "saved": saved.isoformat(timespec="seconds"),
         "exported": datetime.now().isoformat(timespec="seconds"),
         "description": "dernier état du pré-entraînement rapatrié de Colab, pas encore affiné sur les Simpson"},
        indent=2, ensure_ascii=False))
    state["last_export_mtime"], state["last_export_step"] = info["mtime"], info["step"]
    save(state)
    note(f"dernier checkpoint : pas {info['step']:,}, {info['tokens'] / 1e9:.2f} G tokens, val {info['val_loss']:.4f} "
         f"(état de {saved:%H:%M})")


def restore(session: str) -> None:
    """La dernière sauvegarde, envoyée par morceaux puis recollée et vérifiée sur la machine."""
    data_path = BACKUP / STATE_FILE
    digest = hashlib.sha256()
    with open(data_path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    sha = digest.hexdigest()
    tmp = Path(tempfile.mkdtemp(dir=DIR))
    with open(data_path, "rb") as f:
        for i, chunk in enumerate(iter(lambda: f.read(CHUNK), b"")):
            (tmp / f"{i:04d}").write_bytes(chunk)
    vm(session, "import os, shutil; shutil.rmtree('/content/data/state/parts', ignore_errors=True); "
                "os.makedirs('/content/data/state/parts', exist_ok=True)")
    names = sorted(p.name for p in tmp.glob("*"))
    for attempt in range(3):  # en parallèle, puis les morceaux arrivés incomplets ou manquants renvoyés
        sizes = vm_json(session, "import json, os; d = '/content/data/state/parts'; "
                                 "print(json.dumps({f: os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)}))",
                        timeout=120) or {} if attempt else {}
        todo = [n for n in names if sizes.get(n) != (tmp / n).stat().st_size]
        if not todo:
            break
        with ThreadPoolExecutor(TRANSFERS) as pool:
            list(pool.map(lambda n: colab("upload", "-s", session, str(tmp / n), f"/content/data/state/parts/{n}",
                                          timeout=900), todo))
    for part in tmp.glob("*"):
        part.unlink()
    tmp.rmdir()
    out = vm(session, f'''
import glob, hashlib, os
parts = sorted(glob.glob("/content/data/state/parts/*"))
data = b"".join(open(p, "rb").read() for p in parts)
open("/content/data/state/{STATE_FILE}", "wb").write(data)
for p in parts: os.remove(p)
print(hashlib.sha256(data).hexdigest())
''', timeout=1200)
    if sha not in out:
        raise RuntimeError(f"{session} : la sauvegarde restaurée ne correspond pas")
    note(f"{session} : sauvegarde restaurée, empreinte vérifiée")


def stop_training(session: str, wait: float = 1800) -> bool:
    """Demande l'arrêt propre (fichier STOP) et attend que l'état soit sauvegardé."""
    vm(session, f"open('{REMOTE}/STOP', 'w').close()")
    for _ in range(max(1, int(wait // 30))):  # comptées une à une, comme dans fetch_data (Mac en veille)
        time.sleep(30)
        state = status(session, 10**12)[0]
        if state not in ("running", "unreachable"):
            return state in ("stopped", "done")
    return False


LAST_REFUSAL = ""  # la dernière réponse de Colab à une demande de v6e refusée


def try_v6e() -> str | None:
    """Demande un v6e-1 ; renvoie sa session (accordée tout de suite, ou après le délai de réponse de l'outil).

    Seules les machines apparues pendant la demande sont considérées comme les nôtres : un notebook ouvert à la
    main n'est jamais touché.
    """
    global LAST_REFUSAL
    listed = machines() or []
    before = {m["endpoint"] for m in listed}
    if not any(m["name"] == "v6" for m in listed):  # une session « v6 » d'un v6e disparu bloquerait la demande
        colab("stop", "-s", "v6", timeout=120)
    out = colab("new", "-s", "v6", "--tpu", "v6e1", timeout=200)
    if hardware_of("v6") == "V6E1":
        return "v6"
    # la réponse de Colab, pour le journal : manque de TPU libres, quota, délai de l'outil…
    lines = [l.strip(" │╭╮╰╯─") for l in out.splitlines() if l.strip(" │╭╮╰╯─") and "new version" not in l and "colab update" not in l
             and "enable_update_check" not in l]
    # sans l'adresse de la demande (elle porte l'identifiant du notebook, coupé sur deux lignes) : la raison seule
    text = re.sub(r"https?://\S+", "…", re.sub(r"https?://.*?accelerator=\w+", "…", " ".join(lines)))
    LAST_REFUSAL = text[-160:] if lines else "pas de réponse (délai de 200 s dépassé)"
    time.sleep(60)  # une demande « expirée » peut quand même aboutir côté serveur
    new = {m["endpoint"]: m["hardware"] for m in machines() or [] if m["endpoint"] not in before}
    adopted = adopt(set(new)) if new else {}
    for name, accelerator in adopted.items():
        if accelerator == "V6E1":
            return name
        colab("stop", "-s", name, timeout=120)
        note(f"machine inattendue rendue : {name} ({accelerator})")
    return None


def count_v6e(state: dict, granted: str | None) -> None:
    """Les demandes de v6e faites depuis un v5e ou en pause (le dashboard les affiche), et une note par heure dans le
    journal."""
    now = datetime.now().isoformat(timespec="seconds")
    hunt = state.setdefault("v6e_hunt", {"since": now, "tries": 0, "noted": time.time()})
    hunt.update(tries=hunt["tries"] + 1, last=now, answer="accordé" if granted else LAST_REFUSAL[-100:])
    if not granted and time.time() - hunt["noted"] > 3600:
        hunt["noted"] = time.time()
        note(f"v6e : {hunt['tries']} demandes refusées depuis {datetime.fromisoformat(hunt['since']):%d/%m %H:%M} "
             f"(dernière réponse de Colab : {LAST_REFUSAL[-100:]})")
    save(state)


PAUSE_V5E_AFTER = 12 * 3600  # une pause (data/colab/pause) n'attend qu'un v6e pendant 12 h, puis prend aussi un v5e


def get_machine(state: dict | None = None) -> str:
    """Un TPU, v6e de préférence, sinon v5e (en pause : pas avant state["v5e_after"]) ; réessaie jusqu'à en obtenir
    un. Une relance demandée (data/colab/restart) arrête le suivi entre deux demandes, jamais pendant : une machine
    accordée en retard serait perdue de vue."""
    while True:
        if (DIR / "hold").exists():  # l'entraînement tourne ailleurs (une session Kaggle) : aucune demande en attendant
            note("en attente : l'entraînement tourne sur Kaggle (data/colab/hold), pas de demande de TPU")
            while (DIR / "hold").exists() and not (DIR / "restart").exists():
                time.sleep(60)
            note("fin de l'attente : demandes de TPU reprises")
        if (DIR / "restart").exists():
            (DIR / "restart").unlink()
            note("suivi arrêté pour être relancé (data/colab/restart)")
            sys.exit(1)
        session = try_v6e()
        if state is not None:
            count_v6e(state, session)
        if session:
            return session
        if ALLOW_V5E and time.time() >= (state or {}).get("v5e_after", 0):
            colab("new", "-s", "v5", "--tpu", "v5e1", timeout=200)
            if hardware_of("v5") == "V5E1":
                return "v5"
            note(f"aucun TPU libre : nouvel essai dans {HUNT} s · Colab : {LAST_REFUSAL}")
        if (DIR / "restart").exists():
            (DIR / "restart").unlink()
            note("suivi arrêté pour être relancé (data/colab/restart)")
            sys.exit(1)  # run.sh le relance dans la minute
        time.sleep(HUNT)  # une demande refusée ne coûte rien : on redemande vite (Colab la garde ~60 s avant de couper)


def pause(session: str, state: dict) -> None:
    """Arrêt propre de l'entraînement (l'état écrit à l'arrêt part sur Kaggle par la boucle de la machine, sinon sur
    le Mac) puis machine rendue : plus aucune dépense. La reprise passe ensuite par get_machine : un v6e seulement
    pendant PAUSE_V5E_AFTER, puis un v5e aussi."""
    state["paused_at"] = datetime.now().isoformat(timespec="seconds")
    if ALLOW_V5E:
        state["v5e_after"] = time.time() + PAUSE_V5E_AFTER
    set_phase(state, "attente")
    note(f"pause : arrêt propre de {session}, état sur Kaggle, machine rendue ; v6e seulement"
         + (f" jusqu'au {datetime.fromtimestamp(state['v5e_after']):%d/%m à %H:%M}, puis un v5e aussi" if ALLOW_V5E else ""))
    if not stop_training(session):
        note(f"{session} : l'entraînement ne s'est pas arrêté proprement ; la reprise partira de la dernière sauvegarde")
    elif not kaggle_wait(session, state) and backup(session, state) and state.get("kaggle_backup"):
        state["kaggle_backup"]["state"] = "plus ancienne que la copie du Mac"  # setup reprendra la copie du Mac
    colab("stop", "-s", session, timeout=120)
    state["released_at"] = datetime.now().isoformat(timespec="seconds")
    if (state.get("kaggle_backup") or {}).get("state") == "ready":
        state["backup_step"] = state["kaggle_backup"].get("step") or state.get("backup_step")
    save(state)
    note(f"{session} rendue : en pause, plus aucune dépense ; reprise au pas {state['backup_step']}")


# unités gardées pour l'affinage, le score, les exemples et le rapatriement du modèle (~40 min de v6e, ~2 h de v5e)
FT_RESERVE = {"V6E1": 4.0, "V5E1": 6.0}


def adjust_budget(state: dict, session: str, log: str) -> bool:
    """Le nombre de tokens visé, ramené chaque heure à ce que le budget permet. Seulement d'après les mesures prises
    depuis la dernière reprise (celles d'avant ne disent rien de la machine actuelle) ; la vitesse : la médiane des
    lignes d'avancement de chaque minute, sans celles de la compilation. S'il ne reste que de quoi affiner, la cible
    devient les tokens déjà vus : le pré-entraînement s'arrête et l'affinage commence, avant le plafond. False : pas
    encore assez de mesures depuis la reprise (le suivi réessaie au tour suivant, pas dans une heure)."""
    mark = state.get("log_mark", 0)  # le journal de la machine actuelle seulement (pas celui de la précédente)
    log = log[mark:] if mark <= len(log) else log
    since = log[log.rfind("reprise : pas"):] if "reprise : pas" in log else log
    rows = re.findall(r"avancement\s+pas\s+[\d,]+/[\d,]+\s+train \S+\s+lr \S+\s+([\d.]+) G tokens\s+([\d,]+) tokens/s", since)
    speeds = [float(t.replace(",", "")) for _, t in rows]
    steady = sorted(v for v in speeds if speeds and v > 0.5 * max(speeds))
    # la consommation de cette machine seule : une seconde machine d'essais ne doit pas faire baisser la cible
    # (sa dépense est déjà comptée dans `spent`)
    rate = RATE.get(state.get("hardware")) or balance()[1]
    if len(steady) < 5 or not rate:
        return False
    seen, speed = float(rows[-1][0]) * 1e9, steady[len(steady) // 2]
    hours_left = max(0.0, state["cap"] - state.get("spent", 0) - FT_RESERVE.get(state.get("hardware"), 6.0)) / rate
    if hours_left < 0.25:  # plus de quoi pré-entraîner : place à l'affinage, tant qu'il est payable
        target = seen
    else:
        target = min(TOKENS_MAX, max(min(TOKENS_MIN, seen), seen + 0.9 * hours_left * 3600 * speed))
    if abs(target - state["pretrain_tokens"]) > 0.03 * state["pretrain_tokens"] or target == seen:
        vm(session, f"open('{REMOTE}/{STAGE}.tokens', 'w').write('{int(target)}')")
        note(f"budget : {speed:,.0f} tokens/s, {hours_left:.1f} h payables → cible {target / 1e9:.2f} G tokens "
             f"(au lieu de {state['pretrain_tokens'] / 1e9:.2f} ; {seen / 1e9:.2f} G déjà vus)")
        state["pretrain_tokens"] = target
        save(state)
    return True


def finish(session: str) -> None:
    """Le modèle affiné, son score et ses exemples, rapatriés dans checkpoints/ ; la machine est libérée."""
    info = vm_json(session, f'''
import glob, json, os
print(json.dumps({{"files": sorted(glob.glob("{REMOTE}/checkpoints/dialogues_*")) + ["{REMOTE}/exemples.txt"]}}))
''') or {"files": []}
    for remote in info["files"]:
        name = Path(remote).name
        # dialogues_bpe8192_gpt2_w1536_modded[_…].safetensors → v6-591M[_…].safetensors (et son .json)
        target = (ROOT / "checkpoints" / re.sub(r"^dialogues_bpe8192_gpt2_\w+?_modded", VERSION, name)
                  if name.startswith("dialogues_") else DIR / name)
        fetch(session, remote, target)
    colab("stop", "-s", session, timeout=120)
    note(f"{session} libérée")


def main() -> None:
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))  # un arrêt de l'extérieur passe par les `finally` (clé)
    state = load()
    session = state.get("session") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not session:
        sys.exit("donner le nom de la session Colab (colab sessions)")
    if "start_balance" not in state:
        state["start_balance"] = balance()[0]
        state["cap"] = round(state["start_balance"] - MARGIN, 2)
    state["engine"] = ENGINE
    state.pop("halted", None)  # relancé à la main après un arrêt du suivi : la raison de l'arrêt est dans watch.log
    state["allow_v5e"] = ALLOW_V5E  # pour le dashboard
    state.setdefault("pretrain_tokens", TOKENS_START)
    hardware = state.get("hardware") or hardware_of(session)
    state.update(session=session, hardware=hardware)
    save(state)
    if not state.get("launched"):
        set_phase(state, "préparation")
        resume = (BACKUP / STATE_FILE).exists() or (state.get("kaggle_backup") or {}).get("state") == "ready"
        if not setup(session, with_restore=resume, state=state):  # une reprise : la sauvegarde aussi
            halt(session, state, "préparation de la machine ratée (voir data/colab/watch.log)")
        launch(session, hardware, state["pretrain_tokens"], micro_batch_for(state, hardware), state)
        state.update(launched=True, offset=0, last_budget=0, launched_at=time.time(), last_export_mtime=0)
        set_phase(state, "entraînement")
        save(state)
    state.pop("kaggle_upload", None)  # les envois lancés par le suivi (avant la boucle sur la machine)
    live = {"session": session, "offset": state.get("offset", 0), "run_state": "running"}
    lock = threading.Lock()

    def sync() -> None:
        """Chaque minute, à part (les demandes de v6e prennent plusieurs minutes) : le journal pour le dashboard."""
        while True:
            with lock:
                current_session, offset = live["session"], live["offset"]
            run_state, new, new_offset = status(current_session, offset)
            with lock:
                if current_session == live["session"] and offset == live["offset"]:  # pas de changement entre-temps
                    if new:
                        with LOG.open("a", encoding="utf-8") as f:
                            f.write(new)
                        live["offset"] = new_offset
                    live["run_state"] = run_state
            if run_state != "unreachable":
                LOG.touch()  # le dashboard voit que l'entraînement est vivant (pas quand la machine ne répond plus)
            time.sleep(SYNC)

    def switch(new_session: str) -> None:
        with lock:
            live.update(session=new_session, offset=0, run_state="running")

    threading.Thread(target=sync, daemon=True).start()
    # la machine du run n'existe plus (suivi relancé après sa perte) : on le constate au premier tour, pas au troisième
    # la machine du run n'existe plus (suivi relancé après sa perte, Mac endormi entre-temps) : constaté ici, en
    # demandant à Colab puis à la machine elle-même, plutôt qu'après les tours du journal (pendant lesquels un v6e
    # accordé partirait dans un transfert depuis une machine morte)
    gone = state.get("launched") and machines() is not None and hardware_of(session) == ""
    misses = 0
    forced = "lost" if state.get("released_at") or (gone and status(session, 0)[0] == "unreachable") else None
    if not misses and not forced:  # la boucle d'envoi sur Kaggle tout de suite, sans attendre le premier tour
        kaggle_sync(session, state)
    while True:
        hunting = ALLOW_V5E and hardware != "V6E1" and not (DIR / "hold").exists()  # sur un v5e : un v6e à chaque tour
        time.sleep(0 if forced else HUNT if hunting else POLL)  # en pause, machine rendue : la demande de v6e tout de suite
        if (DIR / "restart").exists():  # relance demandée (nouveau code) : ici, entre deux tours, rien n'est en cours
            (DIR / "restart").unlink()
            note("suivi arrêté pour être relancé (data/colab/restart)")
            sys.exit(1)  # run.sh le relance dans la minute
        if (DIR / "pause").exists():  # pause demandée : arrêt propre, état sur Kaggle, machine rendue
            (DIR / "pause").unlink()
            pause(session, state)
            forced = "lost"  # puis la reprise ailleurs (get_machine), sans attendre que le journal dise « injoignable »
        with lock:
            run_state = live["run_state"]
            state["offset"] = live["offset"]
        if forced:
            run_state, forced = forced, None
        current = balance()[0]
        if current is not None and state["start_balance"] is not None:
            state["spent"] = round(state["start_balance"] - current, 2)
        save(state)
        if run_state == "unreachable":
            misses += 1
            gone = machines() is not None and hardware_of(session) == ""
            if misses < 3 or not gone:  # réseau coupé, Mac en veille… : on attend
                continue
            run_state = "lost"
        misses = 0
        if run_state == "done":
            set_phase(state, "terminé")
            note("entraînement terminé : rapatriement du modèle affiné")
            finish(session)
            with LOG.open("a", encoding="utf-8") as f:
                f.write(f"résumé : run terminé, {state.get('spent', 0):.1f} unités\n")
            return
        if state.get("spent", 0) >= state["cap"]:
            note(f"plafond atteint ({state['spent']:.1f} unités) : arrêt propre, sauvegarde, machine libérée")
            stop_training(session)
            kaggle_wait(session, state) or backup(session)  # l'état de l'arrêt sur Kaggle, sinon sur le Mac
            colab("stop", "-s", session, timeout=120)
            return
        if run_state == "crashed":  # la machine est là : on y relance, depuis son propre état
            tail = vm(session, f"print(open('{REMOTE}/train.log', errors='replace').read()[-3000:])", timeout=120)
            cause = next((l for l in reversed(tail.splitlines()) if "Error" in l or "error" in l), "cause inconnue")
            note(f"{session} : entraînement planté : {cause.strip()[:200]}")
            mb = int(micro_batch_for(state, hardware))
            oom = any(m in tail for m in ("RESOURCE_EXHAUSTED", "Attempting to reserve", "That was not possible")) \
                or "out of memory" in tail.lower()
            if oom and mb > 2:
                state.setdefault("micro_batch", {})[hardware] = str(mb // 2)  # la mémoire du TPU : micro-lots / 2
                note(f"{session} : mémoire du TPU dépassée : relance avec des micro-lots de {mb // 2}")
            else:
                state["crashes"] = state.get("crashes", 0) + 1
                if time.time() - state.get("launched_at", 0) < 900 or state["crashes"] > 3:
                    halt(session, state, f"plantage {'au démarrage' if state['crashes'] <= 3 else 'répété'} : {cause.strip()[:150]}")
            launch(session, hardware, state["pretrain_tokens"], micro_batch_for(state, hardware), state)
            with lock:  # le plantage est traité : ne pas le compter deux fois avant la prochaine lecture du journal
                live["run_state"] = "running"
            state["launched_at"] = time.time()
            save(state)
            continue
        if run_state == "lost":
            colab("stop", "-s", session, timeout=120)
            if not state.get("paused_at"):  # en pause, la machine a été rendue exprès
                note(f"{session} : machine perdue — reprise ailleurs")
                state.setdefault("lost_at", datetime.now().isoformat(timespec="seconds"))
            set_phase(state, "attente")
            session = get_machine(state)
            hardware = hardware_of(session)
            state.update(session=session, hardware=hardware)
            set_phase(state, "préparation")
            if not setup(session, with_restore=True, state=state):
                halt(session, state, "préparation de la machine ratée (voir data/colab/watch.log)")
            launch(session, hardware, state["pretrain_tokens"], micro_batch_for(state, hardware), state)
            switch(session)
            state.update(offset=0, launched_at=time.time(), last_export_mtime=0, last_budget=0)
            for key in ("lost_at", "paused_at", "released_at", "v5e_after") + (("v6e_hunt",) if hardware == "V6E1" else ()):
                state.pop(key, None)
            set_phase(state, "entraînement")
            continue
        if (time.time() - state.get("last_budget", 0) > BACKUP_EVERY and LOG.exists()
                and adjust_budget(state, session, LOG.read_text(encoding="utf-8", errors="replace"))):
            state["last_budget"] = time.time()
        if ENGINE == "torch":  # un nouvel état écrit par train_tpu.py : ses poids pour le chat
            export_last(session, state)
        kaggle_sync(session, state)  # la boucle d'envoi de la machine : vivante, et sa dernière sauvegarde prête
        if hunting:
            new_session = try_v6e()
            count_v6e(state, new_session)
            if new_session:  # le v6e se prépare pendant que le v5e continue d'entraîner
                note(f"v6e obtenu ({new_session}) : préparation (le v5e continue), puis transfert")
                if not setup(new_session):
                    colab("stop", "-s", new_session, timeout=120)
                    note(f"{new_session} inutilisable : rendu, on reste sur {session}")
                    continue
                set_phase(state, "transfert")
                moved = False
                if stop_training(session):
                    # l'état de l'arrêt envoyé sur Kaggle par la boucle de la machine, puis téléchargé par le v6e
                    if kaggle_wait(session, state) and fetch_data(new_session, state["kaggle_backup"], state):
                        moved = True
                    elif backup(session, state):  # secours : par le Mac
                        try:
                            restore(new_session)
                            moved = True
                        except Exception as error:
                            note(f"transfert raté ({error})")
                if moved:
                    colab("stop", "-s", session, timeout=120)
                    note(f"{session} libérée ; reprise sur {new_session}")
                    launch(new_session, "V6E1", state["pretrain_tokens"], micro_batch_for(state, "V6E1"), state)
                    switch(new_session)
                    session, hardware = new_session, "V6E1"
                    state.update(session=session, hardware=hardware, offset=0, last_budget=0, launched_at=time.time(),
                                 last_export_mtime=0)
                    state.pop("v6e_hunt", None)
                else:
                    note("transfert raté : on relance sur la machine actuelle et on rend le v6e")
                    launch(session, hardware, state["pretrain_tokens"], micro_batch_for(state, hardware), state)
                    colab("stop", "-s", new_session, timeout=120)
                    state["launched_at"] = time.time()
                set_phase(state, "entraînement")


if __name__ == "__main__":
    main()
