"""Le tableau de bord de l'expérience : l'avancement de chaque étape et l'entraînement, en direct.

    uv run dashboard.py          # puis http://localhost:8766

Les étapes sont décrites dans kaggle/experience.json. Une session Kaggle : son état vient de
`kaggle kernels status` (toutes les 30 s) et son journal de `kaggle kernels logs -f`, qui le transmet en
direct pendant qu'elle tourne. Une étape locale : son fichier de journal. Le dashboard en tire les courbes
(loss, tokens/s, learning rate), les résultats et les erreurs. Il observe seulement : il ne lance ni
n'arrête rien.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import ftfy
import gradio as gr
import plotly.graph_objects as go

from .kaggle_cli import KAGGLE, OWNER, ROOT, env, kaggle, kernel_status

EXPERIENCE = ROOT / "kaggle" / "experience.json"
COLAB_STATE = ROOT / "data" / "colab" / "state.json"  # écrit par colab_watch.py : sa phase, la machine, la dépense
LAB = ROOT / "data" / "colab" / "lab.json"  # les essais de vitesse sur le TPU, rassemblés pendant les essais
LIVE = ROOT / "data" / "kaggle" / "live"  # les journaux et l'état des sessions, gardés d'un lancement à l'autre
POLL = 30  # secondes entre deux nouvelles de Kaggle
FINISHED = ("complete", "error", "cancel")

# --- Lecture des journaux ----------------------------------------------------------------------------------

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
NOISE = ("frozen modules", "Debugger warning", "PYDEVD_DISABLE", "debugger miss", "Debugging will proceed",
         "Xfrozen_modules", "SyntaxWarning", "re.sub(", "cells[i][c]", "[NbConvertApp]", "━━")
CLI_MESSAGE = re.compile(r"Log stream connection lost|\d{3} (?:Client|Server) Error")  # écrits par le CLI de Kaggle
TRAIN = re.compile(r"(?P<clock>\d\d:\d\d:\d\d) (?P<stage>\S+)\s+pas\s+(?P<step>[\d,]+)(?:/(?P<planned>[\d,]+))?\s+"
                   r"train (?P<train>\S+)\s+val (?P<val>[\d.]+)\s+lr ×?(?P<lr>[\d.e+-]+)\s+(?P<tokens>[\d.,]+) "
                   r"(?P<unit>[MG]) tokens\s+(?P<tps>[\d,]+) tokens/s\s+(?:session )?(?P<hours>[\d.]+) h")
STAGE = re.compile(r"=== (?P<stage>\S+) : (?P<hours>[\d.]+) h")
PROGRESS = re.compile(r"(?P<stage>\S+)\s+avancement\s+pas (?P<step>[\d,]+)/(?P<planned>[\d,]+)\s+train (?P<train>\S+)\s+"
                      r"lr ×?(?P<lr>[\d.]+)\s+(?P<tokens>[\d.]+) G tokens\s+(?P<tps>[\d,]+) tokens/s")
CORPUS = re.compile(r"(?P<lines>[\d,]+) M lignes lues\s+(?P<tokens>[\d,]+) M tokens\s+(?P<simpsons>\d+) blocs Simpson "
                    r"retirés\s+(?P<test>\d+) répliques du test retirées\s+\((?P<s>\d+) s\)")
RESULT = re.compile(r"(?P<kind>résultat|terminé) : (?P<json>\{.*\})")
AB = re.compile(r"^(?P<run>\S.*?)\s+pas\s+(?P<step>\d+)\s+val (?P<val>[\d.]+|nan)\s+(?P<s>\d+) s$")
AB_DONE = re.compile(r"^==> (?P<run>.+?) : val sous-titres (?P<val>[\d.]+)\s+Simpson \(jamais vus\) "
                     r"(?P<simpsons>[\d.]+)\s+(?P<s>\d+) s")
CLOCK = re.compile(r"^(\d\d):(\d\d):(\d\d) ")
ERROR = re.compile(r"Traceback|Error|échec|RESOURCE_EXHAUSTED|Killed|exception", re.IGNORECASE)


def clean(line: str) -> str:
    """Une ligne lisible : sans codes de couleur, accents réparés (le flux Kaggle arrive en « tÃ©lÃ©chargement »)."""
    return ftfy.fix_text(ANSI.sub("", line.rstrip("\n")))


def visible(line: str) -> bool:
    return bool(line.strip()) and not any(n in line for n in NOISE)


def number(text: str) -> float:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return float("nan")


def parse(lines: list[str]) -> dict:
    """Ce qu'un journal dit de l'avancement : mesures d'entraînement, corpus, essai de vitesse, résultat, erreurs."""
    out = {"train": [], "stages": [], "corpus": None, "result": None, "corpus_done": None, "bench": [],
           "errors": [], "ab": {}, "ab_done": {}, "first_clock": None, "progress": None, "minutes": []}
    origin = previous = None  # secondes de la première ligne datée, et de la précédente
    day = 0  # les minuits passés depuis la première ligne
    for line in lines:
        if c := CLOCK.match(line):
            now = sum(int(x) * k for x, k in zip(c.groups(), (3600, 60, 1)))
            if previous is not None and now < previous - 3600:
                day += 86400
            origin = now if origin is None else origin
            previous = now
        if out["first_clock"] is None and (m := CLOCK.match(line)):
            out["first_clock"] = tuple(int(x) for x in m.groups())
        if m := PROGRESS.search(line):  # une ligne d'avancement, chaque minute (train_tpu.py) : la plus récente
            out["progress"] = {"stage": m["stage"], "step": number(m["step"]), "planned": number(m["planned"]),
                               "train": number(m["train"]), "lr": number(m["lr"]), "tokens": number(m["tokens"]) * 1e9,
                               "tps": number(m["tps"]), "after": len(out["train"])}
            if c and origin is not None:  # chaque minute : de quoi tracer les courbes sans attendre la validation
                out["minutes"].append(out["progress"] | {"hours": (now + day - origin) / 3600})
        elif m := TRAIN.search(line):
            scale = 1e9 if m["unit"] == "G" else 1e6
            out["train"].append({"stage": m["stage"], "step": number(m["step"]), "planned": m["planned"] and number(m["planned"]),
                                 "train": number(m["train"]), "val": number(m["val"]), "lr": number(m["lr"]),
                                 "tokens": number(m["tokens"]) * scale, "tps": number(m["tps"]), "hours": number(m["hours"])})
        elif m := STAGE.search(line):
            out["stages"].append((m["stage"], number(m["hours"])))
        elif m := CORPUS.search(line):
            out["corpus"] = {k: number(v) for k, v in m.groupdict().items()}
        elif m := RESULT.search(line):
            try:
                out["result" if m["kind"] == "résultat" else "corpus_done"] = json.loads(m["json"])
            except json.JSONDecodeError:
                pass
        elif line.lstrip().startswith("{") and "tokens_per_s_per_core" in line:
            try:
                out["bench"].append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass
        elif m := AB_DONE.match(line):
            out["ab_done"][m["run"]] = {"val": number(m["val"]), "simpsons": number(m["simpsons"]), "s": number(m["s"])}
        elif m := AB.match(line):
            out["ab"].setdefault(m["run"], []).append({"step": number(m["step"]), "val": number(m["val"]), "s": number(m["s"])})
        if ERROR.search(line) and visible(line) and "Warning" not in line:
            out["errors"].append(line.strip())
    return out


# --- Les étapes --------------------------------------------------------------------------------------------


def load_state() -> dict:
    path = LIVE / "state.json"
    return json.loads(path.read_text()) if path.exists() else {}


STATE = load_state()  # {id: {statut: date à laquelle on l'a vu pour la première fois}}
STATE_LOCK = threading.Lock()


class Step:
    """Une étape de l'expérience : son état, son journal, et ce qu'on en déduit."""

    def __init__(self, spec: dict) -> None:
        self.spec, self.id = spec, spec["id"]
        self.status, self.lines = "absent", []
        self.since: dict[str, str] = STATE.get(self.id, {})

    def set_status(self, status: str) -> None:
        if status != self.status:
            if status in ("queued", "running") and self.status in FINISHED:
                self.since = {}  # une nouvelle exécution
            self.since.setdefault(status, datetime.now().isoformat(timespec="seconds"))
            self.status = status
            with STATE_LOCK:
                STATE[self.id] = self.since
                LIVE.mkdir(parents=True, exist_ok=True)
                (LIVE / "state.json").write_text(json.dumps(STATE, indent=2))

    def refresh(self) -> None:
        raise NotImplementedError

    def started(self, parsed: dict) -> datetime | None:
        """Le début de l'exécution : vu par le dashboard, ou la première heure écrite dans le journal (UTC)."""
        seen = datetime.fromisoformat(self.since["running"]) if "running" in self.since else None
        if parsed["first_clock"]:
            h, m, s = parsed["first_clock"]
            utc = datetime.now().astimezone().utcoffset() or timedelta()
            first = datetime.now().replace(hour=h, minute=m, second=s, microsecond=0) + utc
            if first > datetime.now() + timedelta(minutes=5):
                first -= timedelta(days=1)
            seen = min(seen, first) if seen else first
        return seen


def colab_state() -> dict:
    try:
        return json.loads(COLAB_STATE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


# La phase de colab_watch.py → l'état d'une carte Colab (l'entraînement lui-même : d'après son journal)
COLAB_PHASES = {"attente": "waiting", "préparation": "setup", "transfert": "moving", "arrêté": "halted"}


class LocalStep(Step):
    """Une étape sur ce Mac : on lit son fichier de journal."""

    def refresh(self) -> None:
        path = ROOT / self.spec["log"]
        if not path.exists():
            return self.set_status("absent")
        self.lines = [clean(line) for line in path.read_text(errors="replace").splitlines()]
        self.since.setdefault("running", datetime.fromtimestamp(path.stat().st_birthtime).isoformat(timespec="seconds"))
        if self.spec.get("archived"):  # un run remplacé, gardé pour comparer ses courbes
            return self.set_status("archived")
        if any(line.startswith("résumé") or RESULT.search(line) for line in self.lines):
            self.set_status("complete")
        elif any("arrêt demandé" in line for line in self.lines[-3:]):
            self.set_status("paused")  # arrêt propre (fichier STOP), état sauvegardé : la reprise repart de là
        elif time.time() - path.stat().st_mtime < 300:
            self.set_status("running")
        else:
            self.set_status("stopped")
        phase = colab_state().get("phase") if self.spec.get("colab") else None
        if phase in COLAB_PHASES:  # pas de machine ou machine en préparation : le journal ne dit plus rien d'actuel
            self.set_status(COLAB_PHASES[phase])


class KaggleStep(Step):
    """Une session Kaggle : son état, interrogé régulièrement, et son journal, suivi en direct."""

    def __init__(self, spec: dict) -> None:
        super().__init__(spec)
        self.slug = spec["kernel"]
        self.file = LIVE / f"{self.slug}.log"
        if self.file.exists():
            self.lines = [line for line in self.file.read_text(errors="replace").splitlines() if not CLI_MESSAGE.match(line)]
        self.status = self.since.get("_last", "absent")
        self.follower: threading.Thread | None = None
        self.final = False

    def save(self) -> None:
        LIVE.mkdir(parents=True, exist_ok=True)
        self.file.write_text("\n".join(self.lines), encoding="utf-8")

    def refresh(self) -> None:
        status = kernel_status(self.slug)
        if status == "unknown":  # Kaggle injoignable : on garde le dernier état connu
            return
        if status in ("queued", "running") and self.status in FINISHED:
            self.lines, self.final = [], False
        self.set_status(status)
        with STATE_LOCK:
            self.since["_last"] = status
        if status == "running" and not (self.follower and self.follower.is_alive()):
            self.follower = threading.Thread(target=self.follow, daemon=True)
            self.follower.start()
        if status in FINISHED and not self.final:  # le journal complet, tel que Kaggle l'a gardé
            log = kaggle("kernels", "logs", f"{OWNER}/{self.slug}", check=False, timeout=300)
            if log.strip() and log != "timeout":
                self.lines = [clean(line) for line in log.splitlines()]
                self.save()
                self.final = True

    def follow(self) -> None:
        """`kaggle kernels logs -f` : le journal depuis le début de la session, puis chaque nouvelle ligne.

        Le CLI se reconnecte seul quand le flux coupe, mais reste parfois bloqué à se reconnecter sans rien
        recevoir : après 3 coupures d'affilée, on l'arrête, et le suivi suivant (30 s plus tard) repart du début.
        """
        proc = subprocess.Popen([*KAGGLE, "kernels", "logs", "-f", f"{OWNER}/{self.slug}"], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env())
        fresh: list[str] = []
        lost = 0  # coupures du flux d'affilée
        for raw in proc.stdout:
            if CLI_MESSAGE.match(raw):  # un message du CLI, pas une ligne du notebook
                lost += raw.startswith("Log stream connection lost")
                if lost >= 3:
                    break
                continue
            lost = 0
            fresh.append(clean(raw))
            if len(fresh) >= len(self.lines):  # le flux repart du début : il remplace l'ancien journal quand il l'a rattrapé
                self.lines = fresh
            if len(fresh) % 50 == 0:
                self.save()
        stop(proc)
        if fresh and len(fresh) >= len(self.lines):
            self.lines = fresh
            self.save()


def stop(proc: subprocess.Popen) -> None:
    """Arrête le CLI et le Python qu'uvx a lancé pour lui (uvx ne transmet pas le signal à son enfant)."""
    children = subprocess.run(["pgrep", "-P", str(proc.pid)], capture_output=True, text=True).stdout.split()
    for pid in children:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    proc.terminate()
    proc.wait()


class Monitor:
    """Interroge Kaggle en arrière-plan ; l'interface ne fait que lire ce qu'il a trouvé."""

    def __init__(self) -> None:
        self.mtime, self.title, self.steps = 0.0, "", []
        self.checkpoints = "dialogues_*.json"  # les modèles du tableau des scores (motif dans checkpoints/)
        self.quota: dict[str, tuple[float, float]] = {}
        self.updated: datetime | None = None
        self.load()
        threading.Thread(target=self.loop, daemon=True).start()

    def load(self) -> None:
        """(Re)lit kaggle/experience.json : une étape ajoutée apparaît sans relancer le dashboard."""
        mtime = EXPERIENCE.stat().st_mtime
        if mtime == self.mtime:
            return
        spec = json.loads(EXPERIENCE.read_text(encoding="utf-8"))
        known = {s.id: s for s in self.steps}
        self.title = spec.get("title", "")
        self.checkpoints = spec.get("checkpoints", "dialogues_*.json")
        steps = []
        for s in spec["steps"]:
            step = known.get(s["id"]) or (KaggleStep(s) if s["kind"] == "kaggle" else LocalStep(s))
            step.spec = s  # une étape déjà suivie prend aussi sa nouvelle description (total attendu, détail…)
            steps.append(step)
        self.steps = steps
        self.mtime = mtime

    def loop(self) -> None:
        last_quota = 0.0
        while True:
            try:
                self.load()
            except (OSError, json.JSONDecodeError):
                pass
            for step in self.steps:
                try:
                    step.refresh()
                except Exception as error:  # une étape en panne ne doit pas arrêter les autres
                    step.lines = step.lines + [f"(dashboard) {type(error).__name__}: {error}"]
            if not any(isinstance(s, KaggleStep) for s in self.steps):
                self.quota = {}  # aucun run sur Kaggle : pas de quota Kaggle à afficher
            elif time.time() - last_quota > 300:
                self.quota = read_quota()
                last_quota = time.time()
            self.updated = datetime.now()
            time.sleep(POLL)


def read_quota() -> dict[str, tuple[float, float]]:
    """{"GPU": (heures utilisées, total), "TPU": …}, d'après `kaggle quota`."""
    out = {}
    for line in kaggle("quota", check=False, timeout=60).splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] in ("GPU", "TPU"):
            out[parts[0]] = (number(parts[1].rstrip("h")), number(parts[3].rstrip("h")))
    return out


# --- Mise en forme -----------------------------------------------------------------------------------------

STATUS = {"absent": ("pas encore lancé", "idle"), "queued": ("dans la file", "queued"), "running": ("en cours", "running"),
          "complete": ("terminé", "done"), "error": ("erreur", "error"), "cancel": ("annulé", "error"),
          "stopped": ("arrêté", "error"), "paused": ("en pause", "queued"),
          "archived": ("remplacé", "idle"), "waiting": ("en attente d'un TPU", "queued"),
          "setup": ("préparation de la machine", "queued"), "moving": ("transfert vers un v6e", "queued"),
          "halted": ("suivi arrêté", "error")}
COLORS = {"GPU": "#f97316", "TPU": "#8b5cf6", "Colab": "#14b8a6"}


def duration(seconds: float) -> str:
    minutes = max(0, int(seconds // 60))
    return f"{minutes // 60} h {minutes % 60:02d}" if minutes >= 60 else f"{minutes} min"


def fmt(x: float, digits: int = 3) -> str:
    return f"{x:,.{digits}f}".replace(",", " ").replace(".", ",")


def step_view(step: Step) -> dict:
    """Ce que la carte d'une étape affiche : progression, temps, mesures clés, dernière erreur."""
    parsed = parse(step.lines)
    now = datetime.now()
    view = {"fraction": 1.0 if step.status == "complete" else 0.0, "time": "", "metrics": [], "parsed": parsed}
    started = step.started(parsed) if step.status in ("running",) + FINISHED else None
    if step.status == "queued" and "queued" in step.since:
        view["time"] = f"en file depuis {duration((now - datetime.fromisoformat(step.since['queued'])).total_seconds())}"
    elif step.status == "running" and started:
        elapsed = (now - started).total_seconds()
        hours = step.spec.get("hours")
        if hours:
            view["fraction"] = min(0.99, elapsed / (hours * 3600))
            view["time"] = f"depuis {duration(elapsed)} · fin vers {(started + timedelta(hours=hours)):%H:%M}"
        else:
            view["time"] = f"depuis {duration(elapsed)}"
    elif step.status in FINISHED and step.status in step.since:
        view["time"] = f"fini à {datetime.fromisoformat(step.since[step.status]):%H:%M}"
    elif step.status in ("waiting", "setup", "moving", "halted"):
        c = colab_state()
        since = datetime.fromisoformat(c["phase_since"]) if c.get("phase_since") else now
        resume = f" · reprise au pas {c['backup_step']:,}".replace(",", " ") if c.get("backup_step") else ""
        if step.status == "waiting" and c.get("paused_at"):  # pause demandée : la machine rendue, plus de dépense
            def when(t: datetime, at: str = "") -> str:  # « 16:48 », ou « le 22/10 à 16:48 » si ce n'est pas aujourd'hui
                return f"{at}{t:%H:%M}" if t.date() == now.date() else f"le {t:%d/%m à %H:%M}"
            paused = datetime.fromisoformat(c["paused_at"])
            state = (f"machine rendue {when(datetime.fromisoformat(c['released_at']), 'à ')}, plus aucune dépense"
                     if c.get("released_at") else "arrêt propre et sauvegarde sur Kaggle en cours")
            wanted = (f"en attente d'un v6e seulement jusqu'au {datetime.fromtimestamp(c['v5e_after']):%d/%m à %H:%M}, "
                      "puis v5e aussi" if c.get("v5e_after") else
                      "demande d'un v6e, sinon d'un v5e" if c.get("allow_v5e") else "en attente d'un v6e seulement, pas de v5e")
            view["time"] = f"en pause depuis {when(paused)} ({state}) · {wanted}{resume}"
        elif step.status == "waiting":
            lost = f"machine perdue à {datetime.fromisoformat(c['lost_at']):%H:%M} · " if c.get("lost_at") else ""
            view["time"] = f"{lost}en attente d'un v6e depuis {duration((now - since).total_seconds())}{resume}"
        elif step.status == "setup":
            machine = {"V5E1": "v5e", "V6E1": "v6e"}.get(c.get("hardware"), "TPU")
            view["time"] = (f"{machine} obtenu à {since:%H:%M} · préparation depuis {duration((now - since).total_seconds())} "
                            f"(environnement, corpus, sauvegarde){resume}")
        elif step.status == "moving":
            view["time"] = (f"v6e obtenu : transfert depuis {since:%H:%M} (arrêt propre du v5e, sauvegarde, reprise sur le "
                            f"v6e) · {duration((now - since).total_seconds())}")
        else:
            view["time"] = c.get("halted", "suivi arrêté")

    colab = step.spec.get("colab")  # un run sur Colab : son avancement, d'après « pas X/Y » de sa dernière ligne
    progress = parsed["progress"]
    if progress and progress["after"] < len(parsed["train"]):
        progress = None  # une mesure complète (avec validation) est plus récente
    latest = progress or (parsed["train"][-1] if parsed["train"] else None)
    tps = parsed["progress"]["tps"] if parsed["progress"] else None  # chaque minute : sans les pauses d'évaluation
    if colab and latest and latest["planned"] and step.status != "complete":
        view["fraction"] = min(0.99, latest["step"] / latest["planned"])
        tps = tps or latest["tps"]
        if step.status == "running" and tps:
            left = (latest["planned"] - latest["step"]) * latest["tokens"] / max(latest["step"], 1) / tps
            view["time"] = (view["time"] + " · " if view["time"] else "") + f"étape finie dans ~{duration(left)}"
    if step.id == "ab" or (step.spec["kind"] == "local" and not colab):  # les runs de l'A/B
        runs, per_run = step.spec.get("runs", 1), step.spec.get("steps_per_run", 1)
        done = sum(max(p["step"] for p in pts) for pts in parsed["ab"].values())
        if step.status != "complete":
            view["fraction"] = min(0.99, done / (runs * per_run))
        for name, pts in parsed["ab"].items():
            final = parsed["ab_done"].get(name)
            if final:
                view["metrics"].append(f"<b>{name}</b> : {fmt(final['val'])}")
            else:
                view["metrics"].append(f"<b>{name}</b> : pas {int(pts[-1]['step'])} · val {fmt(pts[-1]['val'])}")
    if parsed["corpus"]:  # la préparation du corpus
        c = parsed["corpus"]
        expected = step.spec.get("expected_lines")
        if expected and step.status == "running":  # le débit réel donne une meilleure estimation que la durée prévue
            view["fraction"] = min(0.99, c["lines"] * 1e6 / expected)
            rate = c["lines"] * 1e6 / max(c["s"], 1)
            view["time"] = f"depuis {duration(c['s'])} · reste ~{duration((expected - c['lines'] * 1e6) / max(rate, 1))}"
        view["metrics"] += [f"{fmt(c['lines'], 0)} M répliques lues · {fmt(c['tokens'], 0)} M tokens",
                            f"{fmt(c['simpsons'], 0)} blocs Simpson retirés · {fmt(c['test'], 0)} répliques du test"]
    if parsed["corpus_done"]:
        d = parsed["corpus_done"]
        view["metrics"] = [f"<b>{fmt(d.get('tokens_train', 0) / 1e6, 0)} M tokens</b> d'entraînement",
                           f"{fmt(d.get('blocks_simpsons', 0), 0)} blocs Simpson et {fmt(d.get('lines_ngram', 0), 0)} répliques du test retirés"]
    for b in parsed["bench"]:
        if b.get("ok"):
            view["metrics"].append(f"{b['model']} ×{b['micro_batch']} : {fmt(b['tokens_per_s_per_core'] * 8, 0)} tokens/s (8 cœurs)")
        else:
            view["metrics"].append(f"{b['model']} ×{b['micro_batch']} : <span class='sm-bad'>{b.get('error', 'échec')[:60]}</span>")
    if step.spec.get("stage"):  # une carte limitée à une étape du journal (le pré-entraînement, pas l'affinage qui suit)
        parsed["train"] = [t for t in parsed["train"] if t["stage"] == step.spec["stage"]]
        progress = progress if progress and progress["stage"] == step.spec["stage"] else None
        tps = progress["tps"] if progress else None
    if parsed["train"] or progress:
        last = progress or parsed["train"][-1]
        stage = "affinage" if last["stage"].startswith("dialogues") else "pré-entraînement"
        paused = colab and colab_state().get("paused_at")
        before = (" (avant la pause)" if paused else " (avant la perte de la machine)") if step.status in ("waiting", "setup") else (
            " (avant le transfert)" if step.status == "moving" else "")
        view["metrics"].append(f"<b>{stage}</b>{before} · pas {int(last['step']):,}".replace(",", " ")
                               + (f" / {int(last['planned']):,}".replace(",", " ") if last["planned"] else ""))
        # la validation de l'étape affichée : en affinage, le texte des Simpson ; en pré-entraînement, le corpus général
        # (les deux ne se comparent pas) ; la dernière mesure (toutes les 5 à 10 min) et la meilleure
        measures = [t for t in parsed["train"] if t["stage"] == last["stage"]]
        if measures:
            best = min(measures, key=lambda t: t["val"])
            on = "Simpson" if stage == "affinage" else "corpus général"
            view["metrics"].append(f"val ({on}) {fmt(measures[-1]['val'])} au pas {int(measures[-1]['step'])}"
                                   + (f" · meilleure {fmt(best['val'])} au pas {int(best['step'])}" if best is not measures[-1] else ""))
        view["metrics"].append(f"train {fmt(last['train'])}")
        view["metrics"].append(f"{fmt(last['tokens'] / 1e9, 2)} G tokens · {fmt(tps or last['tps'], 0)} tokens/s")
    c = colab_state() if colab and not step.spec.get("finetune") else {}  # sauvegardes, v6e : le pré-entraînement
    kb = c.get("kaggle_backup") or {}
    if kb.get("step"):  # la dernière sauvegarde de l'état, envoyée sur Kaggle par la machine elle-même
        saved = datetime.fromtimestamp(kb["mtime"])
        when = f"{saved:%H:%M}" if saved.date() == datetime.now().date() else f"{saved:%d/%m %H:%M}"
        view["metrics"].append(f"sauvegardé sur Kaggle : pas {kb['step']:,} (état de {when})".replace(",", " "))
    hunt = c.get("v6e_hunt") or {}
    if hunt.get("tries") and c.get("hardware") == "V5E1":  # sur un v5e : les demandes de v6e (une toutes les ~2 min)
        since, last = datetime.fromisoformat(hunt["since"]), datetime.fromisoformat(hunt["last"])
        view["metrics"].append(f"v6e demandé {hunt['tries']} fois depuis {since:%H:%M} · dernier essai à {last:%H:%M} : "
                               f"{'accordé' if hunt.get('answer') == 'accordé' else 'refusé'}")
    if parsed["result"]:
        r = parsed["result"]
        view["metrics"].append(f"<b>test {fmt(r['test_bits_per_char'])} bits/car.</b> · val {fmt(r['val_bits_per_char'])}")
    if step.status == "error" and parsed["errors"]:
        view["error"] = parsed["errors"][-1][:220]
    return view


def card(step: Step, view: dict) -> str:
    label, cls = STATUS.get(step.status, (step.status, "idle"))
    pct = round(100 * view["fraction"])
    metrics = "".join(f"<div>{m}</div>" for m in view["metrics"][-6:])
    error = f"<div class='sm-error'>{view['error']}</div>" if view.get("error") else ""
    hardware = f"<span class='sm-hw' style='background:{COLORS[step.spec['hardware']]}'>{step.spec['hardware']}</span>" if step.spec.get("hardware") else ""
    return (f"<div class='sm-card {cls}'><div class='sm-card-top'><div class='sm-card-title'>{step.spec['label']} {hardware}</div>"
            f"<span class='sm-badge {cls}'><i></i>{label}</span></div>"
            f"<div class='sm-detail'>{step.spec.get('detail', '')}</div>"
            f"<div class='sm-bar'><div style='width:{pct}%'></div></div>"
            f"<div class='sm-time'><span>{pct} %</span><span>{view['time']}</span></div>"
            f"<div class='sm-metrics'>{metrics}</div>{error}</div>")


def figure(title: str, xaxis: str, yaxis: str) -> go.Figure:
    fig = go.Figure()
    grid = "rgba(148,163,184,0.22)"
    fig.update_layout(title=dict(text=title, x=0.02, font=dict(size=14)), height=350, margin=dict(l=55, r=15, t=46, b=82),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
                      font=dict(family="Inter, ui-sans-serif, system-ui, sans-serif", size=12, color="#64748b"),
                      legend=dict(orientation="h", y=-0.22, x=0, xanchor="left", yanchor="top", font=dict(size=11)),
                      xaxis=dict(title=xaxis, gridcolor=grid, zeroline=False), yaxis=dict(title=yaxis, gridcolor=grid, zeroline=False))
    return fig


def finish(fig: go.Figure, message: str) -> go.Figure:
    if not fig.data:
        fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
                           font=dict(size=13, color="#94a3b8"))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
    return fig


def charts(steps: list[Step], views: dict) -> tuple[go.Figure, ...]:
    """Les quatre graphiques des runs Kaggle : loss du pré-entraînement, vitesse, learning rate, affinage."""
    pretrain = figure("Pré-entraînement · loss (validation pleine, entraînement pointillée)", "tokens vus (milliards)", "loss")
    speed = figure("Vitesse d'entraînement", "heures de session", "tokens / s")
    lr = figure("Learning rate du pré-entraînement (multiplicateur du planning)", "heures de session", "× learning rate")
    finetune = figure("Affinage sur les Simpson · loss de validation", "tokens vus (millions)", "loss")
    for step in steps:
        parsed = views[step.id]["parsed"]
        if not step.spec.get("hardware"):  # seuls les vrais runs (GPU, TPU) ont des courbes
            continue
        color, name = step.spec.get("color") or COLORS[step.spec["hardware"]], step.spec["label"]
        pre = [p for p in parsed["train"] if not p["stage"].startswith("dialogues")]
        fine = [p for p in parsed["train"] if p["stage"].startswith("dialogues")]
        minutes = [p for p in parsed["minutes"] if not p["stage"].startswith("dialogues")]
        if minutes:  # les lignes d'avancement, chaque minute : des courbes sans attendre la validation
            peak = max(p["tps"] for p in minutes)
            minutes = [p for p in minutes if p["tps"] > 0.3 * peak]  # sans les minutes de compilation (≈ 7 000 tokens/s)
            pretrain.add_scatter(x=[p["tokens"] / 1e9 for p in minutes], y=[p["train"] for p in minutes],
                                 name=f"{name} · entraînement", mode="lines", line=dict(color=color, width=1.5, dash="dot"),
                                 opacity=0.6)
            hours = [p["hours"] for p in minutes]
            speed.add_scatter(x=hours, y=[p["tps"] for p in minutes], name=name, mode="lines",
                              line=dict(color=color, width=2.5))
            lr.add_scatter(x=hours, y=[p["lr"] for p in minutes], name=name, mode="lines",
                           line=dict(color=color, width=2.5, shape="linear"))
        if pre:
            tokens = [p["tokens"] / 1e9 for p in pre]
            pretrain.add_scatter(x=tokens, y=[p["val"] for p in pre], name=f"{name} · validation", mode="lines+markers",
                                 line=dict(color=color, width=2.5), marker=dict(size=5))
        if pre and not minutes:
            pretrain.add_scatter(x=tokens, y=[p["train"] for p in pre], name=f"{name} · entraînement", mode="lines",
                                 line=dict(color=color, width=1.5, dash="dot"), opacity=0.6)
            hours = [p["hours"] for p in pre]
            speed.add_scatter(x=hours, y=[p["tps"] for p in pre], name=name, mode="lines+markers",
                              line=dict(color=color, width=2.5), marker=dict(size=5))
            lr.add_scatter(x=hours, y=[p["lr"] for p in pre], name=name, mode="lines+markers",
                           line=dict(color=color, width=2.5, shape="linear"), marker=dict(size=4))
        if fine:
            finetune.add_scatter(x=[p["tokens"] / 1e6 for p in fine], y=[p["val"] for p in fine], name=name,
                                 mode="lines+markers", line=dict(color=color, width=2.5), marker=dict(size=6))
    values = [y for trace in pretrain.data for y in trace.y if y == y]
    if len(values) > 4:  # le tout début (loss ≈ 9) écraserait la suite : on cadre sur ce qui compte
        pretrain.update_yaxes(range=[min(values) - 0.05, min(max(values), min(values) + 2.0)])
    waits = []  # heures de pré-entraînement restantes, d'après la dernière mesure de chaque run
    for step in steps:
        parsed = views[step.id]["parsed"]
        pre = [p for p in parsed["train"] if not p["stage"].startswith("dialogues")]
        progress = parsed["progress"]
        if (step.spec.get("colab") and step.status == "running" and progress and progress["tps"]
                and not progress["stage"].startswith("dialogues")):  # Colab : d'après les pas restants et la vitesse
            per_step = progress["tokens"] / max(progress["step"], 1)
            waits.append((progress["planned"] - progress["step"]) * per_step / progress["tps"] / 3600)
            continue
        if step.spec.get("hardware") and step.status == "running" and pre and len(pre) == len(parsed["train"]):
            planned = dict(parsed["stages"]).get(pre[-1]["stage"])
            if planned:
                waits.append(planned - pre[-1]["hours"])
    later = "après le pré-entraînement" if not waits else (
        f"dans ~{round(min(waits))} h" if min(waits) >= 1 else "dans moins d'une heure")
    return (finish(pretrain, "en attente des premières mesures du run"),
            finish(speed, "en attente des mesures du run"), finish(lr, "en attente des mesures du run"),
            finish(finetune, f"en attente de l'affinage ({later})"))


def results(steps: list[Step], views: dict, pattern: str = "dialogues_*.json") -> str:
    """Deux tableaux : les runs Kaggle (où ils en sont), et les modèles affinés (score de test des Simpson)."""
    rows = []
    for step in steps:
        if not step.spec.get("hardware"):
            continue
        parsed = views[step.id]["parsed"]
        label, _ = STATUS.get(step.status, (step.status, ""))
        last = parsed["train"][-1] if parsed["train"] else None
        progress = parsed["progress"]
        if progress and (last is None or progress["after"] >= len(parsed["train"])):  # plus récente que la dernière mesure
            last = progress | {"val": last["val"] if last and last["stage"] == progress["stage"] else None}
        stage = ("affinage" if last["stage"].startswith("dialogues") else "pré-entraînement") if last else "—"
        best = min((p["val"] for p in parsed["train"] if p["stage"] == last["stage"]), default=None) if last else None
        test = parsed["result"]["test_bits_per_char"] if parsed["result"] else None
        rows.append(f"<tr><td><b>{step.spec['label']}</b></td><td>{label}</td><td>{stage}</td>"
                    f"<td>{fmt(last['tokens'] / 1e9, 2) + ' G' if last else '—'}</td>"
                    f"<td>{fmt(last['val']) if last and last['val'] else '—'}</td><td>{fmt(best) if best else '—'}</td>"
                    f"<td>{fmt((progress or last)['tps'], 0) if last else '—'}</td><td>{fmt(test) if test else '—'}</td></tr>")
    models = []
    for path in sorted((ROOT / "checkpoints").glob(pattern)):
        info = json.loads(path.read_text())
        if "test_bits_per_char" in info:
            arch = "modded" if info.get("arch") == "modern" or "modded" in path.stem else "cours"
            models.append((path.stem, arch, info["test_bits_per_char"], info.get("val_bits_per_char")))
    for step in steps:
        r = views[step.id]["parsed"]["result"]
        if r and step.spec.get("hardware") and not any(abs(m[2] - r["test_bits_per_char"]) < 1e-9 for m in models):
            models.append((f"{step.spec['label']} ({step.spec.get('hardware', '')})", "modded", r["test_bits_per_char"], r.get("val_bits_per_char")))
    best_model = min((m[2] for m in models), default=None)
    models_html = "".join(f"<tr class='{'sm-best' if m[2] == best_model else ''}'><td>{m[0]}</td><td>{m[1]}</td>"
                          f"<td>{fmt(m[2])}</td><td>{fmt(m[3]) if m[3] else '—'}</td></tr>" for m in sorted(models, key=lambda m: m[2]))
    return ("<div class='sm-tables'>"
            "<div class='sm-table'><div class='sm-h'>Runs · en direct</div><table><tr><th>run</th><th>état</th>"
            "<th>étape</th><th>tokens vus</th><th>val</th><th>meilleure val</th><th>tokens/s</th><th>test (bits/car.)</th></tr>"
            f"{''.join(rows)}</table></div>"
            "<div class='sm-table'><div class='sm-h'>Modèles affinés · test des Simpson (bits/car., plus bas = mieux)</div>"
            f"<table><tr><th>modèle</th><th>architecture</th><th>test</th><th>val</th></tr>{models_html}</table></div></div>")


def lab_error(text: str) -> str:
    """La cause d'un essai raté, en quelques mots."""
    if "out of memory" in text or "RESOURCE_EXHAUSTED" in text:
        return "mémoire du TPU dépassée"
    if "Mosaic" in text:
        return "noyau Pallas refusé (libtpu trop ancienne)"
    return text[:70]


def experiment_html(current: dict | None) -> str:
    """L'expérience de vitesse en cours (ou finie depuis moins de 30 min) : ses essais, faits, en cours, à venir."""
    try:  # l'âge du journal de l'expérience au moment de la synchro, plus le temps écoulé depuis
        age = (current.get("age_s") or 0) + time.time() - LAB.stat().st_mtime if current else 0
    except OSError:
        age = 0
    if not current or (current.get("finished") and age > 1800):
        return ""
    items = current["items"]
    done = sum(i["status"] == "done" for i in items)
    rows = []
    for i in items:
        if i["status"] == "done":
            state = (f"<b>{fmt(i['tokens_per_s'], 0)} tokens/s</b>" if i.get("ok") and i.get("tokens_per_s")
                     else f"<span class='sm-bad'>{lab_error(i.get('error') or '')}</span>")
            badge = "✅"
        elif i["status"] == "running":
            state, badge = f"en cours depuis {duration(i.get('elapsed_s') or 0)} (compilation ~2 min, puis 12 pas)", "⏳"
        else:
            state, badge = "à venir", "·"
        rows.append(f"<tr><td>{badge}</td><td>{i['engine']}</td><td><code>{i['label']}</code></td><td>{state}</td></tr>")
    status = "terminée" if current.get("finished") else f"{done} / {len(items)} essais faits"
    return (f"<div class='sm-table sm-exp'><div class='sm-h'>Expérience en cours · {current['name']} · {status}</div>"
            f"<div class='sm-sub'>{current.get('description', '')}</div>"
            f"<div class='sm-bar'><div style='width:{100 * done / max(1, len(items)):.0f}%'></div></div>"
            f"<table>{''.join(rows)}</table></div>")


def lab_html() -> str:
    """Les essais de vitesse sur le TPU v6e (data/colab/lab.json) : les réussis du plus rapide au plus lent, avec la
    référence (le run PyTorch), puis les ratés et leur cause. Le MFU : tokens/s × FLOPs par token / 918 TFLOPS."""
    try:
        lab = json.loads(LAB.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    entries = [e for e in lab["entries"] if "resource busy" not in (e.get("error") or "")]  # TPU pris : pas un résultat
    # tailles de modèle différentes : classés par MFU (la part de la puce utilisée), pas par tokens/s
    done = sorted((e for e in entries if e.get("ok") and e.get("tokens_per_s")), key=lambda e: -e["mfu"])
    failed = [e for e in entries if not (e.get("ok") and e.get("tokens_per_s"))]
    reference = next((e for e in entries if e.get("source") == "run réel"), None)  # le premier run, 16 têtes
    best = max((e["tokens_per_s"] for e in done), default=1)
    rows = []
    for e in done + failed:
        tps = e.get("tokens_per_s")
        width, heads = e.get("width") or 1024, e.get("heads") or 16
        layers = e.get("layers") or 24
        model = f"{heads} × {width // heads}" + ("" if (layers, width) == (24, 1024) else f" · {layers} × {width}")
        if tps:
            speed = (f"<div class='sm-lab'><div class='sm-lab-bar'><div style='width:{100 * tps / best:.0f}%'></div></div>"
                     f"<span>{fmt(tps, 0)}</span></div>")
            extra = (f"<td>{fmt(100 * e['mfu'], 1)} %</td><td>× {fmt(e['mfu'] / reference['mfu'], 2) if reference else '—'}</td>"
                     f"<td>{e.get('source', '')}</td>")
        else:
            speed, extra = "—", f"<td>—</td><td>—</td><td class='sm-bad'>{lab_error(e.get('error') or '')}</td>"
        cls = "sm-best" if done and e is done[0] else ""
        rows.append(f"<tr class='{cls}'><td>{e['engine']}</td><td>{model}</td><td>{e.get('micro_batch') or '—'}</td>"
                    f"<td>{e.get('remat') or '—'}</td><td>{e.get('attn') or '—'}</td><td>{e.get('options') or ''}</td>"
                    f"<td>{speed}</td>{extra}</tr>")
    return (experiment_html(lab.get("current")) +
            "<div class='sm-table'><div class='sm-h'>Essais de vitesse · TPU v6e · 24 couches × 1024 sauf mention · "
            "classés par MFU (tokens/s × FLOPs par token / 918 TFLOPS) · référence : le premier run PyTorch (16 têtes) · "
            f"mis à jour à {lab.get('updated', '?')}</div>"
            "<table><tr><th>moteur</th><th>têtes</th><th>micro-lot</th><th>recalcul</th><th>attention</th><th>options</th>"
            "<th>tokens/s</th><th>MFU</th><th>MFU vs réf.</th><th>source</th></tr>" + "".join(rows) + "</table></div>")


def header(monitor: Monitor) -> str:
    quota = "".join(f"<span class='sm-chip'>{k} : {fmt(used, 1)} h / {fmt(total, 0)} h</span>"
                    for k, (used, total) in monitor.quota.items())
    try:  # les unités Colab dépensées, d'après colab_watch.py
        c = json.loads((ROOT / "data" / "colab" / "state.json").read_text())
        machine = "aucun TPU (pause)" if c.get("released_at") else \
            c.get("hardware", "?").replace("V5E1", "TPU v5e").replace("V6E1", "TPU v6e")
        quota += f"<span class='sm-chip'>Colab : {fmt(c.get('spent', 0), 1)} / {fmt(c.get('cap', 150), 0)} unités · {machine}</span>"
    except (OSError, json.JSONDecodeError):
        pass
    c = colab_state()  # le suivi Colab sans machine : bien visible en haut
    if c.get("phase") in ("attente", "préparation", "transfert", "arrêté"):
        machine = {"V5E1": "v5e", "V6E1": "v6e"}.get(c.get("hardware"), "TPU")
        text = {"attente": "en pause · en attente d'un v6e" if c.get("paused_at") else "TPU perdu · en attente d'un TPU",
                "préparation": f"{machine} obtenu · préparation de la machine",
                "transfert": "v6e obtenu · transfert depuis le v5e", "arrêté": "suivi Colab arrêté"}[c["phase"]]
        since = f" depuis {datetime.fromisoformat(c['phase_since']):%H:%M}" if c.get("phase_since") else ""
        quota += f"<span class='sm-chip' style='color:#dc2626'>{text}{since}</span>"
    updated = f"mis à jour à {monitor.updated:%H:%M:%S}" if monitor.updated else "première lecture…"
    return (f"<div class='sm-header'><div><div class='sm-title'>HomerGPT · suivi</div>"
            f"<div class='sm-sub'>{monitor.title}</div></div><div class='sm-chips'>{quota}"
            f"<span class='sm-chip'><span class='sm-live'></span>{updated} · affiché à {datetime.now():%H:%M:%S}</span></div></div>")


CSS = """
.sm-header {display:flex; justify-content:space-between; align-items:flex-end; gap:14px; flex-wrap:wrap; margin:4px 0 2px}
.sm-title {font-size:24px; font-weight:750; letter-spacing:-0.02em}
.sm-sub {color:var(--body-text-color-subdued); font-size:14px; margin-top:2px}
.sm-chips {display:flex; gap:8px; flex-wrap:wrap}
.sm-chip {border:1px solid var(--border-color-primary); border-radius:999px; padding:5px 11px; font-size:12.5px;
          color:var(--body-text-color-subdued); display:inline-flex; align-items:center; gap:7px}
.sm-live {width:8px; height:8px; border-radius:50%; background:#22c55e; animation:sm-pulse 1.6s infinite}
@keyframes sm-pulse {0%{box-shadow:0 0 0 0 rgba(34,197,94,.55)} 70%{box-shadow:0 0 0 7px rgba(34,197,94,0)} 100%{box-shadow:0 0 0 0 rgba(34,197,94,0)}}
.sm-grid {display:grid; grid-template-columns:repeat(auto-fit, minmax(230px, 1fr)); gap:12px}
.sm-card {border:1px solid var(--border-color-primary); border-radius:16px; padding:14px 15px 13px;
          background:var(--block-background-fill); display:flex; flex-direction:column; gap:7px; min-height:205px}
.sm-card.running {border-color:#3b82f6; box-shadow:0 0 0 1px #3b82f6 inset, 0 6px 18px rgba(59,130,246,.12)}
.sm-card.error {border-color:#ef4444}
.sm-card.done {border-color:rgba(34,197,94,.55)}
.sm-card-top {display:flex; justify-content:space-between; align-items:center; gap:8px}
.sm-card-title {font-weight:700; font-size:15.5px; display:flex; align-items:center; gap:7px}
.sm-hw {color:white; font-size:10.5px; font-weight:700; padding:2px 7px; border-radius:6px; letter-spacing:.03em}
.sm-detail {color:var(--body-text-color-subdued); font-size:12.5px; line-height:1.35}
.sm-badge {font-size:11.5px; font-weight:650; padding:3px 9px; border-radius:999px; display:inline-flex; align-items:center;
           gap:6px; white-space:nowrap}
.sm-badge i {width:7px; height:7px; border-radius:50%; background:currentColor}
.sm-badge.idle {background:rgba(148,163,184,.16); color:#64748b}
.sm-badge.queued {background:rgba(234,179,8,.16); color:#ca8a04}
.sm-badge.running {background:rgba(59,130,246,.15); color:#2563eb}
.sm-badge.running i {animation:sm-pulse 1.6s infinite}
.sm-badge.done {background:rgba(34,197,94,.15); color:#16a34a}
.sm-badge.error {background:rgba(239,68,68,.15); color:#dc2626}
.sm-bar {height:7px; border-radius:99px; background:rgba(148,163,184,.22); overflow:hidden; margin-top:2px}
.sm-bar > div {height:100%; border-radius:99px; background:linear-gradient(90deg,#3b82f6,#8b5cf6); transition:width .8s ease}
.sm-card.done .sm-bar > div {background:#22c55e}
.sm-card.error .sm-bar > div {background:#ef4444}
.sm-time {display:flex; justify-content:space-between; font-size:12px; color:var(--body-text-color-subdued); gap:8px}
.sm-time span:first-child {font-weight:700; color:var(--body-text-color); white-space:nowrap}
.sm-time span:last-child {text-align:right}
.sm-metrics {font-size:12.5px; line-height:1.6; font-variant-numeric:tabular-nums}
.sm-error {color:#dc2626; font-family:ui-monospace,monospace; font-size:11.5px; background:rgba(239,68,68,.08);
           border-radius:8px; padding:6px 8px; word-break:break-word}
.sm-bad {color:#dc2626}
.sm-tables {display:grid; grid-template-columns:repeat(auto-fit, minmax(380px, 1fr)); gap:14px}
.sm-table {border:1px solid var(--border-color-primary); border-radius:16px; padding:12px 14px; background:var(--block-background-fill)}
.sm-h {font-weight:700; font-size:14px; margin-bottom:8px}
.sm-table table {width:100%; border-collapse:collapse !important; border:none !important; font-size:12.5px;
                 font-variant-numeric:tabular-nums; margin:0 !important}
.sm-table tr, .sm-table thead, .sm-table tbody {border:none !important}
.sm-table th, .sm-table td {border:none !important; background:transparent !important}
.sm-table th {text-align:left; color:var(--body-text-color-subdued); font-weight:600; padding:6px 8px !important;
              border-bottom:1px solid var(--border-color-primary) !important; font-size:12px}
.sm-table td {padding:6px 8px !important; border-bottom:1px solid rgba(148,163,184,.18) !important}
.sm-table tr:last-child td {border-bottom:none !important}
.sm-table tr.sm-best td {font-weight:700; color:#16a34a}
.sm-exp {margin-bottom:14px; border-color:#14b8a6 !important}
.sm-exp .sm-bar {margin:8px 0}
.sm-lab {display:flex; align-items:center; gap:8px; min-width:150px}
.sm-lab span {white-space:nowrap}
.sm-lab-bar {flex:1; height:8px; border-radius:4px; background:rgba(148,163,184,.2); overflow:hidden; min-width:60px}
.sm-lab-bar > div {height:100%; background:#14b8a6; border-radius:4px}
.sm-plot {border:1px solid var(--border-color-primary) !important; border-radius:16px !important; overflow:hidden}
footer {display:none !important}
"""


def main() -> None:
    parser = argparse.ArgumentParser(prog="dashboard.py", description="Le suivi de l'expérience, en direct.")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    monitor = Monitor()

    def render(choice: str):
        for step in monitor.steps:
            if isinstance(step, LocalStep):
                step.refresh()  # un fichier local : relu à chaque affichage
        views = {s.id: step_view(s) for s in monitor.steps}
        cards = "<div class='sm-grid'>" + "".join(card(s, views[s.id]) for s in monitor.steps) + "</div>"
        return (header(monitor), cards, lab_html(), *charts(monitor.steps, views),
                results(monitor.steps, views, monitor.checkpoints), log_text(choice))

    def log_text(choice: str) -> str:
        step = next((s for s in monitor.steps if s.spec["label"] == choice), None)
        if step is None:
            return ""
        lines = [line for line in step.lines if visible(line)]
        return "\n".join(lines[-150:]) or "(pas encore de journal)"

    with gr.Blocks(title="HomerGPT · suivi", fill_width=True) as demo:
        top = gr.HTML()
        cards = gr.HTML()
        lab = gr.HTML()
        with gr.Row(equal_height=True):
            pretrain = gr.Plot(show_label=False, elem_classes="sm-plot")
            speed = gr.Plot(show_label=False, elem_classes="sm-plot")
        with gr.Row(equal_height=True):
            ab = gr.Plot(show_label=False, elem_classes="sm-plot")
            finetune = gr.Plot(show_label=False, elem_classes="sm-plot")
        tables = gr.HTML()
        labels = [s.spec["label"] for s in monitor.steps]
        default = next((s.spec["label"] for s in monitor.steps if s.spec.get("hardware")), labels[0])  # le run GPU
        choice = gr.Dropdown(labels, value=default, label="Journal en direct", interactive=True)
        log = gr.Textbox(lines=16, max_lines=16, show_label=False, autoscroll=True, interactive=False, buttons=["copy"])
        outputs = [top, cards, lab, pretrain, speed, ab, finetune, tables, log]
        demo.load(render, choice, outputs)
        gr.Timer(10).tick(render, choice, outputs)
        choice.change(log_text, choice, log)

    print(f"suivi → http://localhost:{args.port}")
    demo.launch(server_name="127.0.0.1", server_port=args.port, css=CSS,
                theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"))
