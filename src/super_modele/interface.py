"""L'interface de discussion, avec Gradio : on choisit le modèle, qui on est et à qui on parle.

La conversation est gardée comme une scène (`[(personnage, réplique), …]`), remise en forme de script
(`Marge: …`) à chaque message : le modèle n'a plus qu'à écrire la réplique suivante.

La liste des modèles est relue toutes les minutes : un modèle rapatrié de Kaggle par `kaggle_watch.py`
apparaît tout seul, avec son score ; l'avancement de l'entraînement sur Kaggle est affiché en haut.
"""

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch

from . import CHECKPOINT_DIRS, find_checkpoint
from .gpt import load_checkpoint, reply

ROOT = Path(__file__).resolve().parents[2]
DIALOGUES = ROOT / "data" / "simpsons_scripts" / "dialogues.jsonl"
KAGGLE_STATUS = ROOT / "data" / "kaggle" / "status.json"
FALLBACK_CHARACTERS = ["Homer", "Marge", "Bart", "Lisa", "Burns", "Moe", "Ned", "Skinner", "Milhouse", "Krusty"]
SESSION_NAMES = {"clic": "👆 attend « Save & Run All » dans l'éditeur Kaggle", "en attente": "⏸ en attente", "queued": "⏳ dans la file", "running": "🟢 en cours",
                 "complete": "✅ terminée", "error": "❌ erreur", "cancel": "⛔ annulée"}


def main_characters(path: Path, n: int = 40) -> list[str]:
    """Les `n` personnages qui parlent le plus dans le dataset, ou une liste par défaut."""
    if not path.exists():
        return FALLBACK_CHARACTERS
    counts = Counter()
    with path.open(encoding="utf-8") as f:
        for line in f:
            counts.update(re.findall(r"^([^:\n]{1,40}): ", json.loads(line)["text"], flags=re.MULTILINE))
    return [name for name, _ in counts.most_common(n)]


def info_of(path: Path) -> dict:
    """Le .json à côté du modèle (score, version, et pour un « -last » : pas, tokens, heure de l'export)."""
    info = path.with_suffix(".json")
    try:
        return json.loads(info.read_text()) if info.exists() else {}
    except (OSError, json.JSONDecodeError):  # en cours de réécriture par export_state.py
        return {}


def score(path: Path) -> float | None:
    """Bits par caractère sur le test des Simpson (plus bas = meilleur), lus dans le .json à côté du modèle."""
    return info_of(path).get("test_bits_per_char")


def is_last(path: Path) -> bool:
    """Le dernier checkpoint d'un entraînement en cours (vX-YM-last, exporté par export_state.py)."""
    return path.stem.endswith("-last")


def dialogue_checkpoints() -> list[Path]:
    """Les modèles qui savent discuter (pas ceux pré-entraînés sur Wikipédia seulement), le meilleur d'abord. Ils
    s'appellent vX-YM (X : le numéro du pré-entraînement, Y : millions de paramètres, puis la variante d'affinage) ;
    dialogues_* : les noms d'avant."""
    found = {p.name: p for d in CHECKPOINT_DIRS if d.is_dir()
             for pattern in ("v[0-9]*-*M*.safetensors", "dialogues_*.safetensors") for p in d.glob(pattern)}
    found = {name: p for name, p in found.items() if ".tmp" not in name}  # un export en cours
    return sorted(found.values(), key=lambda p: (not is_last(p), score(p) is None, score(p) or 0, -p.stat().st_mtime))


def label(path: Path) -> str:
    if is_last(path):  # étiquette fixe : l'avancement (pas, val, heure) est dans la ligne d'infos, rafraîchie
        return f"{path.stem}  ·  dernier checkpoint Colab (pré-entraînement seul)"
    s = score(path)
    return f"{path.stem}  ·  test {s:.3f} bits/car." if s is not None else f"{path.stem}  ·  pas encore de score"


def kaggle_status() -> str:
    """L'avancement sur Kaggle, d'après data/kaggle/status.json (écrit par kaggle_watch.py)."""
    if not KAGGLE_STATUS.exists():
        return "**Kaggle** · pas d'entraînement en cours (`caffeinate -is uv run kaggle_watch.py` pour le lancer)"
    if datetime.now().timestamp() - KAGGLE_STATUS.stat().st_mtime > 24 * 3600:  # kaggle_watch.py arrêté : rien à suivre
        return ""
    state = json.loads(KAGGLE_STATUS.read_text())
    lines = []
    for s, info in sorted(state.get("sessions", {}).items()):
        text = f"{ {'prep': 'corpus', 'gpu1': 'session GPU 1', 'gpu2': 'session GPU 2', 'dialogue': 'corpus de dialogue', 'gpumod': 'GPU modded', 'tpumod': 'TPU modded'}.get(s, s)} : {SESSION_NAMES.get(info['status'], info['status'])}"
        if info["status"] == "running" and info.get("started"):
            started = datetime.fromisoformat(info["started"])
            hours = (datetime.now() - started).total_seconds() / 3600
            end = datetime.fromtimestamp(started.timestamp() + info.get("hours", 0) * 3600)
            text += f" depuis {hours:.1f} h, fin vers {end:%H:%M}"
        if info.get("result"):
            text += f" (test {info['result']['test_bits_per_char']:.3f} bits/car.)"
        lines.append(text)
    checked = (state.get("checked") or state.get("updated", ""))[11:16]
    return "**Kaggle** · " + " · ".join(lines) + f"  \n_{state.get('message', '')} — vérifié à {checked}_"


def main() -> None:
    parser = argparse.ArgumentParser(prog="app.py", description="Interface pour discuter avec le modèle.")
    parser.add_argument("checkpoint", nargs="?", help="fichier .safetensors (par défaut : le meilleur score)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", help="cpu, mps, cuda… (par défaut : le GPU s'il y en a un)")
    args = parser.parse_args()

    device = args.device or (torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu")
    found = dialogue_checkpoints()
    scored = [p for p in found if score(p) is not None]
    first = find_checkpoint(args.checkpoint) if args.checkpoint or not found else (scored or found)[0]
    loaded = {}  # un seul modèle en mémoire à la fois : {(nom, date du fichier): (modèle, tokenizer, infos)}
    characters = main_characters(DIALOGUES)

    def get_model(path: Path):
        key = (path.name, path.stat().st_mtime)  # un « -last » réécrit est rechargé
        if key not in loaded:
            loaded.clear()
            loaded[key] = load_checkpoint(path, device)
        return loaded[key]

    def describe(path: Path) -> str:
        model = get_model(path)[0]
        parts = [f"`{path.name}`", f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M paramètres",
                 f"contexte {model.block_size} tokens"]
        if score(path) is not None:
            parts.append(f"test **{score(path):.3f}** bits/car.")
        if is_last(path):
            info = info_of(path)
            when = (info.get("saved") or info.get("exported", ""))[11:16]  # l'heure de l'état sur Colab
            parts.append(f"pas **{info.get('step', 0):,}** · {info.get('tokens', 0) / 1e9:.2f} G tokens · "
                         f"val **{info.get('val_loss', float('nan')):.3f}** · état de {when}".replace(",", " "))
        return " · ".join(parts + [device])

    def options() -> dict[str, Path]:
        return {label(p): p for p in dialogue_checkpoints()} or {label(first): first}

    def chosen(choice: str) -> Path:
        """Le modèle d'une étiquette ; celle d'un « -last » change à chaque export : on retrouve le fichier par son nom."""
        current = options()
        stem = choice.split("  ·")[0]
        return current.get(choice) or next((p for p in current.values() if p.stem == stem), first)

    def respond(message, chat, scene, choice, me, character, temperature, candidates):
        message, me, character = message.strip(), me.strip(), character.strip()
        if not (message and me and character):
            return message, chat, scene
        model, tokenizer, _ = get_model(chosen(choice))
        script = "".join(f"{speaker}: {text}\n" for speaker, text in scene[-30:])
        prompt = script + f"{me}: {tokenizer.known(message)}\n{character}:"
        text = reply(model, tokenizer, prompt, character, device, candidates=int(candidates), temperature=temperature,
                     said=[t for s, t in scene if s == character],  # pour éviter les redites
                     echo_penalty=2.0)  # et les reprises du message (« Krusty Burger » → « on va chez Krusty Burger »)
        chat = chat + [{"role": "user", "content": f"**{me}** : {message}"},
                       {"role": "assistant", "content": f"**{character}** : {text}"}]
        return "", chat, scene + [(me, message), (character, text)]

    def switch(choice):
        return describe(chosen(choice)), [], []  # nouveau modèle : nouvelle conversation

    def refresh(choice):
        labels = list(options())
        stem = choice.split("  ·")[0]  # garde le même fichier, même si son étiquette a changé (nouvel export)
        value = next((lab for lab in labels if lab.split("  ·")[0] == stem), labels[0])
        return gr.update(choices=labels, value=value), kaggle_status(), describe(options()[value])

    with gr.Blocks(title="HomerGPT") as demo:
        gr.Markdown("## HomerGPT")
        status = gr.Markdown(kaggle_status())
        with gr.Row():
            model_choice = gr.Dropdown(list(options()), value=label(first), scale=3,
                                       label="Modèle (le meilleur score en premier ; plus bas = meilleur)")
            refresh_button = gr.Button("↻", scale=0, min_width=60)
        about = gr.Markdown(describe(first))
        with gr.Row():
            me = gr.Dropdown(characters, value="Bart", label="Je suis", allow_custom_value=True)
            swap = gr.Button("⇄", scale=0, min_width=60)
            character = gr.Dropdown(characters, value="Homer", label="Je parle à", allow_custom_value=True)
            temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Température")
            candidates = gr.Dropdown([1, 4, 8, 16, 32], value=16, label="Réponses essayées",
                                     info="le modèle en écrit plusieurs et garde la plus liée au message")
        chat = gr.Chatbot(height=460, placeholder="Choisis qui tu es et à qui tu parles, puis écris un message.")
        scene = gr.State([])
        message = gr.Textbox(placeholder="Écris un message… (Entrée pour envoyer)", show_label=False, autofocus=True)
        with gr.Row():
            send = gr.Button("Envoyer", variant="primary")
            reset = gr.Button("Nouvelle conversation")

        inputs = [message, chat, scene, model_choice, me, character, temperature, candidates]
        message.submit(respond, inputs, [message, chat, scene])
        send.click(respond, inputs, [message, chat, scene])
        swap.click(lambda a, b: (b, a), [me, character], [me, character])
        reset.click(lambda: ([], []), None, [chat, scene])
        model_choice.change(switch, model_choice, [about, chat, scene])
        refresh_button.click(refresh, model_choice, [model_choice, status, about])
        gr.Timer(60).tick(refresh, model_choice, [model_choice, status, about])  # nouveaux modèles, nouvelles de Kaggle

    print(f"modèle {first.name} chargé ({device}) → http://localhost:{args.port}")
    demo.launch(server_name="127.0.0.1", server_port=args.port)
