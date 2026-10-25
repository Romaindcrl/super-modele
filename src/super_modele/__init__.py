"""super-modele : discuter avec le GPT entraîné par transformer_from_scratch.ipynb."""

import argparse
import sys
from pathlib import Path

import torch

from .gpt import generate, load_checkpoint
from .gpt import reply as best_reply

DEFAULT_TEMPLATE = "{moi}: {message}\n{perso}:"
CHECKPOINT_DIRS = [Path.cwd() / "checkpoints", Path(__file__).resolve().parents[2] / "checkpoints"]


def find_checkpoint(name: str | None) -> Path:
    """Le checkpoint demandé, ou le plus récent du dossier `checkpoints/`."""
    if name:
        path = Path(name).expanduser()
        if not path.exists():
            sys.exit(f"{path} introuvable")
        return path
    found = [p for d in CHECKPOINT_DIRS if d.is_dir() for p in d.glob("*.safetensors")]
    if not found:
        sys.exit("aucun checkpoint dans checkpoints/ : entraîner d'abord le modèle avec transformer_from_scratch.ipynb")
    return max(found, key=lambda p: p.stat().st_mtime)


def stream(model, tokenizer, prompt: str, max_new_tokens: int, device: str, stop: str | None, **sampling) -> str:
    """Génère la suite de `prompt` en l'affichant au fil de l'eau ; s'arrête à `stop`. Renvoie le texte généré."""
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    start = idx.shape[1]
    shown = ""
    for _ in range(max_new_tokens):
        idx = generate(model, idx, 1, **sampling)
        text = tokenizer.decode(idx[0, start:].tolist())
        if stop is not None and stop in text:
            text = text[: text.index(stop)]
            print(text[len(shown) :], end="", flush=True)
            return text
        if not text.endswith("�"):  # un caractère multi-octets incomplet (BPE) : on attend la suite
            print(text[len(shown) :], end="", flush=True)
            shown = text
    return tokenizer.decode(idx[0, start:].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="super-modele", description="Discuter avec le modèle entraîné dans transformer_from_scratch.ipynb."
    )
    parser.add_argument("checkpoint", nargs="?", help="fichier .safetensors (par défaut : le plus récent de checkpoints/)")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="< 1 plus sûr, > 1 plus inventif, 0 = toujours le plus probable (défaut : 0.6)")
    parser.add_argument("--top-p", type=float, default=0.9, help="noyau de probabilité cumulée (défaut : 0.9)")
    parser.add_argument("--perso", default="Homer",
                        help="le personnage qui répond : Homer, Marge, Bart, Lisa, Burns, Moe, Ned… (défaut : Homer)")
    parser.add_argument("--moi", default="Homme",
                        help="le personnage que tu joues ; « Homme » = un passant anonyme (défaut : Homme)")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE,
                        help=f"mise en forme d'un message (défaut : {DEFAULT_TEMPLATE!r}) ; ex. pour l'ancien "
                             "modèle HomerSimpson : 'Human: {message}\\nHomer:'")
    parser.add_argument("--libre", type=int, metavar="N", help="au lieu de discuter, génère N tokens de script libre")
    parser.add_argument("--essais", type=int, default=16,
                        help="répliques écrites en parallèle ; on garde la plus liée au message (défaut : 16 ; "
                             "1 = la première qui n'est pas une redite)")
    parser.add_argument("--sans-memoire", action="store_true",
                        help="chaque réponse ne voit que le dernier message, pas la conversation")
    args = parser.parse_args()

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    path = find_checkpoint(args.checkpoint)
    model, tokenizer, checkpoint = load_checkpoint(path, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"modèle : {path.name}   {n_params / 1e6:.1f} M paramètres   val {checkpoint['val_loss']:.4f}   "
          f"(pas {checkpoint['step']:,}, {device})")
    sampling = dict(temperature=args.temperature, top_p=args.top_p)

    if args.libre:
        stream(model, tokenizer, "\n", args.libre, device, None, **sampling)
        print()
        return

    perso, moi = args.perso, args.moi
    print(f"Tu es {moi}, tu parles à {perso}. Écris un message, Entrée pour l'envoyer.\n"
          "/perso NOM : parler à quelqu'un d'autre   /moi NOM : changer qui tu es   "
          "/reset : oublier la conversation   /quit ou Ctrl+D : quitter\n")
    history, said = "", []  # said : ce que le personnage a déjà répondu, pour éviter les redites
    while True:
        try:
            message = input(f"{moi} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message in ("/quit", "/q", "quit", "exit"):
            break
        if message == "/reset":
            history, said = "", []
            print("(conversation oubliée)\n")
            continue
        if message.startswith(("/perso ", "/moi ")):
            command, _, name = message.partition(" ")
            if command == "/perso":
                perso, said = name.strip(), []
            else:
                moi = name.strip()
            print(f"(tu es {moi}, tu parles à {perso})\n")
            continue
        clean = tokenizer.known(message)
        if not clean.strip():
            continue
        if clean != message:
            print("(caractères inconnus du modèle retirés)")
        prompt = ("" if args.sans_memoire else history) + args.template.format(message=clean, moi=moi, perso=perso)
        print(f"{perso} >", end="", flush=True)
        reply = " " + best_reply(model, tokenizer, prompt, perso, device, candidates=args.essais,
                                 said=[] if args.sans_memoire else said, **sampling)
        print(reply, end="", flush=True)
        print("\n")
        said.append(reply.strip())
        history = (prompt + reply + "\n")[-4_000:]  # le modèle ne lit de toute façon que les block_size derniers
