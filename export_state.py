"""Le dernier état d'un pré-entraînement (pretrain_state.pt de kaggle/train_tpu.py, rapatrié de Colab par
colab_watch.py) → un checkpoint que le chat sait charger, sans l'optimiseur.

    uv run export_state.py data/colab/state/pretrain_state.pt checkpoints/v6-591M-last.safetensors

Le modèle n'est pas encore affiné sur les Simpson : il parle comme ses sous-titres, son web et son Wikipédia. Le
fichier est écrit à côté puis renommé (le chat ne lit jamais un fichier à moitié écrit), avec son .json : version,
pas, tokens vus, meilleure loss de validation, heure de l'export.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from super_modele import gpt  # noqa: E402

TOKENIZER = ROOT / "data" / "kaggle" / "upload" / "tokenizer.json"


class Weights:
    """Ce que gpt.save_checkpoint demande d'un modèle : ses poids (pas besoin de le construire)."""

    def __init__(self, tensors: dict) -> None:
        self.tensors = tensors

    def state_dict(self) -> dict:
        return self.tensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("state", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--head-dim", type=int, default=128, help="dimension d'une tête (n_head = n_embd / head_dim)")
    parser.add_argument("--block", type=int, default=512)
    args = parser.parse_args()

    # la meilleure loss est un scalaire numpy : seuls ces types-là sont autorisés en plus des tenseurs
    allowed = [np._core.multiarray.scalar, np.dtype] + [type(np.dtype(t)) for t in ("float32", "float64", "int64")]
    with torch.serialization.safe_globals(allowed):
        state = torch.load(args.state, map_location="cpu", weights_only=True)
    weights = state["model"]
    vocab, n_embd = weights["embed.weight"].shape
    n_layer = len({k.split(".")[1] for k in weights if k.startswith("blocks.")})
    config = {"vocab_size": vocab, "block_size": args.block, "n_embd": n_embd, "n_head": n_embd // args.head_dim,
              "n_layer": n_layer, "dropout": 0.0, "arch": "modern"}
    tokenizer = gpt.tokenizer_from_state(json.loads(TOKENIZER.read_text(encoding="utf-8")))
    info = {"step": int(state["step"]), "tokens": int(state["tokens"]), "val_loss": float(state["best"])}

    tmp = args.out.with_name(args.out.stem + ".tmp.safetensors")
    gpt.save_checkpoint(tmp, Weights(weights), config, tokenizer, **info)
    os.replace(tmp, args.out)
    params = sum(t.numel() for t in weights.values())
    args.out.with_suffix(".json").write_text(json.dumps(
        {"version": args.out.stem, "last": True, "params": params, **info, "exported": datetime.now().isoformat(timespec="seconds"),
         "description": "dernier état du pré-entraînement rapatrié de Colab, pas encore affiné sur les Simpson"},
        indent=2, ensure_ascii=False))
    print(f"{args.out.name} : pas {info['step']:,}, {info['tokens'] / 1e9:.2f} G tokens, val {info['val_loss']:.4f}, "
          f"{params / 1e6:.0f} M paramètres")


if __name__ == "__main__":
    main()
