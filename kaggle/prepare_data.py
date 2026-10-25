"""Le corpus de pré-entraînement, encodé une fois pour toutes, dans une session Kaggle sans accélérateur.

    python prepare_data.py          # → /kaggle/working/data/{wiki,web}_{train,val}.bin (uint16)

- Wikipédia FR en entier (17 fichiers), nettoyé comme dans transformer_from_scratch.ipynb ;
- du web français (FineWeb-2, fra_Latn), un registre plus courant que celui d'une encyclopédie,
  jusqu'à WEB_TOKENS tokens.

Un document sur 200 va à la validation. Les tokens sont ceux du BPE du notebook (tokenizer.json, 8 192
tokens) : ils tiennent sur 16 bits. La session TPU reprend ces fichiers en entrée : le TPU ne passe pas
son temps (compté) à attendre le CPU.
"""

import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

INPUT = Path(glob.glob("/kaggle/input/**/gpt.py", recursive=True)[0]).parent if glob.glob(
    "/kaggle/input/**/gpt.py", recursive=True) else Path(__file__).resolve().parent
sys.path.insert(0, str(INPUT))
import gpt  # noqa: E402

OUT = Path(os.environ.get("OUT", "/kaggle/working")) / "data"
TMP = Path(os.environ.get("TMP_DIR", "/tmp/super-modele"))
SMOKE = os.environ.get("SMOKE") == "1"
WIKI_FILES = 1 if SMOKE else 17
WEB_TOKENS = int(os.environ.get("WEB_TOKENS", 2_000_000 if SMOKE else 3_500_000_000))

END_SECTIONS = re.compile(r"^(Notes et références|Références|Notes|Voir aussi|Liens externes|Bibliographie|"
                          r"Articles connexes|Annexes|Sources|Filmographie|Discographie)\s*$", re.MULTILINE)
SPACE_TABLE = str.maketrans({" ": " ", " ": " ", " ": " "})


def clean_article(text: str) -> str:
    """Le nettoyage du notebook : sans sections de fin, sans lignes vides en trop, sans espaces insécables."""
    end = END_SECTIONS.search(text)
    text = text[: end.start()] if end else text
    return re.sub(r"\n{3,}", "\n\n", text).strip().translate(SPACE_TABLE)


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


class Writer:
    """Écrit les tokens d'un corpus dans {name}_train.bin et {name}_val.bin, un document sur 200 en validation."""

    def __init__(self, name: str, tokenizer) -> None:
        self.name, self.tokenizer, self.sep = name, tokenizer, tokenizer.encode("\n\n")
        self.files = {part: open(OUT / f"{name}_{part}.bin", "wb") for part in ("train", "val")}
        self.tokens = {"train": 0, "val": 0}
        self.docs = self.chars = 0

    def add(self, texts: list[str]) -> None:
        for text, ids in zip(texts, self.tokenizer.encode_batch(texts)):
            part = "val" if self.docs % 200 == 0 else "train"
            self.docs += 1
            arr = np.asarray(ids + self.sep, dtype=np.uint16)
            arr.tofile(self.files[part])
            self.tokens[part] += len(arr)
        self.chars += sum(map(len, texts))

    def close(self) -> dict:
        for f in self.files.values():
            f.close()
        stats = {"docs": self.docs, "chars": self.chars, **self.tokens,
                 "chars_per_token": self.chars / max(1, sum(self.tokens.values()))}
        log(f"{self.name} : {stats}")
        return stats


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    tokenizer = gpt.tokenizer_from_state(json.loads((INPUT / "tokenizer.json").read_text(encoding="utf-8")))
    stats, t0 = {}, time.time()

    wiki = Writer("wiki", tokenizer)
    for shard in range(WIKI_FILES):
        path = hf_hub_download("wikimedia/wikipedia", f"20231101.fr/train-{shard:05d}-of-00017.parquet",
                               repo_type="dataset", local_dir=str(TMP))
        for batch in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=2_000):
            texts = [t for t in map(clean_article, batch.column("text").to_pylist()) if len(t) >= 500]
            wiki.add(texts[:300] if SMOKE else texts)
            if SMOKE:
                break
        os.remove(path)
        log(f"Wikipédia {shard + 1}/{WIKI_FILES} : {wiki.tokens['train'] / 1e6:,.0f} M tokens ({time.time() - t0:.0f} s)")
    stats["wiki"] = wiki.close()

    web = Writer("web", tokenizer)
    shard = 0
    while web.tokens["train"] < WEB_TOKENS:
        path = hf_hub_download("HuggingFaceFW/fineweb-2", f"data/fra_Latn/train/000_{shard:05d}.parquet",
                               repo_type="dataset", local_dir=str(TMP))
        for batch in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=2_000):
            texts = [t.strip().translate(SPACE_TABLE) for t in batch.column("text").to_pylist()]
            web.add([t for t in texts if len(t) >= 200])
            if web.tokens["train"] >= WEB_TOKENS:
                break
            if web.docs % 200_000 < 2_000:
                log(f"web : {web.tokens['train'] / 1e6:,.0f} M tokens ({time.time() - t0:.0f} s)")
        os.remove(path)
        shard += 1
    stats["web"] = web.close()
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    log(f"terminé en {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
