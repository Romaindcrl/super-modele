"""Le corpus de dialogue : les sous-titres français d'OpenSubtitles 2024 (OPUS), nettoyés, décontaminés, encodés.

    python prepare_dialogue.py [chemin ou URL de fr.txt.gz]      # → OUT/data/subs_{train,val}.bin (uint16)

Des sous-titres de films et de séries : du dialogue, le registre des Simpson (Wikipédia en est le plus loin).
Le fichier garde l'ordre des films : on regroupe les répliques par blocs de 100, séparés par une ligne vide ;
un bloc sur 200 va à la validation.

Nettoyage : tirets de changement de locuteur, bruitages « (rire) », « [musique] », ♪, balises <i>.
Décontamination : les Simpson sont eux-mêmes sous-titrés. On retire leurs épisodes (un bloc qui nomme au moins
deux personnages de la famille, ou « Simpson » et l'un d'eux, et ses voisins s'ils en nomment un), puis toute
réplique qui partage 10 mots de suite avec la validation ou le test des Simpson : sinon le modèle aurait vu
des répliques du test, et son score serait faussé.

Variables : OUT (dossier de sortie, /kaggle/working par défaut), MAX_TOKENS (s'arrêter avant), SMOKE=1.
"""

import glob
import gzip
import json
import os
import re
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

INPUT = Path(glob.glob("/kaggle/input/**/gpt.py", recursive=True)[0]).parent if glob.glob(
    "/kaggle/input/**/gpt.py", recursive=True) else Path(__file__).resolve().parent
sys.path.insert(0, str(INPUT))
import gpt  # noqa: E402

URL = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/mono/fr.txt.gz"
OUT = Path(os.environ.get("OUT", "/kaggle/working")) / "data"
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 10**12))
BLOCK_LINES = 100
NGRAM = 10  # l'apostrophe coupe les mots : 8 « mots » attrapaient des phrases banales

FAMILY = re.compile(r"\b(Homer|Marge|Bart|Lisa|Maggie|Milhouse|Flanders|Krusty|Smithers|Apu|Skinner|Wiggum)\b")
NOISE = re.compile(r"<[^>]*>|\([^()]*\)|\[[^\[\]]*\]|♪[^♪]*♪|[♪♫#]")  # balises, bruitages, musique
DASH = re.compile(r"^\s*[-–—]+\s*|\s+[-–—]\s+(?=[A-ZÀ-ÖØ-Þ])")  # tirets de changement de locuteur
LETTER = re.compile(r"[^\W\d_]")


def clean_line(line: str) -> str:
    """Une réplique sans bruitage, balise ni tiret ; vide si elle ne contient pas de lettre."""
    line = NOISE.sub(" ", line)
    line = DASH.sub(" ", line)
    line = re.sub(r"\s+", " ", line).strip(" :;-–—")
    return line if LETTER.search(line) else ""


def words(text: str) -> list[str]:
    return re.sub(r"\W+", " ", text.lower()).split()


def ngram_keys(text: str) -> set[int]:
    """Les suites de NGRAM mots d'un texte, hachées (crc32 : stable d'un processus à l'autre)."""
    w = words(text)
    return {zlib.crc32(" ".join(w[i : i + NGRAM]).encode()) for i in range(len(w) - NGRAM + 1)}


def simpsons_ngrams() -> set[int]:
    """Les suites de NGRAM mots de la validation et du test des Simpson (sans les noms de locuteurs)."""
    keys = set()
    for part in ("val", "test"):
        for line in (INPUT / f"simpsons_{part}.txt").read_text(encoding="utf-8").split("\n"):
            keys |= ngram_keys(line.partition(": ")[2] or line)
    return keys


def is_simpsons(block: list[str]) -> tuple[bool, int]:
    """(épisode des Simpson ?, nombre de personnages de la famille nommés)."""
    text = "\n".join(block)
    names = set(FAMILY.findall(text))  # « Simpson » seul ne suffit pas : Mme Simpson des « Vampires » (1915)…
    return len(names) >= 2 or ("Simpson" in text and len(names) >= 1), len(names)


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else URL
    if source.startswith("http"):
        local = Path("/tmp/opensubtitles_fr.txt.gz")
        if not local.exists():
            log(f"téléchargement de {source}")
            subprocess.run(["curl", "-sSL", "--retry", "8", "--retry-delay", "5", "-o", str(local), source], check=True)
        source = str(local)
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = gpt.tokenizer_from_state(json.loads((INPUT / "tokenizer.json").read_text(encoding="utf-8")))
    sep = tokenizer.encode("\n\n")
    banned = simpsons_ngrams()
    files = {part: open(OUT / f"subs_{part}.bin", "wb") for part in ("train", "val")}
    stats = {"lines_in": 0, "lines_kept": 0, "lines_ngram": 0, "blocks": 0, "blocks_simpsons": 0,
             "tokens_train": 0, "tokens_val": 0, "chars": 0}
    t0 = time.time()
    pending: list[list[str]] = []  # blocs prêts, encodés par paquets (encode_batch est parallèle)
    window: list[tuple[list[str], bool, int]] = []  # bloc précédent, en attente de son voisin

    def flush() -> None:
        texts = ["\n".join(b) for b in pending]
        for text, ids in zip(texts, tokenizer.encode_batch(texts)):
            part = "val" if stats["blocks"] % 200 == 0 else "train"
            stats["blocks"] += 1
            arr = np.asarray(ids + sep, dtype=np.uint16)
            arr.tofile(files[part])
            stats[f"tokens_{part}"] += len(arr)
            stats["chars"] += len(text)
        pending.clear()

    def emit(block: list[str], flagged: bool, n_names: int, neighbour_flagged: bool) -> None:
        if flagged or (n_names >= 1 and neighbour_flagged):
            stats["blocks_simpsons"] += 1
            return
        kept = []
        for line in block:
            if len(words(line)) >= NGRAM and ngram_keys(line) & banned:
                stats["lines_ngram"] += 1
            else:
                kept.append(line)
        stats["lines_kept"] += len(kept)
        if kept:
            pending.append(kept)
        if len(pending) >= 2_000:
            flush()

    block: list[str] = []
    try:
        with gzip.open(source, "rt", encoding="utf-8", errors="replace") as f:
            for raw in f:
                stats["lines_in"] += 1
                line = clean_line(raw)
                if line:
                    block.append(line)
                if len(block) == BLOCK_LINES:
                    flagged, n = is_simpsons(block)
                    window.append((block, flagged, n))
                    if len(window) == 3:  # le bloc du milieu connaît ses deux voisins
                        (_, f0, _), (b1, f1, n1), (_, f2, _) = window
                        emit(b1, f1, n1, f0 or f2)
                        window.pop(0)
                    block = []
                    if stats["tokens_train"] >= MAX_TOKENS:
                        break
                if stats["lines_in"] % 5_000_000 == 0:
                    log(f"{stats['lines_in'] / 1e6:,.0f} M lignes lues   {stats['tokens_train'] / 1e6:,.0f} M tokens   "
                        f"{stats['blocks_simpsons']} blocs Simpson retirés   {stats['lines_ngram']} répliques du test "
                        f"retirées   ({time.time() - t0:.0f} s)")
    except EOFError:  # fichier encore en cours de téléchargement (essai local) : on garde ce qui est lu
        log("fin de fichier inattendue : on s'arrête là")
    for i, (b, fl, n) in enumerate(window):
        neighbours = [window[j][1] for j in (i - 1, i + 1) if 0 <= j < len(window)]
        emit(b, fl, n, any(neighbours))
    flush()
    for f in files.values():
        f.close()
    stats["chars_per_token"] = stats["chars"] / max(1, stats["tokens_train"] + stats["tokens_val"])
    stats["minutes"] = round((time.time() - t0) / 60, 1)
    (OUT / "subs_stats.json").write_text(json.dumps(stats, indent=2))
    log(f"terminé : {json.dumps(stats)}")


if __name__ == "__main__":
    main()
