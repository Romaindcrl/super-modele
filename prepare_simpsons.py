"""Convertit les scripts des Simpson (CSV du crawler) en dialogues « Personnage: réplique » pour le notebook.

    uv run prepare_simpsons.py [entrée.csv] [sortie.jsonl]

Par défaut : data/simpsons_scripts/scripts.csv → data/simpsons_scripts/dialogues.jsonl, un épisode par
ligne ({"title", "text"}), une réplique par ligne (`Marge: …`), les scènes séparées par une ligne vide.
Les variantes d'un même nom sont fusionnées (« M. Burns » → « Burns ») ; les indications de mise en scène,
les en-têtes de séquence et les remarques entre parenthèses sont retirés.
"""

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import ftfy

ROOT = Path(__file__).resolve().parent
DEFAULT_IN = ROOT / "data" / "simpsons_scripts" / "scripts.csv"
DEFAULT_OUT = ROOT / "data" / "simpsons_scripts" / "dialogues.jsonl"

# Une étiquette de personnage : un nom court, sans ponctuation de phrase (« M. Burns », « Grand-père de Milhouse »).
LABEL_SHAPE = re.compile(r"^[A-ZÀ-ÖØ-Þ0-9][\w'’\-\.,() ]{0,38}$")
# Premiers mots qui trahissent un bout de phrase pris pour un nom par le crawler (« Tu préfères : »).
SENTENCE_STARTS = {"je", "j'", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles", "alors", "et", "mais",
                   "très", "bien", "voilà", "voici", "écoute", "regarde", "dis", "aucun", "en", "le", "la", "les",
                   "l'", "un", "une", "ce", "c'est", "oui", "non", "si", "quand", "comme", "pour", "avec"}
SEQ_HEADER = re.compile(r"\bSEQ\s*\d+\b")  # « SEQ 7 - INT/JOUR … », « SEQ 7 : … », « SEQ 32 EXT/… »
HOMER = re.compile(r"^(?:Jeune |Voix d'|Fantôme d')?Homer(?: Simpson)?(?: \(.*\)| jeune| enfant| vieux| fantôme"
                   r"| du passé| d'aujourd'hui| \d+ ans)?$")
ALIASES = {"M. Burns": "Burns", "Dr Hibbert": "Hibbert", "Chef Wiggum": "Wiggum", "Révérend Lovejoy": "Lovejoy",
           "Pr Frink": "Frink", "Kent Brockman": "Kent", "Flanders": "Ned"}


def label_name(item: str) -> str:
    """« Bart : » → « Bart »."""
    return item.strip().removesuffix(":").strip()


def speaker(name: str) -> str:
    """Le nom sous lequel la réplique apparaît dans le dataset : variantes fusionnées."""
    return "Homer" if HOMER.match(name) else ALIASES.get(name, name)


def is_label(name: str, counts: Counter) -> bool:
    """Vrai si `name` ressemble à un nom de personnage, pas à un bout de réplique."""
    if not LABEL_SHAPE.match(name) or "SEQ" in name or len(name.split()) > 6:
        return False
    if counts[name] >= 2:
        return True
    first = name.split()[0].lower()
    return "," not in name and len(name.split()) <= 4 and first not in SENTENCE_STARTS and not first.startswith(("j'", "l'", "c'"))


def strip_directions(s: str) -> str:
    """Retire les indications [entre crochets] (même imbriquées) et les remarques (entre parenthèses)."""
    while (t := re.sub(r"\[[^\[\]]*\]", " ", s)) != s:
        s = t
    s = s.split("[")[0]  # crochet ouvert jamais refermé : la suite est une indication
    s = re.sub(r"\([^()]*\)", " ", s).replace("]", " ")
    # ftfy répare l'encodage : caractères Windows-1252 mal décodés (\x92 → ’, \x8c → Œ…), mojibake, NFC ;
    # les caractères de contrôle restants (U+0080 à U+009F) et les � ne sont pas du texte.
    s = re.sub(r"[\x80-\x9f�]", "", ftfy.fix_text(s))
    return re.sub(r"\s+", " ", s).strip()


def split_embedded(name: str, segment: str, embedded: re.Pattern) -> list[tuple[str, str]]:
    """Coupe un segment aux étiquettes fréquentes que le crawler n'a pas listées (« … ! Kent : Il est 3 h … »)."""
    parts, start = [], 0
    for m in embedded.finditer(segment):
        parts.append((name, segment[start : m.start()]))
        name, start = m.group(1), m.end()
    parts.append((name, segment[start:]))
    return parts


def convert_episode(text: str, items: list[str], counts: Counter, embedded: re.Pattern) -> tuple[str, Counter]:
    """Le script d'un épisode, mis en forme ; et le nombre de répliques par personnage."""
    # 1. Retrouver chaque étiquette du crawler dans le texte, dans l'ordre, pour ne jamais se décaler.
    marks = []  # (début, fin, nom) des étiquettes valides
    pos = 0
    for item in items:
        i = text.find(item.strip(), pos)
        if i == -1:
            continue
        pos = i + len(item.strip())
        name = label_name(item)
        if is_label(name, counts) and (i == 0 or text[i - 1] in " ]\n"):
            marks.append((i, pos, name))

    # 2. Le texte entre deux étiquettes est la réplique de la première ; un en-tête SEQ coupe la scène.
    scenes, current, roles = [], [], Counter()
    for k, (_, end, name) in enumerate(marks):
        nxt = marks[k + 1][0] if k + 1 < len(marks) else len(text)
        segment = text[end:nxt]
        header = SEQ_HEADER.search(segment)
        for who, part in split_embedded(name, segment[: header.start()] if header else segment, embedded):
            speech = strip_directions(part)
            if speech:
                roles[speaker(who)] += 1
                current.append(f"{speaker(who)}: {speech}")
        if header and current:
            scenes.append("\n".join(current))
            current = []
    if current:
        scenes.append("\n".join(current))
    return "\n\n".join(scenes), roles


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IN
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    csv.field_size_limit(2**31 - 1)
    with src.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("full_script_text", "").strip()]
    episodes = [(r["episode_title"].strip(), r["full_script_text"], json.loads(r["character_names"])) for r in rows]
    counts = Counter(label_name(item) for _, _, items in episodes for item in items)
    frequent = sorted((n for n, c in counts.items() if c >= 20 and is_label(n, counts)), key=len, reverse=True)
    embedded = re.compile(r"(?<=[.!?…»\]] )(" + "|".join(map(re.escape, frequent)) + r") : ")

    seen, total, speakers = set(), Counter(), Counter()
    with dst.open("w", encoding="utf-8") as out:
        for title, text, items in episodes:
            digest = hashlib.sha1(text.encode()).hexdigest()
            if digest in seen:
                continue  # épisode en double
            seen.add(digest)
            dialogue, roles = convert_episode(text, items, counts, embedded)
            if not roles:
                continue
            speakers.update(roles)
            total["épisodes"] += 1
            total["caractères"] += len(dialogue)
            out.write(json.dumps({"title": title, "text": dialogue}, ensure_ascii=False) + "\n")
    print(f"{total['épisodes']} épisodes   {sum(speakers.values()):,} répliques   {len(speakers):,} personnages   "
          f"{total['caractères']:,} caractères → {dst}")
    print("personnages principaux : " + ", ".join(f"{n} ({c:,})" for n, c in speakers.most_common(20)))


if __name__ == "__main__":
    main()
