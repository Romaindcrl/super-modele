"""Pré-entraînement d'un GPT taille GPT-2 sur Wikipédia FR, puis affinage sur les Simpson : la version Kaggle.

    python train_gpt2.py prepare                                     # session 1 : Wikipédia → tokens
    torchrun --standalone --nproc_per_node=2 train_gpt2.py train     # pré-entraînement + affinage + test
    INIT=modele.safetensors python train_gpt2.py train               # affinage seul (FT_OPTIM=adamw ou normuon)

Une session Kaggle dure 12 h au plus : l'entraînement se fait en plusieurs sessions (SESSION=1, 2, 3…). Chaque
session reprend l'état laissé par la précédente (poids, optimiseur, temps déjà passé), pré-entraîne
HOURS_PRETRAIN heures de plus, puis affine une copie sur les Simpson : chaque session livre un modèle testable,
dialogues_bpe8192_gpt2_s{SESSION}.safetensors. Le learning rate suit un cosinus sur TOTAL_PRETRAIN_HOURS.

Même modèle (gpt.py), même tokenizer, même nettoyage et même découpage train/val/test des Simpson que
transformer_from_scratch.ipynb ; seuls changent la taille, le GPU (NVIDIA, fp16) et la durée.

ARCH=modern : la recette modded-nanoGPT (gpt.ModernGPT, Muon + Adam, planning warmup-stable-decay) au lieu du
GPT du cours avec AdamW. MIX=subs:0.5,web:0.3,wiki:0.2 : la part de chaque corpus dans les lots (par défaut,
proportionnelle à sa taille) ; subs = les sous-titres de prepare_dialogue.py. Le temps est
compté en heures, pas en pas : le learning rate suit une montée puis un cosinus sur le temps imparti, pour que
tout tienne dans une session Kaggle (12 h au plus). SMOKE=1 : un essai miniature (CPU ou Mac, quelques minutes).
"""

import glob
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

INPUT = Path(glob.glob("/kaggle/input/**/gpt.py", recursive=True)[0]).parent if glob.glob(
    "/kaggle/input/**/gpt.py", recursive=True) else Path(__file__).resolve().parent
sys.path.insert(0, str(INPUT))
import gpt  # noqa: E402

OUT = Path(os.environ.get("OUT", "/kaggle/working"))
WORK = Path(os.environ.get("WORK", "/tmp/super-modele"))
SMOKE = os.environ.get("SMOKE") == "1"
SESSION = int(os.environ.get("SESSION", 1))
HOURS_PRETRAIN = float(os.environ.get("HOURS_PRETRAIN", 0.03 if SMOKE else 9.5))  # pour cette session
TOTAL_PRETRAIN_HOURS = float(os.environ.get("TOTAL_PRETRAIN_HOURS", 0.06 if SMOKE else 20))  # toutes sessions
HOURS_FINETUNE = float(os.environ.get("HOURS_FINETUNE", 0.02 if SMOKE else 0.75))
SHARDS = int(os.environ.get("SHARDS", 1 if SMOKE else 17))
ARCH = os.environ.get("ARCH", "course")  # course : le GPT du cours + AdamW ; modern : modded-nanoGPT + Muon
MIX = dict((k, float(v)) for k, v in (item.split(":") for item in os.environ["MIX"].split(","))) if os.environ.get("MIX") else None
LR_SCALE = float(os.environ.get("LR_SCALE", 1.0))  # multiplie les learning rates de la recette modded
SEED = 1337

# Taille GPT-2 « small » : 12 blocs, 768 dimensions, 12 têtes. Fenêtre de 512 tokens (GPT-2 : 1 024),
# largement assez pour une scène de dialogue, et deux fois moins chère en attention.
CONFIG = dict(vocab_size=8192, block_size=512, n_embd=768, n_head=12, n_layer=12, dropout=0.0)
MICRO_BATCH = int(os.environ.get("MICRO_BATCH", 16))  # par GPU ; 16 × 512 tient dans les 16 Go d'une T4
TOKENS_PER_STEP = 262_144  # ~0,26 M tokens par pas, accumulés sur plusieurs micro-lots
if SMOKE:
    CONFIG, MICRO_BATCH, TOKENS_PER_STEP = CONFIG | dict(block_size=64, n_embd=64, n_head=4, n_layer=2), 4, 1024
PRETRAIN = dict(lr=6e-4, min_lr=6e-5, warmup=0.02, weight_decay=0.1, dropout=0.0)
# modded-nanoGPT : Muon pour les matrices des blocs, Adam pour le reste ; plateau puis descente sur 40 % du temps.
PRETRAIN_MODERN = dict(muon=True, muon_lr=0.05, embed_lr=0.6, head_lr=0.008, scalar_lr=0.04, warmup=0.01,
                       cooldown=0.4, dropout=0.0, normuon=os.environ.get("OPTIM", "muon") == "normuon")  # OPTIM=normuon : NorMuon
FINETUNE = dict(lr=1.5e-4, min_lr=1.5e-5, warmup=0.03, weight_decay=0.1, dropout=0.1, replay=0.25)
# Affinage avec NorMuon, comme le pré-entraînement : ses learning rates × FT_SCALE, montée puis descente jusqu'à 0.
FINETUNE_MODERN = dict(muon=True, normuon=True, warmup=0.05, cooldown=0.95, dropout=0.1, replay=0.25,
                       **{k: PRETRAIN_MODERN[k] * float(os.environ.get("FT_SCALE", 0.1))
                          for k in ("muon_lr", "embed_lr", "head_lr", "scalar_lr")})
INIT = os.environ.get("INIT")  # affinage seul : les poids d'un modèle déjà pré-entraîné (.safetensors de gpt.py)
FT_OPTIM = os.environ.get("FT_OPTIM", "adamw")  # l'optimiseur de l'affinage : adamw ou normuon
FT_TOKENS = float(os.environ.get("FT_TOKENS", 0))  # affinage : un budget en tokens au lieu de HOURS_FINETUNE
FT_TAG = os.environ.get("FT_TAG", "gpu")  # dans le nom du modèle affiné seul : la machine (a100…)
EVAL_EVERY_MIN = float(os.environ.get("EVAL_EVERY_MIN", 0.3 if SMOKE else 15))  # une validation (et un checkpoint)

END_SECTIONS = re.compile(r"^(Notes et références|Références|Notes|Voir aussi|Liens externes|Bibliographie|"
                          r"Articles connexes|Annexes|Sources|Filmographie|Discographie)\s*$", re.MULTILINE)
SPACE_TABLE = str.maketrans({" ": " ", " ": " ", " ": " "})


def clean_article(text: str) -> str:
    """Le nettoyage du notebook : sans sections de fin, sans lignes vides en trop, sans espaces insécables."""
    end = END_SECTIONS.search(text)
    text = text[: end.start()] if end else text
    return re.sub(r"\n{3,}", "\n\n", text).strip().translate(SPACE_TABLE)


def log(*args) -> None:
    if RANK == 0:
        print(time.strftime("%H:%M:%S"), *args, flush=True)


# --- 1. Préparation : Wikipédia FR entier → un fichier de tokens uint16 ------------------------------------


def find_input(name: str) -> Path | None:
    """Un fichier laissé par la session précédente (ajoutée en entrée du notebook), s'il y en a un."""
    found = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True)) + (
        sorted(glob.glob(f"{os.environ['RESUME_DIR']}/**/{name}", recursive=True)) if os.environ.get("RESUME_DIR") else [])
    return Path(found[-1]) if found else None  # la session la plus récente (s1 < s2 < s3)


def prepare() -> None:
    """Télécharge les 17 fichiers de Wikipédia FR, nettoie, encode ; 1 article sur 200 va à la validation.

    Les tokens sont écrits dans OUT/data : ils font partie de la sortie de la session, et la suivante les
    reprend au lieu de tout refaire.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    if find_input("wiki_train.bin"):
        log(f"tokens de Wikipédia repris de la session précédente : {find_input('wiki_train.bin').parent}")
        return
    WORK.mkdir(parents=True, exist_ok=True)
    tokenizer = gpt.tokenizer_from_state(json.loads((INPUT / "tokenizer.json").read_text(encoding="utf-8")))
    sep = tokenizer.encode("\n\n")
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    out = {"train": open(OUT / "data" / "wiki_train.bin", "wb"), "val": open(OUT / "data" / "wiki_val.bin", "wb")}
    counts = {"train": 0, "val": 0}
    n_chars, n_docs, k = 0, 0, 0
    t0 = time.time()
    for shard in range(SHARDS):
        path = hf_hub_download("wikimedia/wikipedia", f"20231101.fr/train-{shard:05d}-of-00017.parquet",
                               repo_type="dataset", local_dir=str(WORK / "hf"))
        texts = [clean_article(t) for t in pq.read_table(path, columns=["text"]).column("text").to_pylist()]
        if SMOKE:
            texts = texts[:3_000]
        texts = [t for t in texts if len(t) >= 500]  # les ébauches, comme dans le notebook
        for i in range(0, len(texts), 2_000):
            batch = texts[i : i + 2_000]
            for text, ids in zip(batch, tokenizer.encode_batch(batch)):
                part = "val" if k % 200 == 0 else "train"
                k += 1
                arr = np.asarray(ids + sep, dtype=np.uint16)
                arr.tofile(out[part])
                counts[part] += len(arr)
            n_chars += sum(map(len, batch))
        n_docs += len(texts)
        os.remove(path)  # la place disque de Kaggle est comptée
        log(f"fichier {shard + 1}/{SHARDS}   {n_docs:,} articles   {n_chars / 1e9:.2f} G caractères   "
            f"{counts['train'] / 1e6:,.0f} M tokens   ({time.time() - t0:.0f} s)")
    for f in out.values():
        f.close()
    (OUT / "data" / "wiki_stats.json").write_text(json.dumps({"chars": n_chars, "docs": n_docs, **counts}))
    log(f"prêt : {counts['train']:,} tokens d'entraînement, {counts['val']:,} de validation, "
        f"{n_chars / (counts['train'] + counts['val']):.2f} caractères par token")


# --- 2. Entraînement ------------------------------------------------------------------------------------


def batch_from(data: np.ndarray, n: int, block: int, generator: np.random.Generator) -> tuple[torch.Tensor, ...]:
    """`n` morceaux de `block` tokens tirés au hasard, avec leurs cibles décalées d'un cran."""
    starts = generator.integers(0, len(data) - block - 1, size=n)
    chunk = torch.from_numpy(np.stack([data[s : s + block + 1] for s in starts]).astype(np.int64))
    return chunk[:, :-1].contiguous().to(DEVICE, non_blocking=True), chunk[:, 1:].contiguous().to(DEVICE, non_blocking=True)


def amp_dtype() -> torch.dtype:
    """bf16 sur CPU et sur les GPU récents (A100, H100…) ; fp16 sur les T4, qui n'ont pas de bf16 matériel
    (PyTorch en accepte un émulé, bien trop lent)."""
    if DEVICE_TYPE == "cuda" and torch.cuda.get_device_capability()[0] < 8:
        return torch.float16
    return torch.bfloat16


def autocast():
    return torch.autocast(DEVICE_TYPE, dtype=amp_dtype())


def agree(*values: float) -> list[float]:
    """Les valeurs du GPU 0, envoyées à tous : les processus prennent les mêmes décisions (mesurer, s'arrêter)."""
    t = torch.tensor(values, dtype=torch.float64, device=DEVICE)
    if WORLD > 1:
        dist.broadcast(t, 0)
    return t.tolist()


@torch.no_grad()
def estimate_loss(model: nn.Module, data: np.ndarray, n_batches: int = 40) -> float:
    """Loss moyenne sur des morceaux de validation tirés avec une graine fixe (comparable d'une mesure à l'autre)."""
    model.eval()
    generator = np.random.default_rng(SEED + RANK)
    total = torch.zeros((), device=DEVICE)
    for _ in range(n_batches):
        x, y = batch_from(data, MICRO_BATCH, CONFIG["block_size"], generator)
        with autocast():
            total += model(x, y)[1].float()
    if WORLD > 1:
        dist.all_reduce(total)
    model.train()
    return total.item() / (n_batches * WORLD)


@torch.no_grad()
def full_loss(model: nn.Module, data: np.ndarray) -> float:
    """Loss sur tout un texte, en morceaux de block_size qui se suivent (comme `full_loss` du notebook)."""
    model.eval()
    block = CONFIG["block_size"]
    n = (len(data) - 1) // block
    x = torch.from_numpy(data[: n * block].astype(np.int64)).view(n, block)
    y = torch.from_numpy(data[1 : n * block + 1].astype(np.int64)).view(n, block)
    total = 0.0
    for i in range(0, n, MICRO_BATCH):
        with autocast():
            total += model(x[i : i + MICRO_BATCH].to(DEVICE), y[i : i + MICRO_BATCH].to(DEVICE))[1].item() * len(x[i : i + MICRO_BATCH])
    return total / n


def tokens_per_s(elapsed: float) -> float:
    return TOKENS_THIS_SESSION[0] / max(elapsed, 1e-9)


TOKENS_THIS_SESSION = [0]


def train_stage(name: str, model: nn.Module, raw: nn.Module, train_parts: list[tuple[np.ndarray, float]],
                val_data: np.ndarray, cfg: dict, hours: float, tokenizer, writer, total_hours: float | None = None,
                state_in: Path | None = None, state_out: Path | None = None, max_tokens: float | None = None) -> float:
    """Entraîne pendant `hours` heures ; garde le meilleur modèle (validation) dans OUT/checkpoints/{name}.

    `max_tokens` : un budget en tokens à la place des heures (le planning du learning rate suit alors les tokens
    vus), pour comparer des runs sur des machines de vitesses différentes.

    `total_hours` : le learning rate suit son cosinus sur cette durée, dont les sessions précédentes ont déjà
    fait une partie (lue dans `state_in`) ; l'état de la fin est écrit dans `state_out`, pour la suivante.
    """
    for module in raw.modules():
        if isinstance(module, nn.Dropout):
            module.p = cfg["dropout"]
    if cfg.get("muon"):  # modded-nanoGPT
        optimizers = gpt.modern_optimizers(raw, **{k: cfg[k] * LR_SCALE for k in ("muon_lr", "embed_lr", "head_lr", "scalar_lr")}, normuon=cfg["normuon"])
        clip = None
    else:  # AdamW, weight decay sur les matrices seulement
        params = [p for p in raw.parameters() if p.requires_grad]
        optimizers = [gpt.AdamW([{"params": [p for p in params if p.dim() >= 2], "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]},
                                 {"params": [p for p in params if p.dim() < 2], "lr": cfg["lr"], "weight_decay": 0.0}])]
        clip = 1.0
    # fp16 sur T4 : la loss est multipliée par `scale` avant le backward (sinon les petits gradients tombent à 0),
    # les gradients divisés d'autant ; si un gradient déborde, on saute le pas et on divise `scale` par 2.
    # En bf16 (CPU, A100…), pas besoin : scale reste à 1.
    fp16 = amp_dtype() == torch.float16
    scale = 2.0**16 if fp16 else 1.0
    skipped = 0  # pas sautés d'affilée (gradients non finis)
    accum = max(1, TOKENS_PER_STEP // (MICRO_BATCH * CONFIG["block_size"] * WORLD))
    generator = np.random.default_rng(SEED * 100 + RANK)
    weights = np.array([w for _, w in train_parts]) / sum(w for _, w in train_parts)
    budget = hours * 3600
    total = (total_hours or hours) * 3600
    checkpoint = OUT / "checkpoints" / f"{name}.safetensors"
    best, step, start, last_eval = float("inf"), 0, time.time(), time.time()
    tokens_seen, train_losses, done_before = 0, [], 0.0  # done_before : secondes faites par les sessions d'avant
    TOKENS_THIS_SESSION[0] = 0
    if state_in is not None:
        state = torch.load(state_in, map_location="cpu", weights_only=False)
        raw.load_state_dict(state["model"])
        for o, o_state in zip(optimizers, state["optimizer"]):
            o.load_state_dict(o_state)
        scale = state["scale"]
        best, step, tokens_seen, done_before = state["best"], state["step"], state["tokens"], state["seconds"]
        generator = np.random.default_rng(SEED * 100 + RANK + 1000 * SESSION)  # d'autres morceaux de texte
        log(f"reprise de {state_in} : pas {step:,}, {tokens_seen / 1e6:,.0f} M tokens, {done_before / 3600:.2f} h faites")

    def save_state() -> None:
        if state_out is not None and RANK == 0:
            torch.save({"model": raw.state_dict(), "optimizer": [o.state_dict() for o in optimizers], "scale": scale,
                        "best": best, "step": step, "tokens": tokens_seen,
                        "seconds": done_before + time.time() - start}, state_out)
    log(f"=== {name} : {f'{max_tokens / 1e6:,.0f} M tokens' if max_tokens else f'{hours:g} h'}, {accum} micro-lots "
        f"× {MICRO_BATCH} × {CONFIG['block_size']} × {WORLD} GPU = {accum * MICRO_BATCH * CONFIG['block_size'] * WORLD:,} tokens par pas")
    evaluated = -1  # le dernier pas mesuré

    def measure() -> None:
        """Validation ; le modèle est sauvegardé s'il est le meilleur jusqu'ici."""
        nonlocal best, last_eval, train_losses, evaluated
        last_eval, evaluated = time.time(), step
        val = estimate_loss(raw, val_data)
        elapsed = time.time() - start
        note = ""
        if val < best:
            best, note = val, "  ← meilleur, sauvegardé"
            if RANK == 0:
                gpt.save_checkpoint(checkpoint, raw, CONFIG | {"dropout": cfg["dropout"]} | ({"arch": "modern"} if ARCH == "modern" else {}), tokenizer, step=step,
                                    val_loss=val, tokens=tokens_seen, hyperparameters={**cfg, **CONFIG})
        log(f"{name}  pas {step:>6,}   train {sum(train_losses) / len(train_losses):.4f}   val {val:.4f}   "
            f"lr ×{lr:.3f}   {tokens_seen / 1e6:,.0f} M tokens   {len(train_losses) and tokens_per_s(elapsed):,.0f} tokens/s   "
            f"session {elapsed / 3600:.2f} h / {hours:g} h{note}")
        save_state()
        train_losses = []
        if RANK == 0:
            writer.add_scalar(f"{name}/loss val", val, step)
            writer.flush()

    while True:
        if max_tokens:  # le budget en tokens décide de la fin et du planning
            progress = session_progress = TOKENS_THIS_SESSION[0] / max_tokens
            do_eval = agree(time.time() - last_eval >= EVAL_EVERY_MIN * 60)[0]
        else:
            session_progress, progress, do_eval = agree((time.time() - start) / budget,
                                                        (done_before + time.time() - start) / total,
                                                        time.time() - last_eval >= EVAL_EVERY_MIN * 60)
        if session_progress >= 1 or progress >= 1:
            break
        warm = cfg["warmup"]
        if cfg.get("muon"):
            lr = gpt.wsd_schedule(progress, warm, cfg["cooldown"])  # un multiplicateur des learning rates
        else:  # montée puis cosinus, en multiplicateur de cfg["lr"]
            lr = min(1.0, (progress + 1e-4) / warm) if progress < warm else (
                cfg["min_lr"] + 0.5 * (1 + math.cos(math.pi * (progress - warm) / (1 - warm))) * (cfg["lr"] - cfg["min_lr"])) / cfg["lr"]
        loss_sum = 0.0
        for micro in range(accum):
            data = train_parts[generator.choice(len(train_parts), p=weights)][0]
            x, y = batch_from(data, MICRO_BATCH, CONFIG["block_size"], generator)
            if WORLD > 1:
                model.require_backward_grad_sync = micro == accum - 1  # synchronise les gradients au dernier
            with autocast():
                loss = model(x, y)[1] / accum
            (loss * scale).backward()
            loss_sum += loss.item()
        grads = [p.grad for p in raw.parameters() if p.grad is not None]
        torch._foreach_mul_(grads, 1.0 / scale)
        norm = nn.utils.clip_grad_norm_(raw.parameters(), clip or float("inf"))
        if not torch.isfinite(norm):  # débordement fp16 (identique sur les 2 GPU : gradients moyennés)
            for o in optimizers:
                o.zero_grad()
            skipped += 1
            if skipped >= 25:  # plus un débordement passager : le calcul lui-même ne tient plus en fp16
                raise RuntimeError(f"gradients non finis depuis {skipped} pas (pas {step}) : entraînement arrêté")
            scale = max(scale / 2, 1.0)
            continue
        skipped = 0
        for o in optimizers:
            o.step(lr)
        for o in optimizers:
            o.zero_grad()
        step += 1
        if fp16 and step % 500 == 0 and scale < 2.0**24:
            scale *= 2
        tokens_seen += accum * MICRO_BATCH * CONFIG["block_size"] * WORLD
        TOKENS_THIS_SESSION[0] += accum * MICRO_BATCH * CONFIG["block_size"] * WORLD
        train_losses.append(loss_sum)
        if step % 20 == 0 and RANK == 0:
            elapsed = time.time() - start
            writer.add_scalar(f"{name}/loss train", sum(train_losses[-20:]) / 20, step)
            writer.add_scalar(f"{name}/learning rate", lr, step)
            writer.add_scalar(f"{name}/tokens par seconde", tokens_per_s(elapsed), step)
        if do_eval or step == 10 + (state_in is not None):
            measure()
    if step > evaluated and train_losses:  # la fin du planning, learning rate au plus bas : souvent le meilleur moment
        measure()
    save_state()
    return best


def train() -> None:
    from torch.utils.tensorboard import SummaryWriter

    torch.manual_seed(SEED)
    tokenizer = gpt.tokenizer_from_state(json.loads((INPUT / "tokenizer.json").read_text(encoding="utf-8")))
    gpt.FAST_ATTENTION = True
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(OUT / "runs" / "gpt2_kaggle")) if RANK == 0 else None

    # Le corpus général : Wikipédia FR + web français (prepare_data.py), sous-titres (prepare_dialogue.py).
    general, vals = [], []
    for name in ("wiki", "web", "subs"):
        path = find_input(f"{name}_train.bin") or (OUT / "data" / f"{name}_train.bin")
        if path.exists() and (MIX is None or MIX.get(name, 0) > 0):
            data = np.memmap(path, dtype=np.uint16, mode="r")
            general.append((data, MIX[name] if MIX else float(len(data))))
            vals.append(np.asarray(np.memmap(path.with_name(f"{name}_val.bin"), dtype=np.uint16, mode="r")))
            log(f"{name} : {len(data):,} tokens, poids {general[-1][1]:g}")
    wiki_train = general[0][0]  # pour l'affichage
    wiki_val = np.concatenate(vals)
    simpsons = {part: (INPUT / f"simpsons_{part}.txt").read_text(encoding="utf-8") for part in ("train", "val", "test")}
    tokens = {part: np.asarray(tokenizer.encode(text), dtype=np.uint16) for part, text in simpsons.items()}
    log("corpus général : " + " + ".join(f"{len(d):,}" for d, _ in general) + " tokens   Simpson : " + "   ".join(f"{p} {len(t):,}" for p, t in tokens.items()))

    raw = (gpt.ModernGPT if ARCH == "modern" else gpt.GPTLanguageModel)(**CONFIG).to(DEVICE)
    log(f"{sum(p.numel() for p in raw.parameters()):,} paramètres")
    def build(compiled: bool) -> nn.Module:
        """Le modèle du pas d'entraînement : compilé ou non, réparti sur les GPU (DDP) s'il y en a plusieurs."""
        m = torch.compile(raw) if compiled else raw
        return nn.parallel.DistributedDataParallel(m, device_ids=[LOCAL_RANK]) if WORLD > 1 else m

    compile_wanted = os.environ.get("COMPILE", "1") == "1"
    model = build(compile_wanted)
    if compile_wanted:  # un pas d'essai : si la compilation échoue (GPU ancien, opération non gérée…), sans elle
        ok = torch.ones((), device=DEVICE)
        try:
            x, y = batch_from(general[0][0], MICRO_BATCH, CONFIG["block_size"], np.random.default_rng(0))
            with autocast():
                model(x, y)[1].backward()
        except Exception as error:
            log(f"torch.compile a échoué ({type(error).__name__}: {str(error)[:300]}) : on continue sans")
            ok.zero_()
        if WORLD > 1:  # tous les GPU prennent la même décision
            dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        if ok.item() == 0:
            model = build(False)
        raw.zero_grad(set_to_none=True)

    name = f"dialogues_bpe8192_gpt2{'_modded' if ARCH == 'modern' else ''}_s{SESSION}"
    if INIT:  # affinage seul, à partir d'un modèle déjà pré-entraîné (par exemple celui d'une session précédente)
        raw.load_state_dict(gpt.load_checkpoint(Path(INIT), "cpu")[0].state_dict())
        log(f"poids pré-entraînés : {INIT}")
        name += f"_ft_{FT_TAG}_{FT_OPTIM}"
    else:
        # Étape 1 : le français, sur Wikipédia (dropout 0 : chaque texte n'est vu qu'environ une fois).
        # Reprise : l'état laissé par la session précédente (son pré-entraînement, pas son affinage).
        train_stage("pretrain_gpt2" + ("_modded" if ARCH == "modern" else ""), model, raw, general, wiki_val,
                    PRETRAIN_MODERN if ARCH == "modern" else PRETRAIN, HOURS_PRETRAIN,
                    tokenizer, writer, total_hours=TOTAL_PRETRAIN_HOURS, state_in=find_input("wiki_state.pt"),
                    state_out=OUT / "wiki_state.pt")
    # Étape 2 : on affine le modèle de la fin du pré-entraînement ; les Simpson, avec 25 % de Wikipédia (replay).
    if WORLD > 1:
        dist.barrier()
    finetune = FINETUNE_MODERN if FT_OPTIM == "normuon" else FINETUNE
    replay = finetune["replay"]
    total = sum(w for _, w in general)
    train_stage(name, model, raw, [(tokens["train"], 1 - replay)] + [(d, replay * w / total) for d, w in general],
                tokens["val"], finetune, HOURS_FINETUNE, tokenizer, writer, max_tokens=FT_TOKENS or None)

    if RANK == 0:  # le score, comme à la fin du notebook : bits par caractère sur la validation et le test
        final, _, info = gpt.load_checkpoint(OUT / "checkpoints" / f"{name}.safetensors", DEVICE)
        result = {"session": SESSION, "arch": ARCH, "mix": MIX, "step": info["step"], "val_loss_estimate": info["val_loss"]}
        for part in ("val", "test"):
            loss = full_loss(final, tokens[part])
            result[f"{part}_loss"] = loss
            result[f"{part}_bits_per_char"] = loss / math.log(2) / (len(simpsons[part]) / len(tokens[part]))
        log("résultat : " + json.dumps(result))
        (OUT / "resultat.json").write_text(json.dumps(result, indent=2))
        meta = json.loads((OUT / "checkpoints" / f"{name}.json").read_text()) if (OUT / "checkpoints" / f"{name}.json").exists() else {}
        (OUT / "checkpoints" / f"{name}.json").write_text(json.dumps(meta | result, indent=2))
        prompts = [("Bart", "Homer", "Papa, je peux avoir un poney ?"), ("Marge", "Homer", "Homer, tu as encore bu chez Moe ?"),
                   ("Homer", "Moe", "Moe, une bière s'il te plaît."), ("Homer", "Burns", "Monsieur Burns, je voudrais une augmentation.")]
        lines = []
        for me, who, message in prompts:
            text = gpt.reply(final, tokenizer, f"{me}: {message}\n{who}:", who, DEVICE)
            lines.append(f"{me}: {message}\n{who}: {text}\n")
        log("exemples :\n" + "\n".join(lines))
        (OUT / "exemples.txt").write_text("\n".join(lines), encoding="utf-8")
        writer.close()


if __name__ == "__main__":
    RANK, LOCAL_RANK, WORLD = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if sys.argv[1] == "prepare":
        if RANK == 0:
            prepare()
        sys.exit(0)
    if WORLD > 1:
        dist.init_process_group("nccl")
    DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = f"cuda:{LOCAL_RANK}" if DEVICE_TYPE == "cuda" else "cpu"
    if DEVICE_TYPE == "cuda":
        torch.cuda.set_device(LOCAL_RANK)
    try:
        train()
    finally:
        if WORLD > 1:
            dist.destroy_process_group()
