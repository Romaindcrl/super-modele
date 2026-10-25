"""Construit kaggle/kernels/notebook/super_modele.ipynb, le notebook Kaggle propre et commenté du projet.

    python kaggle/build_notebook.py

Le notebook est autonome (tout le code est dans ses cellules) ; ses checkpoints se rechargent avec
`super_modele.gpt.load_checkpoint`, donc dans l'interface locale (`uv run app.py`).
"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "kernels" / "notebook"
cells = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "id": f"m{len(cells)}", "metadata": {}, "source": text.strip()})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "id": f"c{len(cells)}", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip()})


# ======================================================================================================
md(r"""
# Super modèle · un GPT français entraîné de zéro, qui parle comme les Simpson

Ce notebook entraîne **de zéro** un modèle de langage de la famille GPT, puis le fait parler comme les personnages des Simpson. Tout le code est ici, sans bibliothèque de modèles toute faite : tokenizer, Transformer, optimiseurs, boucle d'entraînement, évaluation, génération.

**Le parcours**

1. **Pré-entraînement** sur du français général : des sous-titres de films et de séries (OpenSubtitles, pour le registre oral), du web (FineWeb-2) et Wikipédia. Le modèle y apprend la langue.
2. **Affinage** sur les scripts des Simpson (`Personnage: réplique`) : il y apprend le format et le ton des personnages.
3. **Discussion** : on choisit qui on est et à qui on parle ; le modèle écrit plusieurs répliques et garde la meilleure.

**La recette** reprend le speedrun *modded-nanoGPT* (Keller Jordan et al., 2024-2025) : un Transformer modernisé (RoPE, QK-norm, ReLU², connexions en U…) entraîné avec **NorMuon** (Li et al., 2025), une variante de l'optimiseur Muon. À nombre de tokens égal, elle fait passer la loss de validation de 2,93 (le GPT du cours avec AdamW) à 2,70 (section 5).

**Deux modes** (cellule suivante) : `"demo"` entraîne un petit modèle (7 M de paramètres) pour tout vérifier : quelques minutes sur GPU, environ 1 h 10 sur le CPU de Kaggle ; `"complet"` entraîne un modèle de la taille de GPT-2 small (≈ 100 M de paramètres), à lancer sur GPU (~9 h sur les 2 × T4 de Kaggle).

**Les entrées du notebook** : le dataset `super-modele-donnees` (tokenizer, scripts des Simpson) et les sorties des notebooks de préparation `super-modele-prep` (Wikipédia, web) et `super-modele-dialogue` (sous-titres). Sans les corpus, le mode démo pré-entraîne sur les Simpson eux-mêmes.
""")

md("## 0. Réglages\n\nLe seul endroit à modifier.")
code(r'''
MODE = "demo"  # "demo" : un petit modèle, ~1 h 10 sur CPU (quelques minutes sur GPU) ; "complet" : GPT-2 small, ~9 h sur 2 × T4

# Taille du modèle. GPT-2 small : 12 blocs, 768 dimensions, 12 têtes (124 M de paramètres avec son vocabulaire
# de 50 257 tokens, 98 M avec le nôtre de 8 192). block_size : la fenêtre de contexte, en tokens.
MODEL = {
    "demo":    dict(n_layer=4,  n_head=4,  n_embd=256, block_size=256),
    "complet": dict(n_layer=12, n_head=12, n_embd=768, block_size=512),
}[MODE]

# Volume d'entraînement, en tokens vus. Un pas traite `batch` séquences de block_size tokens, par micro-lots de
# `micro_batch` (ce qui tient en mémoire) dont les gradients s'additionnent.
RUN = {
    "demo":    dict(pretrain_tokens=5e6, finetune_tokens=1e6,  batch=32,  micro_batch=16, eval_every=50),
    "complet": dict(pretrain_tokens=1e9, finetune_tokens=90e6, batch=512, micro_batch=16, eval_every=250),
}[MODE]

# Part de chaque corpus dans les lots du pré-entraînement : moitié de dialogue (le registre des Simpson).
MIX = {"subs": 0.5, "web": 0.3, "wiki": 0.2}

# Optimiseur du pré-entraînement : "normuon" (le meilleur de nos essais), "muon", ou "adamw" pour comparer.
OPTIMIZER = "normuon"
LR = dict(muon=0.05, embed=0.6, head=0.008, scalar=0.04, adamw=6e-4)  # ceux du speedrun modded-nanoGPT

# Affinage sur les Simpson : AdamW, petit learning rate, un peu de dropout, et 25 % de corpus général dans les
# lots (« replay ») pour ne pas oublier le français général.
FINETUNE = dict(lr=1.5e-4, min_lr=1.5e-5, dropout=0.1, replay=0.25)

SEED = 1337
''')

# ======================================================================================================
md(r"""
## 1. L'environnement

Le matériel (GPU NVIDIA, puce Apple ou CPU), la précision des calculs, et l'emplacement des données.
""")
code(r'''
import glob
import json
import math
import os
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

torch.manual_seed(SEED)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
N_GPUS = torch.cuda.device_count() if DEVICE == "cuda" else 0

# Précision mixte : les produits matriciels en 16 bits (2 fois plus rapides), le reste en 32 bits. bf16 garde la
# plage du float32 ; les T4 de Kaggle ne font vite que du fp16, dont la plage est petite : on multiplie alors la
# loss avant le backward pour que les petits gradients ne s'arrondissent pas à 0 (« loss scaling », section 4).
if DEVICE == "cuda":
    AMP_DTYPE = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
elif DEVICE == "mps":
    AMP_DTYPE = torch.bfloat16
else:
    AMP_DTYPE = None  # sur CPU : tout en float32


def autocast():
    """Le contexte de précision mixte (sans effet sur CPU)."""
    if AMP_DTYPE is None:
        return torch.autocast("cpu", enabled=False)
    return torch.autocast(DEVICE, dtype=AMP_DTYPE)


# Les données : les entrées du notebook sur Kaggle, ou un dossier local (variable SUPER_MODELE_DATA).
DATA_ROOTS = ["/kaggle/input", os.environ.get("SUPER_MODELE_DATA", "data")]
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("sorties")
OUT.mkdir(parents=True, exist_ok=True)


def find(name: str) -> Path | None:
    """Le chemin d'un fichier des données (le plus récent s'il apparaît dans plusieurs entrées), ou None."""
    for root in DATA_ROOTS:
        hits = sorted(glob.glob(f"{root}/**/{name}", recursive=True))
        if hits:
            return Path(hits[-1])
    return None


print(f"matériel : {DEVICE}" + (f" × {N_GPUS} ({torch.cuda.get_device_name(0)})" if N_GPUS else "")
      + f" · précision : {AMP_DTYPE or torch.float32} · PyTorch {torch.__version__}")
''')

# ======================================================================================================
md(r"""
## 2. Le tokenizer

Le modèle ne lit pas des lettres mais des **tokens** : des morceaux de texte fréquents, chacun numéroté. Le nôtre est un **BPE au niveau des octets**, comme celui de GPT-2 en plus petit (8 192 tokens) : on part des 256 octets possibles et on fusionne à répétition la paire de morceaux la plus fréquente (`" mais"`, `"ais"`, `" Homer"`…). Partir des octets permet d'écrire n'importe quel texte, accents et emojis compris.

Il a été appris une fois pour toutes, sur les dialogues des Simpson et ~100 M de caractères de Wikipédia ; les corpus préparés ont été encodés avec lui, il faut donc garder exactement le même. `train_bpe` montre comment il a été fait.
""")
code(r'''
def train_bpe(texts, vocab_size: int = 8192) -> Tokenizer:
    """Comment ce tokenizer a été appris (pour mémoire : on ne le refait pas ici).

    Le texte est d'abord pré-découpé comme dans GPT-2 : les fusions ne traversent pas les frontières entre mots,
    nombres, ponctuation et espaces.
    """
    from tokenizers import decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # octets + découpage de GPT-2
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    tok.train_from_iterator(texts, trainer)
    return tok


TOKENIZER_STATE = json.loads(find("tokenizer.json").read_text(encoding="utf-8"))  # gardé dans les checkpoints
tokenizer = Tokenizer.from_str(TOKENIZER_STATE["tokenizer"])
VOCAB_SIZE = tokenizer.get_vocab_size()


def encode(text: str) -> list[int]:
    return tokenizer.encode(text).ids


def decode(ids: list[int]) -> str:
    return tokenizer.decode(ids)


example = "Homer: D'oh ! Marge, où sont passés mes donuts ?"
ids = encode(example)
print(f"vocabulaire : {VOCAB_SIZE} tokens")
print(f"{len(example)} caractères → {len(ids)} tokens :", " | ".join(decode([i]) for i in ids))
''')

# ======================================================================================================
md(r"""
## 3. Les données

### Les Simpson

Les scripts de 473 épisodes, mis au format `Personnage: réplique` (une ligne vide entre deux scènes), et déjà découpés en entraînement, validation et test **par épisode** : le modèle n'est jamais évalué sur un épisode qu'il a lu.
""")
code(r'''
simpsons = {part: find(f"simpsons_{part}.txt").read_text(encoding="utf-8") for part in ("train", "val", "test")}
simpsons_tokens = {part: np.array(encode(text), dtype=np.uint16) for part, text in simpsons.items()}
for part, toks in simpsons_tokens.items():
    print(f"{part:5} : {len(simpsons[part]):>10,} caractères → {len(toks):>9,} tokens")
print("-" * 70)
print(simpsons["train"][:500])
''')

md(r"""
### Le corpus général

Des milliards de tokens de français, préparés à part (annexe A) et rangés en `uint16` : chaque fichier est une longue suite de tokens, lue directement sur le disque (`np.memmap`) sans la charger en mémoire.

| corpus | contenu | tokens |
|---|---|---|
| `subs` | sous-titres OpenSubtitles 2024 (dialogues de films et séries), épisodes des Simpson retirés | ~2,2 G |
| `web` | FineWeb-2, partie française | 3,5 G |
| `wiki` | Wikipédia FR entière, sans les sections de références | 1,9 G |
""")
code(r'''
general, general_val = {}, []
for name, weight in MIX.items():
    path = find(f"{name}_train.bin")
    if path is None:
        print(f"{name:5} : absent des entrées du notebook")
        continue
    general[name] = (np.memmap(path, dtype=np.uint16, mode="r"), weight)
    general_val.append(np.array(np.memmap(path.with_name(f"{name}_val.bin"), dtype=np.uint16, mode="r")))
    print(f"{name:5} : {len(general[name][0]) / 1e6:>8,.0f} M tokens · poids {weight}")

if not general:  # sans les corpus (démo) : on pré-entraîne sur les Simpson eux-mêmes
    print("pas de corpus général : le pré-entraînement se fera sur les Simpson (assez pour une démo)")
    general = {"simpson": (simpsons_tokens["train"], 1.0)}
    general_val = [simpsons_tokens["val"]]
general_val = np.concatenate(general_val)
''')

md(r"""
### Les lots

À chaque pas, on tire au hasard des fenêtres de `block_size` tokens ; la cible de chaque position est le token suivant (la même fenêtre décalée d'un cran). Le corpus de chaque micro-lot est tiré selon les poids de `MIX`.
""")
code(r'''
def get_batch(data: np.ndarray, n: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """n fenêtres de block_size tokens tirées au hasard dans data, et leurs cibles (décalées d'un cran)."""
    block = MODEL["block_size"]
    starts = rng.integers(0, len(data) - block - 1, size=n)
    chunk = torch.from_numpy(np.stack([data[s : s + block + 1] for s in starts]).astype(np.int64))
    return chunk[:, :-1].to(DEVICE), chunk[:, 1:].to(DEVICE)


def pick_corpus(sources: dict[str, tuple[np.ndarray, float]], rng: np.random.Generator) -> np.ndarray:
    """Un corpus tiré au hasard selon les poids : {nom: (tokens, poids)} → les tokens."""
    names = list(sources)
    weights = np.array([sources[n][1] for n in names], dtype=float)
    return sources[names[rng.choice(len(names), p=weights / weights.sum())]][0]


x, y = get_batch(simpsons_tokens["train"], 1, np.random.default_rng(0))
print("entrée :", repr(decode(x[0, :24].tolist())))
print("cible  :", repr(decode(y[0, :24].tolist())))
''')

# ======================================================================================================
md(r"""
## 4. Le modèle

C'est le Transformer du cours — des blocs *attention* (les tokens échangent de l'information) puis *MLP* (chacun la traite), avec des connexions résiduelles — modernisé comme dans modded-nanoGPT :

| pièce | GPT du cours | ici | pourquoi |
|---|---|---|---|
| position | embedding de position appris | **RoPE** : q et k tournent d'un angle proportionnel à la position | l'attention voit directement les distances |
| normalisation | LayerNorm (2 paramètres) | **RMSNorm** sans paramètre, et aussi sur q et k (**QK-norm**) | plus simple, stable avec de grands learning rates |
| activation du MLP | ReLU | **ReLU²** | meilleure à coût égal dans le speedrun |
| biais | oui | non | inutiles |
| départ | poids aléatoires partout | projections de sortie **à zéro** | chaque bloc démarre comme l'identité |
| raccourcis | résiduels | + **connexions en U** : la 1ʳᵉ moitié des blocs nourrit la 2ᵉ | l'information des premières couches arrive intacte |
| sortie | logits libres | **plafonnés** à ±30 (30·tanh(z/30)) | évite les explosions |
""")
code(r'''
def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """RMSNorm sans paramètre : chaque vecteur est divisé par sa moyenne quadratique (sa direction ne change pas).
    Calculée en float32 : en fp16, le carré d'une valeur au-delà de 256 dépasserait 65 504, le plus grand fp16."""
    return F.rms_norm(x.float(), (x.size(-1),)).type_as(x)


class Rotary(nn.Module):
    """RoPE (Su et al., 2021) : à la position t, chaque paire de coordonnées de q et de k tourne d'un angle t·θᵢ.
    Le produit q·k ne dépend alors que de l'écart entre les deux positions."""

    def __init__(self, head_dim: int, block_size: int, base: float = 10_000.0) -> None:
        super().__init__()
        theta = base ** (-torch.arange(0, head_dim // 2, dtype=torch.float32) / (head_dim // 2))  # les fréquences
        angles = torch.outer(torch.arange(block_size, dtype=torch.float32), theta)          # (T, head_dim / 2)
        self.register_buffer("cos", angles.cos(), persistent=False)  # recalculés à chaque création : rien à sauver
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x : (B, têtes, T, head_dim)
        T = x.size(-2)
        cos, sin = self.cos[:T].to(x.dtype), self.sin[:T].to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)  # une rotation 2D par paire


class Attention(nn.Module):
    """Self-attention causale à plusieurs têtes : softmax(q·kᵀ/√d)·v, chaque token ne regardant que le passé."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)  # q, k et v de toutes les têtes en un seul produit
        self.proj = nn.Linear(n_embd, n_embd, bias=False)      # recombine les têtes
        self.rotary = Rotary(n_embd // n_head, block_size)
        self.attn_dropout = nn.Dropout(dropout)  # son taux est lu par l'attention de PyTorch
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        q, k = self.rotary(rms_norm(q)), self.rotary(rms_norm(k))  # QK-norm, puis RoPE
        p = self.attn_dropout.p if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)  # le futur est masqué
        return self.dropout(self.proj(y.transpose(1, 2).reshape(B, T, C)))


class MLP(nn.Module):
    """Le calcul de chaque position : couche cachée 4 fois plus large, activation ReLU² (ReLU puis carré)."""

    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc(x))
        if h.dtype == torch.float16:  # fp16 (T4) : le carré d'une valeur au-delà de 256 dépasserait 65 504
            h = h.float().square()
            s = h.amax(dim=-1, keepdim=True).clamp(min=1.0)  # chaque token ramené sous 1, puis rétabli :
            return self.dropout(self.proj((h / s).half()).float() * s)  # même résultat, proj est linéaire
        return self.dropout(self.proj(h.square()))


class Block(nn.Module):
    """Un bloc : attention puis MLP, chacun sur une copie normalisée, ajoutée au flux principal (résiduel)."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.attn = Attention(n_embd, n_head, block_size, dropout)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(rms_norm(x))
        return x + self.mlp(rms_norm(x))


class GPT(nn.Module):
    """Le modèle complet : embedding des tokens, n_layer blocs (connexions en U), tête de sortie plafonnée.

    forward(idx, targets) renvoie (logits, loss) ; loss vaut None sans cibles. Mêmes noms de paramètres que
    super_modele.gpt.ModernGPT : les checkpoints se rechargent dans l'interface locale.
    """

    def __init__(self, vocab_size: int, block_size: int, n_embd: int, n_head: int, n_layer: int,
                 dropout: float = 0.0, softcap: float = 30.0) -> None:
        super().__init__()
        self.block_size, self.softcap = block_size, softcap
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList(Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer))
        self.skip_weights = nn.Parameter(torch.ones(n_layer // 2))  # le poids de chaque connexion en U
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        for name, param in self.named_parameters():
            if name.endswith("proj.weight") or name == "lm_head.weight":
                nn.init.zeros_(param)  # chaque bloc démarre comme l'identité, la sortie comme « tout équiprobable »
            elif param.dim() == 2 and not name.startswith("embed"):
                bound = 3**0.5 * 0.5 * param.size(1) ** -0.5  # uniforme, écart-type 0,5/√(entrées)
                nn.init.uniform_(param, -bound, bound)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = rms_norm(self.embed(idx))
        half = len(self.blocks) // 2
        skips = []
        for i, block in enumerate(self.blocks):
            if i >= len(self.blocks) - half:  # seconde moitié : + la sortie du bloc symétrique de la première
                x = x + self.skip_weights[i - (len(self.blocks) - half)] * skips.pop()
            x = block(x)
            if i < half:
                skips.append(x)
        logits = self.lm_head(rms_norm(x)).float()
        logits = self.softcap * torch.tanh(logits / self.softcap)  # plafonnés à ±softcap
        if targets is None:
            return logits, None
        B, T = idx.shape
        return logits, F.cross_entropy(logits.view(B * T, -1), targets.reshape(B * T))


model = GPT(VOCAB_SIZE, **MODEL).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
x, y = get_batch(simpsons_tokens["train"], 4, np.random.default_rng(0))
with torch.no_grad():
    loss0 = model(x, y)[1].item()
print(f"{n_params / 1e6:.1f} M paramètres · loss au départ {loss0:.4f} "
      f"(= ln {VOCAB_SIZE} = {math.log(VOCAB_SIZE):.4f} : tous les tokens équiprobables, tête à zéro)")
''')

# ======================================================================================================
md(r"""
## 5. Les optimiseurs

**AdamW** met à jour chaque poids séparément, selon la moyenne et la variance de ses gradients passés.

**Muon** (Keller Jordan, 2024) traite chaque matrice de poids comme un tout : il prend la mise à jour avec momentum et la remplace par la matrice orthogonale la plus proche (le facteur UVᵀ de sa décomposition en valeurs singulières). Toutes ses directions ont alors la même amplitude : les directions rares mais utiles ne sont plus écrasées par les dominantes. Le calcul se fait en 5 itérations de Newton-Schulz — de simples produits matriciels.

**NorMuon** (Li et al., 2025) ajoute à Muon une échelle par neurone : après l'orthogonalisation, toutes les lignes de la matrice n'ont pas la même amplitude ; on suit, pour chacune, la moyenne glissante de son carré et on la divise par sa racine (comme Adam, mais avec un seul nombre par neurone).

Muon ne s'applique qu'aux matrices des blocs ; l'embedding, la tête et les poids des connexions en U restent sous Adam. Tous les learning rates suivent un **planning** : une courte montée, un plateau, puis une descente linéaire jusqu'à 0 sur les derniers 40 % (*warmup-stable-decay*).

Sur le Mac, à 16 M tokens chacun (même modèle de 17 M de paramètres, mêmes lots), la loss de validation sur des sous-titres :

| run | loss |
|---|---|
| GPT du cours + AdamW | 2,933 |
| modèle ci-dessus + AdamW | 2,944 |
| modèle ci-dessus + Muon | 2,712 |
| modèle ci-dessus + **NorMuon** | **2,696** |
""")
code(r'''
@torch.no_grad()
def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """La matrice orthogonale « la plus proche » de G (UVᵀ si G = USVᵀ), approchée par 5 itérations de
    Newton-Schulz quintiques (coefficients de modded-nanoGPT)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transposed = X.size(-2) > X.size(-1)
    if transposed:  # l'itération travaille sur la dimension la plus petite
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)  # valeurs singulières ≤ 1
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X  # pousse chaque valeur singulière vers 1
    return X.mT if transposed else X


class Muon:
    """Muon : momentum de Nesterov, puis la mise à jour est orthogonalisée. normuon=True : NorMuon."""

    def __init__(self, params, lr: float = 0.05, momentum: float = 0.95, normuon: bool = False,
                 beta2: float = 0.95) -> None:
        self.params, self.lr, self.momentum = list(params), lr, momentum
        self.normuon, self.beta2 = normuon, beta2
        self.buffers = [torch.zeros_like(p) for p in self.params]                     # le momentum
        self.second = [torch.zeros(p.size(0), 1, device=p.device) for p in self.params]  # NorMuon : 1 par neurone

    @torch.no_grad()
    def step(self, mult: float = 1.0) -> None:
        for p, buf, v in zip(self.params, self.buffers, self.second):
            if p.grad is None:
                continue
            buf.lerp_(p.grad, 1 - self.momentum)
            update = newton_schulz(p.grad.lerp(buf, self.momentum))  # Nesterov, puis orthogonalisation
            if self.normuon:
                v.lerp_(update.square().mean(dim=-1, keepdim=True), 1 - self.beta2)  # amplitude de chaque neurone
                scaled = update / (v.sqrt() + 1e-8)
                update = scaled * (update.norm() / (scaled.norm() + 1e-8))  # garde la norme totale de Muon
            scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5  # matrices plus hautes que larges
            p.sub_(update.to(p.dtype) * (mult * self.lr * scale))

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


class AdamW:
    """AdamW (Loshchilov & Hutter, 2019), par groupes de paramètres {"params", "lr", "weight_decay", "betas"}."""

    def __init__(self, groups: list[dict], eps: float = 1e-8) -> None:
        self.groups, self.eps, self.t = groups, eps, 0
        self.m = [[torch.zeros_like(p) for p in g["params"]] for g in groups]  # moyenne des gradients
        self.v = [[torch.zeros_like(p) for p in g["params"]] for g in groups]  # moyenne de leurs carrés

    @torch.no_grad()
    def step(self, mult: float = 1.0) -> None:
        self.t += 1
        for group, ms, vs in zip(self.groups, self.m, self.v):
            b1, b2 = group.get("betas", (0.9, 0.95))
            lr = mult * group["lr"]
            for p, m, v in zip(group["params"], ms, vs):
                if p.grad is None:
                    continue
                p.mul_(1 - lr * group.get("weight_decay", 0.0))  # weight decay découplé
                m.lerp_(p.grad, 1 - b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                m_hat, v_hat = m / (1 - b1**self.t), v / (1 - b2**self.t)  # corrigées du départ à zéro
                p.sub_(lr * m_hat / (v_hat.sqrt() + self.eps))

    def zero_grad(self) -> None:
        for g in self.groups:
            for p in g["params"]:
                p.grad = None


def adamw_for(model: nn.Module, lr: float) -> AdamW:
    """AdamW pour tout le modèle : weight decay 0,1 sur les matrices seulement."""
    params = [p for p in model.parameters() if p.requires_grad]
    return AdamW([{"params": [p for p in params if p.dim() >= 2], "lr": lr, "weight_decay": 0.1},
                  {"params": [p for p in params if p.dim() < 2], "lr": lr}])


def modded_optimizers(model: GPT, normuon: bool) -> list:
    """La recette modded-nanoGPT : (Nor)Muon pour les matrices des blocs, Adam pour le reste."""
    matrices = [p for p in model.blocks.parameters() if p.dim() == 2]
    adam = AdamW([{"params": [model.embed.weight], "lr": LR["embed"], "betas": (0.8, 0.95)},
                  {"params": [model.lm_head.weight], "lr": LR["head"], "betas": (0.8, 0.95)},
                  {"params": [model.skip_weights], "lr": LR["scalar"], "betas": (0.8, 0.95)}])
    return [Muon(matrices, lr=LR["muon"], normuon=normuon), adam]


def wsd(progress: float, warmup: float = 0.01, cooldown: float = 0.4) -> float:
    """Planning warmup-stable-decay : multiplicateur du learning rate selon l'avancement (0 → 1)."""
    if progress < warmup:
        return (progress + 1e-3) / warmup
    return min(1.0, (1 - progress) / cooldown)


def cosine(progress: float, warmup: float = 0.05, floor: float = 0.1) -> float:
    """Montée puis cosinus jusqu'à `floor` (en fraction du learning rate) : le planning de l'affinage."""
    if progress < warmup:
        return (progress + 1e-3) / warmup
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * (progress - warmup) / (1 - warmup)))


G = torch.randn(256, 1024)
print("valeurs singulières après Newton-Schulz :",
      " ".join(f"{s:.2f}" for s in torch.linalg.svdvals(newton_schulz(G))[[0, 64, 128, 192, 255]].tolist()),
      "(toutes proches de 1 : la matrice est quasi orthogonale)")
''')

# ======================================================================================================
md(r"""
## 6. L'entraînement

La boucle du cours, avec quatre ajouts : l'**accumulation** de gradients (des lots plus grands que la mémoire), la **précision mixte**, le **loss scaling** en fp16, et, s'il y a 2 GPU, **DataParallel** (chacun traite la moitié de chaque micro-lot). On mesure la validation régulièrement et on garde le meilleur modèle.
""")
code(r'''
CONFIG = dict(vocab_size=VOCAB_SIZE, arch="modern", **MODEL)  # de quoi reconstruire le modèle


def save_checkpoint(path: Path, model: GPT, **info) -> None:
    """Les poids (safetensors), avec dans les métadonnées la configuration et le tokenizer : le format de
    super_modele.gpt, que l'interface locale sait recharger."""
    dropout = next(m.p for m in model.modules() if isinstance(m, nn.Dropout))
    tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(tensors, str(path), metadata={"config": json.dumps(CONFIG | {"dropout": dropout}),
                                            "tokenizer": json.dumps(TOKENIZER_STATE, ensure_ascii=False),
                                            "info": json.dumps(info, ensure_ascii=False)})


@torch.no_grad()
def evaluate(model: GPT, data: np.ndarray, n_batches: int = 20) -> float:
    """Loss moyenne sur des fenêtres tirées avec une graine fixe : comparable d'une mesure à l'autre."""
    model.eval()
    rng, total = np.random.default_rng(SEED), 0.0
    for _ in range(n_batches):
        x, y = get_batch(data, RUN["micro_batch"], rng)
        with autocast():
            total += model(x, y)[1].item()
    model.train()
    return total / n_batches


class LossOnly(nn.Module):
    """Le modèle qui ne renvoie que la loss : avec DataParallel, seules les losses (pas les énormes logits) sont
    rassemblées sur le premier GPU."""

    def __init__(self, model: GPT) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.model(x, y)[1]


def duration(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.0f} min"


def train(model: GPT, sources: dict, val_data: np.ndarray, tokens: float, optimizers: list, schedule,
          name: str, clip: float | None = None) -> dict:
    """Entraîne sur `tokens` tokens tirés de `sources` ({nom: (tokens, poids)}) ; garde le meilleur modèle
    (validation sur `val_data`) dans OUT/{name}.safetensors. Renvoie l'historique des mesures."""
    batch, micro, block = RUN["batch"], RUN["micro_batch"], MODEL["block_size"]
    accum, steps = batch // micro, int(tokens // (batch * block))
    step_model = nn.DataParallel(LossOnly(model)) if N_GPUS > 1 else LossOnly(model)
    rng = np.random.default_rng(SEED)
    scale = 2.0**16 if AMP_DTYPE == torch.float16 else 1.0  # loss scaling : seulement en fp16
    skipped = 0  # pas sautés d'affilée
    history = {"step": [], "tokens": [], "train": [], "val": []}
    best, recent, start = float("inf"), [], time.time()
    print(f"{name} : {steps:,} pas × {batch} séquences × {block} tokens = {steps * batch * block / 1e6:,.0f} M tokens")
    model.train()
    for step in range(steps + 1):
        if step % RUN["eval_every"] == 0 or step == steps:  # mesure (et sauvegarde si c'est la meilleure)
            val = evaluate(model, val_data)
            elapsed = time.time() - start
            note = ""
            if val < best:
                best, note = val, "  ← meilleur, sauvegardé"
                save_checkpoint(OUT / f"{name}.safetensors", model, step=step, val_loss=val)
            train_loss = sum(recent) / len(recent) if recent else float("nan")
            history["step"].append(step)
            history["tokens"].append(step * batch * block)
            history["train"].append(train_loss)
            history["val"].append(val)
            speed = step * batch * block / elapsed if step else 0
            eta = duration(elapsed / step * (steps - step)) if step else "?"
            shown = f"{train_loss:.4f}" if recent else "—"  # au pas 0, aucun lot n'a encore été vu
            print(f"pas {step:>6,}/{steps:,} · train {shown} · val {val:.4f} · "
                  f"{speed:,.0f} tokens/s · reste {eta}{note}")
            recent = []
        if step == steps:
            break
        loss_sum = 0.0
        for _ in range(accum):  # les gradients des micro-lots s'additionnent
            x, y = get_batch(pick_corpus(sources, rng), micro, rng)
            with autocast():
                loss = step_model(x, y).mean() / accum
            (loss * scale).backward()
            loss_sum += loss.item()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if scale != 1.0:
            torch._foreach_mul_(grads, 1.0 / scale)  # on retire le facteur du loss scaling
        norm = nn.utils.clip_grad_norm_(model.parameters(), clip or float("inf"))
        if not torch.isfinite(norm):  # débordement fp16 : on saute ce pas et on réduit l'échelle
            for o in optimizers:
                o.zero_grad()
            skipped += 1
            if skipped >= 25:  # plus un débordement passager : mieux vaut s'arrêter avec un message clair
                raise RuntimeError(f"gradients non finis depuis {skipped} pas : l'entraînement a divergé")
            scale = max(scale / 2, 1.0)
            continue
        skipped = 0
        for o in optimizers:
            o.step(schedule(step / steps))
            o.zero_grad()
        if scale != 1.0 and step % 500 == 499 and scale < 2.0**24:
            scale *= 2
        recent.append(loss_sum)
    print(f"{name} : terminé en {duration(time.time() - start)}, meilleure val {best:.4f}")
    return history
''')

md(r"""
### 6.1 Pré-entraînement : le français

Le modèle apprend à prédire le token suivant sur le corpus général (sans dropout : chaque texte n'est vu qu'à peu près une fois, il n'y a pas de quoi réciter).
""")
code(r'''
if OPTIMIZER == "adamw":
    optimizers, schedule, clip = [adamw_for(model, LR["adamw"])], cosine, 1.0
else:
    optimizers, schedule, clip = modded_optimizers(model, normuon=OPTIMIZER == "normuon"), wsd, None

pretrain_history = train(model, general, general_val, RUN["pretrain_tokens"], optimizers, schedule, "pretrain", clip)
''')

md(r"""
### 6.2 Affinage : les Simpson

On continue sur les scripts des Simpson, plus doucement : AdamW avec un petit learning rate, un peu de dropout (le corpus est petit, le modèle pourrait l'apprendre par cœur), et 25 % de corpus général dans les lots. On garde le meilleur modèle sur la validation.
""")
code(r'''
for module in model.modules():
    if isinstance(module, nn.Dropout):
        module.p = FINETUNE["dropout"]

replay, total_weight = FINETUNE["replay"], sum(w for _, w in general.values())
finetune_sources = {"simpson": (simpsons_tokens["train"], 1 - replay)}
finetune_sources |= {name: (data, replay * w / total_weight) for name, (data, w) in general.items() if name != "simpson"}

finetune_history = train(model, finetune_sources, simpsons_tokens["val"], RUN["finetune_tokens"],
                         [adamw_for(model, FINETUNE["lr"])],
                         lambda p: cosine(p, floor=FINETUNE["min_lr"] / FINETUNE["lr"]), "simpson", clip=1.0)

model.load_state_dict(load_file(str(OUT / "simpson.safetensors")))  # le meilleur moment de l'affinage
model.eval()
''')

code(r'''
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, history, title in [(axes[0], pretrain_history, "Pré-entraînement"), (axes[1], finetune_history, "Affinage (Simpson)")]:
    tokens = np.array(history["tokens"]) / 1e6
    ax.plot(tokens, history["val"], "o-", label="validation")
    ax.plot(tokens[1:], history["train"][1:], "--", label="entraînement (moyenne des pas)")
    ax.set_title(title)
    ax.set_xlabel("tokens vus (millions)")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.show()
''')

# ======================================================================================================
md(r"""
## 7. Évaluation

La loss sur **tout** le texte de validation et de test (des épisodes jamais vus), ramenée en **bits par caractère** : le nombre moyen de bits qu'il faudrait pour coder chaque caractère avec les prédictions du modèle. Indépendant du tokenizer, donc comparable d'un modèle à l'autre — plus bas = meilleur.

| modèle | test (bits/car.) |
|---|---|
| GPT du cours, 17 M, entraîné sur les Simpson seuls | 1,420 |
| idem, pré-entraîné sur Wikipédia | 1,401 |
| GPT du cours, taille GPT-2 small, 1,7 G tokens (Wikipédia + web), 2 × T4 | 1,257 |
""")
code(r'''
@torch.no_grad()
def full_loss(model: GPT, data: np.ndarray) -> float:
    """Loss moyenne sur tout un texte encodé, en fenêtres de block_size tokens qui se suivent."""
    block, n_win = MODEL["block_size"], (len(data) - 1) // MODEL["block_size"]
    x = torch.from_numpy(data[: n_win * block].astype(np.int64)).view(n_win, block)
    y = torch.from_numpy(data[1 : n_win * block + 1].astype(np.int64)).view(n_win, block)
    total = 0.0
    for i in range(0, n_win, RUN["micro_batch"]):
        xb, yb = x[i : i + RUN["micro_batch"]].to(DEVICE), y[i : i + RUN["micro_batch"]].to(DEVICE)
        with autocast():
            total += model(xb, yb)[1].item() * len(xb)
    return total / n_win


scores = {}
for part in ("val", "test"):
    loss = full_loss(model, simpsons_tokens[part])
    bits = loss / math.log(2) / (len(simpsons[part]) / len(simpsons_tokens[part]))
    scores[part] = {"loss": loss, "bits_per_char": bits}
    print(f"{part:4} : loss {loss:.4f} · perplexité {math.exp(loss):.1f} · {bits:.3f} bits par caractère")
''')

# ======================================================================================================
md(r"""
## 8. Discuter

La conversation est écrite comme un script : `Bart: Papa, je peux avoir un poney ?` puis `Homer:` — le modèle n'a plus qu'à écrire la suite, jusqu'au retour à la ligne. Pour de meilleures réponses :

- il en écrit **plusieurs en parallèle** (température 0,6, échantillonnage *top-p*, pénalité sur les mots qu'il vient d'écrire contre les bégaiements) ;
- il écarte les **redites** (une réplique déjà dite, ou l'écho du message) ;
- il garde celle qui maximise : *log p(réplique | conversation) − ½·log p(réplique | « Homer: » seul) + 0,3·ln(longueur) − part des mots déjà employés*. Une réplique passe-partout (« Quoi ? ») est probable partout et perd des points ; une réplique liée au message n'est probable qu'après lui (Li et al., 2016).
""")
code(r'''
NEWLINE = torch.tensor([i for i in range(VOCAB_SIZE) if "\n" in decode([i])], device=DEVICE)  # fins de réplique


@torch.no_grad()
def sample_replies(prefix: list[int], n: int, temperature: float = 0.6, top_p: float = 0.9,
                   repetition_penalty: float = 1.15, max_new_tokens: int = 80) -> list[list[int]]:
    """n répliques tirées en parallèle après prefix, chacune arrêtée à son premier retour à la ligne."""
    idx = torch.tensor([prefix], device=DEVICE).repeat(n, 1)
    generated = torch.zeros(n, 0, dtype=torch.long, device=DEVICE)
    done = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    for _ in range(max_new_tokens):
        logits = model(idx[:, -model.block_size :])[0][:, -1, :].float()
        if generated.shape[1]:  # pénalité sur les tokens déjà écrits dans la réplique (CTRL, Keskar et al., 2019)
            seen = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, generated, True)
            logits = torch.where(seen, torch.where(logits > 0, logits / repetition_penalty, logits * repetition_penalty), logits)
        probs = F.softmax(logits / temperature, dim=-1)
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)  # top-p : les tokens qui font p de masse
        sorted_probs[sorted_probs.cumsum(dim=-1) - sorted_probs > top_p] = 0.0
        choice = torch.multinomial(sorted_probs / sorted_probs.sum(dim=-1, keepdim=True), num_samples=1)
        next_token = sorted_idx.gather(-1, choice)
        idx, generated = torch.cat((idx, next_token), 1), torch.cat((generated, next_token), 1)
        done |= torch.isin(next_token[:, 0], NEWLINE)
        if done.all():
            break
    replies = []
    for row in generated.tolist():
        end = next((j for j, t in enumerate(row) if t in set(NEWLINE.tolist())), len(row) - 1)
        replies.append(row[: end + 1])
    return replies


@torch.no_grad()
def mean_logprob(prefix: list[int], replies: list[list[int]]) -> torch.Tensor:
    """La log-probabilité moyenne (par token) de chaque réplique quand elle suit prefix."""
    rows = [((prefix + r)[-model.block_size :], len(r)) for r in replies]
    idx = torch.zeros(len(rows), max(len(s) for s, _ in rows), dtype=torch.long, device=DEVICE)
    for i, (s, _) in enumerate(rows):
        idx[i, : len(s)] = torch.tensor(s, device=DEVICE)
    logprobs = F.log_softmax(model(idx)[0], dim=-1)
    out = []
    for i, (s, n) in enumerate(rows):
        positions = torch.arange(len(s) - n - 1, len(s) - 1, device=DEVICE)  # chaque token prédit par le précédent
        out.append(logprobs[i, positions, idx[i, len(s) - n : len(s)]].mean())
    return torch.stack(out)


def words(text: str) -> list[str]:
    return re.sub(r"\W+", " ", text.lower()).split()


def reply(prompt: str, character: str, said: list[str] = (), candidates: int = 16) -> str:
    """La meilleure de `candidates` répliques de `character` pour continuer `prompt` (voir plus haut)."""
    prefix = encode(prompt)
    message = prompt.rstrip().split("\n")[-2].partition(": ")[2] if "\n" in prompt else ""
    banned = {" ".join(words(t)) for t in [*said, message]}
    replies = sample_replies(prefix, candidates)
    texts = [decode(r).split("\n")[0].strip() for r in replies]
    keep = [i for i, t in enumerate(texts) if words(t) and " ".join(words(t)) not in banned] or list(range(len(texts)))
    kept = [replies[i] for i in keep]
    score = mean_logprob(prefix, kept) - 0.5 * mean_logprob(encode(f"\n{character}:"), kept)  # anti-passe-partout
    score += 0.3 * torch.log(torch.tensor([float(len(r)) for r in kept], device=DEVICE))   # bonus de longueur
    used = {w for t in said for w in words(t) if len(w) >= 4}
    for j, i in enumerate(keep):  # mots déjà employés par le personnage, et mots répétés dans la réplique
        long_words, all_words = {w for w in words(texts[i]) if len(w) >= 4}, words(texts[i])
        score[j] -= (len(long_words & used) / len(long_words) if long_words else 0.0) + 1 - len(set(all_words)) / max(1, len(all_words))
    return texts[keep[int(score.argmax())]] or "…"


def conversation(me: str, character: str, messages: list[str]) -> None:
    """Une petite conversation : chaque message de `me`, puis la réponse de `character`."""
    script, said = "", []
    for message in messages:
        script += f"{me}: {message}\n{character}:"
        answer = reply(script, character, said)
        print(f"{me:>7} : {message}\n{character:>7} : {answer}\n")
        script += f" {answer}\n"
        said.append(answer)


torch.manual_seed(SEED)
conversation("Bart", "Homer", ["Papa, je peux avoir un poney ?", "Pourquoi ?", "Tu m'aimes ?"])
conversation("Homer", "Moe", ["Moe, une bière s'il te plaît.", "Tu me fais crédit ?"])
conversation("Marge", "Homer", ["Homer, tu as encore oublié notre anniversaire !"])
''')

# ======================================================================================================
md(r"""
## 9. Sauvegarde

Le meilleur modèle affiné est dans `simpson.safetensors` (dans `/kaggle/working`, onglet *Output* de Kaggle). On y joint ses scores. Pour discuter avec lui dans l'interface du projet : le copier dans `checkpoints/` puis `uv run app.py`.
""")
code(r'''
final = OUT / f"dialogues_bpe8192_gpt2_modded_notebook_{MODE}.safetensors"
save_checkpoint(final, model, mode=MODE, optimizer=OPTIMIZER, test_bits_per_char=scores["test"]["bits_per_char"])
final.with_suffix(".json").write_text(json.dumps({"arch": "modern", "mode": MODE, "optimizer": OPTIMIZER,
                                                  "params": n_params, **{f"{p}_{k}": v for p, s in scores.items() for k, v in s.items()}},
                                                 indent=2))
print(f"→ {final} ({final.stat().st_size / 1e6:.0f} Mo)")
''')

# ======================================================================================================
md(r"""
## Annexe A · La préparation des corpus

Faite une fois, dans des notebooks à part (`super-modele-prep`, `super-modele-dialogue`, sur le CPU de Kaggle) : on télécharge, on nettoie, on encode, et on range les tokens en `uint16`. Un document sur 200 va à la validation.

Pour les sous-titres, deux étapes comptent :

- **le nettoyage** de chaque réplique : tirets de changement de locuteur, bruitages « (rire) », « [musique] », ♪, balises ;
- **la décontamination** : les Simpson sont eux-mêmes sous-titrés. On retire leurs épisodes (un bloc de 100 répliques qui nomme au moins deux personnages de la famille, et ses voisins s'il en nomme un), puis toute réplique qui partage 10 mots de suite avec la validation ou le test. Sinon le modèle aurait vu des répliques du test, et son score serait faussé. Sur les 203 M de répliques, ~4 000 blocs d'épisodes ont été retirés.

Les fonctions, appliquées ici à quelques lignes :
""")
code(r'''
FAMILY = re.compile(r"\b(Homer|Marge|Bart|Lisa|Maggie|Milhouse|Flanders|Krusty|Smithers|Apu|Skinner|Wiggum)\b")
NOISE = re.compile(r"<[^>]*>|\([^()]*\)|\[[^\[\]]*\]|♪[^♪]*♪|[♪♫#]")      # balises, bruitages, musique
DASH = re.compile(r"^\s*[-–—]+\s*|\s+[-–—]\s+(?=[A-ZÀ-ÖØ-Þ])")           # tirets de changement de locuteur


def clean_line(line: str) -> str:
    """Une réplique de sous-titre sans bruitage, balise ni tiret ; vide si elle ne contient pas de lettre."""
    line = DASH.sub(" ", NOISE.sub(" ", line))
    line = re.sub(r"\s+", " ", line).strip(" :;-–—")
    return line if re.search(r"[^\W\d_]", line) else ""


def is_simpsons_episode(block: list[str]) -> bool:
    """Un bloc de répliques qui nomme au moins deux personnages de la famille (ou « Simpson » et l'un d'eux)."""
    text = "\n".join(block)
    names = set(FAMILY.findall(text))
    return len(names) >= 2 or ("Simpson" in text and len(names) >= 1)


raw = ["-Pour le royaume de France !", "(rire)", "<i>♪ Musique ♪</i>", "- Qu'avez-vous fait, Henri ?",
       "Homer, tu as encore bu ?", "Pas du tout, Marge ! - Bart, file dans ta chambre."]
for line in raw:
    print(f"{line!r:58} → {clean_line(line)!r}")
print("épisode des Simpson ?", is_simpsons_episode([clean_line(l) for l in raw]))
''')

md(r"""
## Annexe B · Aller plus vite : plusieurs GPU, TPU

- **2 × T4 en vrai parallèle** : `DataParallel` (utilisé ici) est simple mais copie le modèle à chaque pas. Le script `train_gpt2.py` du projet utilise `DistributedDataParallel` (un processus par GPU, lancé avec `torchrun`) et `torch.compile` : ~1,5 fois plus rapide.
- **TPU v5e-8** : le script `train_tpu.py` fait tourner le même modèle et les mêmes optimiseurs avec PyTorch/XLA sur les 8 cœurs du TPU. Deux différences : le learning rate est passé au TPU comme un tenseur (sinon le graphe est recompilé à chaque pas), et les 8 processus moyennent leurs gradients à chaque pas.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "super_modele.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "romaindl/super-modele-notebook", "title": "Super modele notebook",
    "code_file": "super_modele.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
    "dataset_sources": ["romaindl/super-modele-donnees"],
    "kernel_sources": ["romaindl/super-modele-prep", "romaindl/super-modele-dialogue"], "competition_sources": []},
    indent=2, ensure_ascii=False))
print(f"{len(cells)} cellules → {OUT / 'super_modele.ipynb'}")
