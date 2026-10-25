"""Le GPT, ses tokenizers, les checkpoints et la génération : le code commun au notebook, au chat et à l'interface.

C'est le GPT des scripts 05 à 08 du cours Creative Technology, écrit une seule fois : transformer_from_scratch.ipynb
l'importe (et en affiche le code) au lieu d'en garder une copie.
"""

import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch import nn

# --- Tokenizers -----------------------------------------------------------------------------------


class CharTokenizer:
    """Un token = un caractère (script 01).

    Args:
        chars: Le vocabulaire, trié pour que l'indice d'un caractère ne change jamais.
    """

    def __init__(self, chars: list[str]) -> None:
        self.chars = chars
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @classmethod
    def train(cls, texts: list[str]) -> "CharTokenizer":
        """Le vocabulaire est l'ensemble des caractères distincts des textes."""
        return cls(sorted(set().union(*map(set, texts))))

    def encode(self, s: str) -> list[int]:
        """Texte → liste d'indices."""
        unknown = set(s) - self.stoi.keys()
        if unknown:
            raise ValueError(f"caractères absents du vocabulaire : {''.join(sorted(unknown))!r}")
        return [self.stoi[c] for c in s]

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Plusieurs textes d'un coup."""
        return [self.encode(t) for t in texts]

    def decode(self, ids: list[int]) -> str:
        """Liste d'indices → texte."""
        return "".join(self.itos[i] for i in ids)

    def known(self, s: str) -> str:
        """`s` sans les caractères absents du vocabulaire."""
        return "".join(c for c in s if c in self.stoi)

    def state(self) -> dict:
        """Ce qu'il faut sauvegarder pour reconstruire le tokenizer."""
        return {"type": "char", "chars": self.chars}


class BPETokenizer:
    """BPE au niveau des octets (script 08), avec la bibliothèque `tokenizers` de Hugging Face.

    Même algorithme que `train_bpe` du script 08 : on part des 256 octets UTF-8 et on fusionne à répétition la
    paire la plus fréquente. Le texte est d'abord pré-découpé comme dans GPT-2 (script 08, chapitre 8) : les
    fusions ne traversent pas les frontières entre mots, nombres, ponctuation et espaces. La bibliothèque est
    écrite en Rust : apprendre des milliers de fusions sur des centaines de mégaoctets prend quelques secondes.

    Args:
        tokenizer: Un tokenizer Hugging Face déjà entraîné.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()

    @classmethod
    def train(cls, texts, vocab_size: int) -> "BPETokenizer":
        """Apprend les fusions sur des textes (une liste ou un itérateur), jusqu'à `vocab_size` tokens (≥ 256)."""
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # octets + découpage de GPT-2
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                      show_progress=False)
        tokenizer.train_from_iterator(texts, trainer)
        return cls(tokenizer)

    def encode(self, s: str) -> list[int]:
        """Texte → tokens."""
        return self.tokenizer.encode(s).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Plusieurs textes d'un coup, encodés en parallèle."""
        return [encoding.ids for encoding in self.tokenizer.encode_batch(texts)]

    def decode(self, ids: list[int]) -> str:
        """Tokens → texte. Un caractère coupé en deux (octets incomplets) s'affiche `�` au lieu de planter."""
        return self.tokenizer.decode(ids)

    def known(self, s: str) -> str:
        """Partir des octets permet d'écrire n'importe quel texte : rien à retirer."""
        return s

    def state(self) -> dict:
        """Ce qu'il faut sauvegarder pour reconstruire le tokenizer : sa description JSON complète."""
        return {"type": "hf-bpe", "tokenizer": self.tokenizer.to_str()}


def tokenizer_from_state(state: dict) -> CharTokenizer | BPETokenizer:
    """Reconstruit un tokenizer sauvegardé dans un checkpoint."""
    if state["type"] == "char":
        return CharTokenizer(state["chars"])
    if state["type"] == "hf-bpe":
        return BPETokenizer(Tokenizer.from_str(state["tokenizer"]))
    raise ValueError(f"tokenizer inconnu dans le checkpoint : {state['type']!r}")


# --- Modèle ---------------------------------------------------------------------------------------

# True : l'attention passe par `F.scaled_dot_product_attention` (FlashAttention sur GPU NVIDIA), la même formule
# softmax(QKᵀ/√d)·V calculée par blocs : 2 à 3 fois plus rapide sur un gros modèle, sans changer les poids.
# False (par défaut) : le calcul écrit à la main, comme dans le cours.
FAST_ATTENTION = False


class Head(nn.Module):
    """Les poids d'une tête de self-attention masquée (script 05) : key, query, value, et le masque du futur.

    Le calcul se fait pour toutes les têtes à la fois, dans `MultiHeadAttention`.

    Args:
        n_embd: Dimension des vecteurs en entrée.
        head_size: Dimension des query / key / value.
        block_size: Longueur maximale de séquence (taille du masque).
        dropout: Taux de dropout sur les poids d'attention.
    """

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # persistent=False : le masque ne s'apprend pas, il n'a rien à faire dans le checkpoint.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)), persistent=False)
        self.dropout = nn.Dropout(dropout)


class MultiHeadAttention(nn.Module):
    """Plusieurs têtes en parallèle, concaténées puis projetées (script 06).

    Le script 06 calcule `torch.cat([h(x) for h in self.heads])`, une tête après l'autre. Ici, les matrices
    query / key / value des têtes sont empilées et l'attention se calcule pour toutes les têtes d'un coup,
    dans une dimension de plus : mêmes paramètres, même résultat au bit près, mais quelques gros calculs au
    lieu de dizaines de petits (~20 % plus rapide sur Mac).
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        assert n_embd % n_head == 0, f"n_embd ({n_embd}) doit être divisible par n_head ({n_head})"
        self.heads = nn.ModuleList(Head(n_embd, n_embd // n_head, block_size, dropout) for _ in range(n_head))
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        n_head = len(self.heads)

        def project(name: str) -> torch.Tensor:
            weight = torch.cat([getattr(h, name).weight for h in self.heads])  # (C, C) : les têtes empilées
            return (x @ weight.T).view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, n_head, T, head_size)

        q, k, v = project("query"), project("key"), project("value")
        if FAST_ATTENTION:  # la même formule, calculée par blocs sans stocker la matrice T × T (FlashAttention)
            p = self.heads[0].dropout.p if self.training else 0.0
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)
            return self.dropout(self.proj(out.transpose(1, 2).reshape(B, T, C)))
        wei = q @ k.transpose(-2, -1) * (C // n_head) ** -0.5  # facteur 1/√d (script 05, chapitre 6)
        wei = wei.masked_fill(self.heads[0].tril[:T, :T] == 0, float("-inf"))  # le futur est interdit
        wei = self.heads[0].dropout(F.softmax(wei, dim=-1))
        return self.dropout(self.proj((wei @ v).transpose(1, 2).reshape(B, T, C)))


class FeedForward(nn.Module):
    """MLP appliqué à chaque position indépendamment, couche cachée 4 fois plus large."""

    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Bloc Transformer pre-norm : attention (communication) puis feed-forward (calcul), avec résiduels."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


class GPTLanguageModel(nn.Module):
    """GPT : embeddings de token et de position, `n_layer` blocs, LayerNorm finale, tête de sortie.

    Args:
        vocab_size: Nombre de tokens du vocabulaire.
        block_size: Fenêtre de contexte, en tokens.
        n_embd: Dimension des embeddings.
        n_head: Nombre de têtes d'attention par bloc.
        n_layer: Nombre de blocs.
        dropout: Taux de dropout.
    """

    def __init__(self, vocab_size: int, block_size: int, n_embd: int, n_head: int, n_layer: int,
                 dropout: float) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Poids N(0, 0.02) et biais nuls, comme dans le `gpt.py` de Karpathy : la loss de départ vaut ln(vocab)."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """Renvoie `(logits, loss)` ; la loss vaut `None` sans cibles (la convention du script 06)."""
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=idx.device))
        logits = self.lm_head(self.ln_f(self.blocks(x)))
        if targets is None:
            return logits, None
        # cross_entropy attend (N, C) : on aplatit batch et temps ; calcul en float32, même sous bf16.
        return logits, F.cross_entropy(logits.view(B * T, -1).float(), targets.view(B * T))


# --- Modèle moderne : la recette modded-nanoGPT --------------------------------------------------
#
# Le même Transformer (attention, MLP, résiduels), avec les pièces du speedrun modded-nanoGPT (Keller Jordan
# et al., 2024-2025) : RoPE, RMSNorm sans paramètre, QK-norm, ReLU², pas de biais, projections de sortie à
# zéro, connexions en U entre la première et la seconde moitié des blocs, logits plafonnés. Il s'entraîne avec
# Muon (plus bas) pour les matrices des blocs et Adam pour le reste. GPTLanguageModel reste la référence.


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """RMSNorm sans paramètre appris : chaque vecteur est ramené à une moyenne quadratique de 1.

    Calculée en float32 : en fp16 (GPU T4), le carré d'une valeur au-delà de 256 dépasserait 65 504.
    """
    return F.rms_norm(x.float(), (x.size(-1),)).type_as(x)


class Rotary(nn.Module):
    """RoPE (Su et al., 2021) : la position t est codée en faisant tourner chaque paire de coordonnées de q et k
    d'un angle t·θᵢ. Le produit q·k ne dépend alors que de la distance entre les deux positions."""

    def __init__(self, head_dim: int, block_size: int, base: float = 10_000.0) -> None:
        super().__init__()
        theta = base ** (-torch.arange(0, head_dim // 2, dtype=torch.float32) / (head_dim // 2))
        angles = torch.outer(torch.arange(block_size, dtype=torch.float32), theta)  # (T, head_dim / 2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x : (B, n_head, T, head_dim)
        T = x.size(-2)
        cos, sin = self.cos[:T].to(x.dtype), self.sin[:T].to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class ModernAttention(nn.Module):
    """Self-attention masquée : q, k, v d'un seul produit, QK-norm, RoPE, puis l'attention causale de PyTorch
    (la même formule softmax(qkᵀ/√d)·v que le script 05, calculée par blocs)."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        assert n_embd % n_head == 0, f"n_embd ({n_embd}) doit être divisible par n_head ({n_head})"
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.rotary = Rotary(n_embd // n_head, block_size)
        self.attn_dropout = nn.Dropout(dropout)  # lu comme un taux par l'attention de PyTorch
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        q, k = self.rotary(rms_norm(q)), self.rotary(rms_norm(k))  # QK-norm : des scores bornés, un entraînement stable
        p = self.attn_dropout.p if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)
        return self.dropout(self.proj(y.transpose(1, 2).reshape(B, T, C)))


class ModernMLP(nn.Module):
    """MLP à couche cachée 4 fois plus large, activation ReLU² (ReLU puis carré), sans biais."""

    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc(x))
        if h.dtype == torch.float16:  # fp16 (GPU T4) : le carré d'une valeur au-delà de 256 dépasserait 65 504
            h = h.float().square()
            s = h.amax(dim=-1, keepdim=True).clamp(min=1.0)  # ramène chaque token sous 1, puis le rétablit :
            return self.dropout(self.proj((h / s).half()).float() * s)  # même résultat (proj est linéaire)
        return self.dropout(self.proj(h.square()))


class ModernBlock(nn.Module):
    """Bloc pre-norm, comme `Block`, avec RMSNorm sans paramètre."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.attn = ModernAttention(n_embd, n_head, block_size, dropout)
        self.mlp = ModernMLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(rms_norm(x))
        return x + self.mlp(rms_norm(x))


class ModernGPT(nn.Module):
    """GPT modded-nanoGPT : mêmes arguments que `GPTLanguageModel` (et le même `forward`), plus `softcap`.

    Pas d'embedding de position (RoPE s'en charge). Les blocs de la seconde moitié reçoivent aussi la sortie
    des blocs symétriques de la première (connexions en U, poids appris). Les projections de sortie des blocs et
    la tête démarrent à zéro : au départ chaque bloc est l'identité et la loss vaut exactement ln(vocab_size).
    """

    def __init__(self, vocab_size: int, block_size: int, n_embd: int, n_head: int, n_layer: int,
                 dropout: float = 0.0, softcap: float = 30.0) -> None:
        super().__init__()
        self.block_size, self.softcap = block_size, softcap
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList(ModernBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer))
        self.skip_weights = nn.Parameter(torch.ones(n_layer // 2))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        for name, param in self.named_parameters():
            if name.endswith("proj.weight") or name == "lm_head.weight":
                nn.init.zeros_(param)
            elif param.dim() == 2 and not name.startswith("embed"):
                bound = 3**0.5 * 0.5 * param.size(1) ** -0.5  # uniforme, écart-type 0,5/√(entrées)
                nn.init.uniform_(param, -bound, bound)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """Renvoie `(logits, loss)` ; la loss vaut `None` sans cibles."""
        x = rms_norm(self.embed(idx))
        half = len(self.blocks) // 2
        skips = []
        for i, block in enumerate(self.blocks):
            if i >= len(self.blocks) - half:  # seconde moitié : la sortie du bloc symétrique
                x = x + self.skip_weights[i - (len(self.blocks) - half)] * skips.pop()
            x = block(x)
            if i < half:
                skips.append(x)
        logits = self.lm_head(rms_norm(x)).float()
        logits = self.softcap * torch.tanh(logits / self.softcap)  # logits plafonnés à ±softcap
        if targets is None:
            return logits, None
        B, T = idx.shape
        return logits, F.cross_entropy(logits.view(B * T, -1), targets.reshape(B * T))


# --- Optimiseurs -----------------------------------------------------------------------------------------
#
# Écrits à la main pour deux raisons : Muon n'existe pas dans PyTorch, et sur TPU (XLA) un learning rate qui
# change à chaque pas doit être un tenseur, sinon le graphe est recompilé à chaque pas. `step(mult)` applique
# le planning : le learning rate de chaque groupe est multiplié par `mult`.


@torch.no_grad()
def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """La matrice orthogonale « la plus proche » de G (UVᵀ si G = USVᵀ), approchée par 5 itérations de
    Newton-Schulz quintiques (coefficients de modded-nanoGPT) : seulement des produits matriciels."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)  # valeurs singulières ≤ 1
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X.mT if transposed else X


class Muon:
    """Muon (Keller Jordan, 2024) : momentum de Nesterov, puis la mise à jour est orthogonalisée — toutes ses
    directions ont la même amplitude, les rares ne sont plus noyées par les dominantes. Pour les matrices des
    blocs seulement (pas les embeddings, la tête ni les scalaires).

    `normuon=True` : NorMuon (Li et al., 2025). Après l'orthogonalisation, les neurones (lignes de la matrice)
    n'ont pas tous la même amplitude ; on suit, pour chacun, la moyenne glissante (β₂) du carré de sa mise à
    jour et on divise sa ligne par la racine, comme Adam mais avec un seul nombre par neurone. La norme totale
    de la mise à jour reste celle de Muon : ses learning rates restent valables.
    """

    def __init__(self, params: list[torch.Tensor], lr: float = 0.05, momentum: float = 0.95,
                 normuon: bool = False, beta2: float = 0.95) -> None:
        self.params, self.lr, self.momentum = [p for p in params], lr, momentum
        self.normuon, self.beta2 = normuon, beta2
        self.buffers = [torch.zeros_like(p) for p in self.params]
        self.second = [torch.zeros(p.size(0), 1, device=p.device) for p in self.params] if normuon else []

    @torch.no_grad()
    def step(self, mult: float | torch.Tensor = 1.0) -> None:
        for i, (p, buf) in enumerate(zip(self.params, self.buffers)):
            if p.grad is None:
                continue
            buf.lerp_(p.grad, 1 - self.momentum)
            update = newton_schulz(p.grad.lerp(buf, self.momentum))  # Nesterov
            if self.normuon:  # une échelle par neurone (ligne), puis la norme de Muon
                v = self.second[i]
                v.lerp_(update.square().mean(dim=-1, keepdim=True), 1 - self.beta2)
                scaled = update / (v.sqrt() + 1e-8)
                update = scaled * (update.norm() / (scaled.norm() + 1e-8))
            scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
            p.sub_(update.to(p.dtype) * (mult * self.lr * scale))

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def state_dict(self) -> dict:
        return {"buffers": [b.cpu() for b in self.buffers], "second": [v.cpu() for v in self.second]}

    def load_state_dict(self, state: dict) -> None:
        self.buffers = [b.to(p.device) for b, p in zip(state["buffers"], self.params)]
        if self.normuon and state.get("second"):
            self.second = [v.to(p.device) for v, p in zip(state["second"], self.params)]


class AdamW:
    """AdamW (Loshchilov & Hutter) par groupes {"params", "lr", "weight_decay", "betas"} ; même calcul que
    torch.optim.AdamW, le compteur de pas en tenseur (pas de constante qui change sur TPU)."""

    def __init__(self, groups: list[dict], eps: float = 1e-8) -> None:
        self.groups, self.eps = groups, eps
        self.m = [[torch.zeros_like(p) for p in g["params"]] for g in groups]
        self.v = [[torch.zeros_like(p) for p in g["params"]] for g in groups]
        device = groups[0]["params"][0].device
        self.t = torch.zeros((), device=device)

    @torch.no_grad()
    def step(self, mult: float | torch.Tensor = 1.0) -> None:
        self.t += 1
        for group, ms, vs in zip(self.groups, self.m, self.v):
            b1, b2 = group.get("betas", (0.9, 0.95))
            lr = mult * group["lr"]
            c1, c2 = 1 - b1**self.t, 1 - b2**self.t
            for p, m, v in zip(group["params"], ms, vs):
                if p.grad is None:
                    continue
                if group.get("weight_decay"):
                    p.mul_(1 - lr * group["weight_decay"])
                m.lerp_(p.grad, 1 - b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                p.sub_(lr * (m / c1) / ((v / c2).sqrt() + self.eps))

    def zero_grad(self) -> None:
        for g in self.groups:
            for p in g["params"]:
                p.grad = None

    def state_dict(self) -> dict:
        return {"m": [[x.cpu() for x in ms] for ms in self.m], "v": [[x.cpu() for x in vs] for vs in self.v],
                "t": self.t.cpu()}

    def load_state_dict(self, state: dict) -> None:
        device = self.t.device
        self.m = [[x.to(device) for x in ms] for ms in state["m"]]
        self.v = [[x.to(device) for x in vs] for vs in state["v"]]
        self.t = state["t"].to(device)


def modern_optimizers(model: ModernGPT, muon_lr: float = 0.05, embed_lr: float = 0.6, head_lr: float = 0.008,
                      scalar_lr: float = 0.04, normuon: bool = False) -> list:
    """Les optimiseurs de modded-nanoGPT : Muon (ou NorMuon) pour les matrices des blocs, Adam pour l'embedding,
    la tête et les poids des connexions en U (learning rates du speedrun, pour des lots de ~0,5 M tokens)."""
    matrices = [p for name, p in model.blocks.named_parameters() if p.dim() == 2]
    adam = AdamW([{"params": [model.embed.weight], "lr": embed_lr, "betas": (0.8, 0.95)},
                  {"params": [model.lm_head.weight], "lr": head_lr, "betas": (0.8, 0.95)},
                  {"params": [model.skip_weights], "lr": scalar_lr, "betas": (0.8, 0.95)}])
    return [Muon(matrices, lr=muon_lr, normuon=normuon), adam]


def wsd_schedule(progress: float, warmup: float = 0.01, cooldown: float = 0.4) -> float:
    """Multiplicateur du learning rate : courte montée, plateau, puis descente linéaire jusqu'à 0 sur les
    derniers `cooldown` (warmup-stable-decay, le planning du speedrun)."""
    if progress < warmup:
        return (progress + 1e-3) / warmup
    if progress < 1 - cooldown:
        return 1.0
    return max(0.0, (1 - progress) / cooldown)


# --- Checkpoints ------------------------------------------------------------------------------------


def save_checkpoint(path: Path, model: GPTLanguageModel, config: dict,
                    tokenizer: CharTokenizer | BPETokenizer, **info) -> None:
    """Les poids au format safetensors, avec dans ses métadonnées tout ce qu'il faut pour s'en servir.

    Args:
        path: Fichier `.safetensors`.
        model: Le modèle.
        config: Les arguments de `GPTLanguageModel`, pour reconstruire l'architecture.
        tokenizer: Le tokenizer ; sans lui, les indices ne veulent rien dire.
        **info: Ce qu'on veut garder en plus (pas, loss de validation, hyperparamètres…), sérialisable en JSON.
    """
    tensors = {name: t.detach().cpu().contiguous() for name, t in model.state_dict().items()}
    metadata = {"config": json.dumps(config), "tokenizer": json.dumps(tokenizer.state(), ensure_ascii=False),
                "info": json.dumps(info, ensure_ascii=False)}
    save_file(tensors, str(path), metadata=metadata)


def load_checkpoint(path: Path, device: str) -> tuple[GPTLanguageModel | ModernGPT, CharTokenizer | BPETokenizer, dict]:
    """Recharge un modèle, son tokenizer et ses infos (`step`, `val_loss`…) depuis un `.safetensors`.

    `config["arch"] == "modern"` : un `ModernGPT` ; sinon le GPT du cours.
    """
    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata()
    config = json.loads(metadata["config"])
    arch = config.get("arch", "course")
    model = (ModernGPT if arch == "modern" else GPTLanguageModel)(**{k: v for k, v in config.items() if k != "arch"})
    model.load_state_dict(load_file(str(path)))
    model.to(device).eval()
    info = {"config": config, **json.loads(metadata["info"])}
    return model, tokenizer_from_state(json.loads(metadata["tokenizer"])), info


# --- Génération -------------------------------------------------------------------------------------


@torch.no_grad()
def generate(model: GPTLanguageModel, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
             top_k: int | None = None, top_p: float | None = None) -> torch.Tensor:
    """Prolonge `idx` de `max_new_tokens` tokens (la fonction `generate` du script 07, plus le top-p).

    Args:
        model: Le GPT.
        idx: Contexte de départ, `(B, T)`.
        max_new_tokens: Nombre de tokens à générer.
        temperature: Divise les logits ; < 1 concentre, > 1 aplatit, 0 = glouton.
        top_k: Si fourni, ne tire que parmi les `top_k` tokens les plus probables.
        top_p: Si fourni, ne tire que dans le noyau de probabilité cumulée `top_p`.

    Returns:
        Le contexte prolongé, `(B, T + max_new_tokens)`.
    """
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(idx[:, -model.block_size :])[0][:, -1, :]  # seulement la dernière position
        if temperature == 0:
            idx_next = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            if top_p is not None:
                sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
                # On retire un choix si la masse cumulée AVANT lui dépasse déjà p : le premier est toujours gardé.
                sorted_probs[sorted_probs.cumsum(dim=-1) - sorted_probs > top_p] = 0.0
                choice = torch.multinomial(sorted_probs / sorted_probs.sum(dim=-1, keepdim=True), num_samples=1)
                idx_next = sorted_idx.gather(-1, choice)
            else:
                idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def _newline_ids(tokenizer: CharTokenizer | BPETokenizer) -> torch.Tensor:
    """Les tokens qui contiennent un retour à la ligne : ils terminent une réplique."""
    if not hasattr(tokenizer, "_newline_ids"):
        tokenizer._newline_ids = torch.tensor([i for i in range(tokenizer.vocab_size) if "\n" in tokenizer.decode([i])])
    return tokenizer._newline_ids


@torch.no_grad()
def _mean_logprob(model: GPTLanguageModel, prefix: list[int], replies: list[list[int]], device: str) -> torch.Tensor:
    """Log-probabilité moyenne de chaque réplique (une liste de tokens) quand elle suit `prefix`."""
    rows = []
    for r in replies:
        r = r[-(model.block_size - 1) :]  # au moins un token de contexte dans la fenêtre
        rows.append(((prefix + r)[-model.block_size :], len(r)))
    width = max(len(s) for s, _ in rows)
    idx = torch.zeros(len(rows), width, dtype=torch.long, device=device)  # complété à droite : sans effet,
    for i, (s, _) in enumerate(rows):                                      # l'attention ne regarde que le passé
        idx[i, : len(s)] = torch.tensor(s, device=device)
    logprobs = F.log_softmax(model(idx)[0].float(), dim=-1)
    scores = []
    for i, (s, n) in enumerate(rows):
        start = len(s) - n  # chaque token de la réplique est prédit depuis la position précédente
        positions = torch.arange(start - 1, len(s) - 1, device=device)
        scores.append(logprobs[i, positions, idx[i, start : len(s)]].mean())
    return torch.stack(scores)


@torch.no_grad()
def _sample_replies(model: GPTLanguageModel, prefix: list[int], n: int, device: str, newline: torch.Tensor,
                    temperature: float, top_p: float | None, repetition_penalty: float,
                    max_new_tokens: int) -> list[list[int]]:
    """`n` répliques tirées en parallèle après `prefix`, chacune arrêtée à son premier retour à la ligne."""
    idx = torch.tensor([prefix], dtype=torch.long, device=device).repeat(n, 1)
    generated = torch.zeros(n, 0, dtype=torch.long, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    for _ in range(max_new_tokens):
        logits = model(idx[:, -model.block_size :])[0][:, -1, :].float()
        if generated.shape[1] and repetition_penalty != 1.0:  # pénalité « CTRL » sur les tokens déjà écrits
            seen = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, generated, True)
            penalized = torch.where(logits > 0, logits / repetition_penalty, logits * repetition_penalty)
            logits = torch.where(seen, penalized, logits)
        probs = F.softmax(logits / max(temperature, 1e-5), dim=-1)
        if top_p is not None:
            sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
            sorted_probs[sorted_probs.cumsum(dim=-1) - sorted_probs > top_p] = 0.0
            choice = torch.multinomial(sorted_probs / sorted_probs.sum(dim=-1, keepdim=True), num_samples=1)
            idx_next = sorted_idx.gather(-1, choice)
        else:
            idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        generated = torch.cat((generated, idx_next), dim=1)
        done |= torch.isin(idx_next[:, 0], newline)
        if done.all():
            break
    stop = set(newline.tolist())
    replies = []
    for row in generated.tolist():
        end = next((j for j, t in enumerate(row) if t in stop), len(row) - 1)
        replies.append(row[: end + 1])  # la réplique, retour à la ligne compris
    return replies


def _normalized(text: str) -> str:
    """Le texte sans casse ni ponctuation, pour reconnaître une redite (« Non. » = « non ! »)."""
    return re.sub(r"\W+", " ", text.lower()).strip()


def _echoes(text: str, message: str) -> bool:
    """Vrai si la réplique recopie le message auquel elle répond (« On mange quoi ce soir ? » après « Papa, on
    mange quoi ce soir ? ») : un morceau de suite, ou les trois quarts de ses mots. Une réplique de un ou deux
    mots (« Quoi ? ») reste permise, sauf si c'est le message entier."""
    words, said = _normalized(text).split(), _normalized(message).split()
    if len(words) < 3 or not said:
        return False
    inside = any(said[i : i + len(words)] == words for i in range(len(said) - len(words) + 1))
    return inside or sum(w in said for w in words) / len(words) >= 0.75


def _long_words(text: str) -> set[str]:
    """Les mots d'au moins 4 lettres : ceux qui portent le sens (« poney », pas « je » ni « un »)."""
    return {w for w in _normalized(text).split() if len(w) >= 4}


@torch.no_grad()
def reply(model: GPTLanguageModel, tokenizer: CharTokenizer | BPETokenizer, prompt: str, character: str,
          device: str, candidates: int = 16, temperature: float = 0.6, top_p: float | None = 0.9,
          repetition_penalty: float = 1.15, anti_generic: float = 0.5, length_bonus: float = 0.3,
          novelty: float = 1.0, said: list[str] = (), max_new_tokens: int = 80, echo_penalty: float = 0.0) -> str:
    """La meilleure de `candidates` répliques écrites en parallèle pour continuer `prompt` (une seule ligne).

    Chaque réplique est tirée comme d'habitude (température, top-p), avec une pénalité sur les tokens qu'elle a
    déjà écrits, contre les boucles (« D'oh ! D'oh ! D'oh ! »). `said` : ce que le personnage a déjà dit dans
    la conversation. Le modèle relit la conversation et aime la recopier (« Non. », « Non. », « Non. »…) :
    une redite exacte, ou l'écho du dernier message, est écartée ; si tous les tirages en sont, on retire une
    série un peu plus inventive. Parmi les autres, on garde celle qui maximise

        log p(réplique | conversation) − anti_generic × log p(réplique | « {character}: » seul)
        + length_bonus × ln(longueur) − novelty × (part de ses mots déjà employés + part de ses mots répétés)

    (log-probabilités moyennes par token). Une réplique passe-partout (« Quoi ? ») est probable partout et perd
    des points ; une réplique liée au message n'est probable qu'après lui (Li et al., 2016, « A
    Diversity-Promoting Objective Function for Neural Conversation Models ») ; le bonus de longueur compense
    la moyenne, qui favorise sinon les répliques de deux mots ; le dernier terme évite les quasi-redites
    (« Un poney ? », puis « Une poney ? », puis « Des poney ? »), comme le blocage des n-grammes déjà dits de
    BlenderBot (Roller et al., 2020), et les répliques qui bégaient (« Saleté de pudding, saleté de pudding… »).
    """
    model.eval()
    prefix = tokenizer.encode(prompt)
    newline = _newline_ids(tokenizer).to(device)
    lines = prompt.rstrip().split("\n")
    message = lines[-2].partition(": ")[2] if len(lines) >= 2 else ""  # la ligne avant « {character}: »
    banned = {_normalized(t) for t in [*said, message]}
    replies, texts = [], []
    for attempt in range(2):  # une deuxième série, plus inventive, si la première n'a donné que des redites
        new = _sample_replies(model, prefix, candidates, device, newline, temperature + 0.3 * attempt,
                              top_p, repetition_penalty, max_new_tokens)
        replies += new
        texts += [tokenizer.decode(r).split("\n")[0].strip() for r in new]
        keep = [i for i, t in enumerate(texts) if _normalized(t) and _normalized(t) not in banned and not _echoes(t, message)]
        if keep:
            break
    keep = keep or [i for i, t in enumerate(texts) if _normalized(t)] or [0]  # rien de neuf : tant pis
    if candidates == 1 or len(keep) == 1:
        return texts[keep[0]] or "…"
    kept = [replies[i] for i in keep]
    score = _mean_logprob(model, prefix, kept, device)
    if anti_generic:
        generic = tokenizer.encode(tokenizer.known(f"\n{character}:"))
        score = score - anti_generic * _mean_logprob(model, generic, kept, device)
    score = score + length_bonus * torch.log(torch.tensor([float(len(r)) for r in kept], device=device))
    if novelty:
        used = set().union(*map(_long_words, said))
        share = [len(w & used) / len(w) if (w := _long_words(texts[i])) else 0.0 for i in keep]
        words = [_normalized(texts[i]).split() for i in keep]
        repeated = [1 - len(set(w)) / len(w) if w else 0.0 for w in words]
        score = score - novelty * (torch.tensor(share, device=device) + torch.tensor(repeated, device=device))
    if echo_penalty and (asked := _long_words(message)):  # le terme anti-générique favorise les mots du message
        # (« Krusty Burger » n'est probable qu'après lui) : la part des mots de la réplique repris du message
        echo = [len(w & asked) / len(w) if (w := _long_words(texts[i])) else 0.0 for i in keep]
        score = score - echo_penalty * torch.tensor(echo, device=device)
    return texts[keep[int(score.argmax())]]
