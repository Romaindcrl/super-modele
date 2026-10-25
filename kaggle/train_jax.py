"""Pré-entraînement de ModernGPT en JAX sur un TPU (v6e-1 de Colab), puis affinage sur les Simpson.

    python train_jax.py bench       # essai : vitesse, MFU et mémoire de chaque configuration de BENCH
    python train_jax.py train       # une session : reprise, pré-entraînement, affinage, test
    INIT=modele.safetensors python train_jax.py train   # affinage seul (FT_OPTIM=adamw ou normuon), test

Le même modèle que gpt.ModernGPT (RoPE, QK-norm, ReLU², connexions en U, logits plafonnés), les mêmes optimiseurs
(NorMuon + Adam pour le pré-entraînement, AdamW pour l'affinage), le même journal et les mêmes fichiers que
train_tpu.py (PyTorch/XLA, ~17 % du calcul de la puce) : colab_watch.py et le dashboard le lisent sans changement,
gpt.load_checkpoint relit ses modèles. Ce qui change, c'est la façon de calculer :

- un pas d'optimiseur = une seule fonction compilée (jax.jit) : les micro-lots défilent dans une boucle de XLA
  (lax.scan), leurs gradients s'additionnent en float32, puis l'optimiseur s'applique, sans repasser par Python ;
- les blocs aussi passent par lax.scan (poids empilés sur un axe « couche ») : un seul bloc à compiler ;
- l'attention sur TPU : splash attention (un noyau Pallas), qui ne stocke jamais la matrice T × T ;
- poids et état de l'optimiseur sont « donnés » au pas suivant (donate) : aucune copie en mémoire ;
- Python ne se synchronise avec le TPU que pour journaliser ; les lots sont préparés à l'avance par un thread,
  les sauvegardes écrites par un autre.

ATTN : auto (splash sur TPU si son autotest passe, sinon xla), splash, flash, xla (jax.nn.dot_product_attention) ou
explicit (le calcul écrit à la main, le seul qui gère le dropout de l'affinage). REMAT : none, dots (garde les
produits matriciels, recalcule le reste), dots_attn (dots, plus la sortie de l'attention et son logsumexp : le
backward ne relance pas le noyau splash ; utile avec splash seulement : avec flash, xla ou explicit, seule la sortie
est gardée en plus ; flash relance son noyau, xla et explicit recalculent leurs probabilités) ou full (recalcule tout
le bloc). COMPUTE_DTYPE=float32 : le calcul exact, pour les tests d'équivalence avec PyTorch.

Laboratoire de vitesse (mode bench) : BENCH_STEPS (10) pas mesurés après BENCH_WARMUP (3) pas de chauffe.
PROFILE=<dossier> : une trace jax.profiler (XProf, TensorBoard) de PROFILE_STEPS (3) pas complets, après la mesure,
dans <dossier>/<configuration>__<attention, recalcul, leviers, empreinte des flags de XLA>, avec le HLO optimisé du
pas (step.hlo.txt) ; il implique BENCH_COST=1 : dans le JSON, l'estimation de XLA (flops, octets lus et écrits ; un
lax.scan n'y compte qu'un tour) et l'analyse mémoire du programme compilé. BENCH_PARTS=1 : forward + loss seuls (un
forward d'inférence, qui n'écrit pas les résidus du backward), forward + backward (gradients, sans optimiseur) et
optimiseur seul, compilés et chronométrés à part, à côté du pas complet : où passe le temps.
peak_hbm_bytes est le pic de mémoire du processus depuis son début (JAX ne le remet jamais à zéro) : BENCH_PARTS et
PROFILE le font monter pour les configurations suivantes du même BENCH (marquées peak_hbm_is_process_max). Pour
comparer la mémoire : une configuration par processus, ou compiled_peak_bytes (le pic du programme compilé seul).

Leviers de vitesse, lus dans l'environnement ; chaque défaut reproduit le calcul ci-dessus à l'identique :
- EMBED : gather (lecture indexée, scatter-add float32 au backward) ou onehot (one_hot(idx) @ embedding sur le MXU,
  l'« iota embed » de MaxText : lignes arrondies en bfloat16 avant la norme ; gradient en produit matriciel bf16 :
  la contribution de chaque token arrondie en bf16, sommée en float32, le total arrondi en bf16) ;
- LOSS_PICK : gather ou onehot (le logit de la cible pris par un produit avec one_hot(y), fusionné par XLA : même
  résultat, sans le scatter (B, T, V) du backward) ;
- HEAD_CHUNKS : 1, ou n morceaux de la séquence pour la tête et la loss, recalculés au backward : les logits
  float32 n'existent qu'un morceau à la fois ;
- CHEAP_REMAT : 0 ou 1, pour REMAT=none (rms_norm et la fin du MLP, ReLU² puis mlp_proj, recalculés au backward
  depuis leurs entrées bf16 : ni les copies float32 des quatre rms_norm, ni relu(u), son masque et relu(u)²
  (B, T, 4C) ne restent en mémoire entre forward et backward ; seule u, la sortie de fc, est gardée) ;
- ACCUM_DTYPE : float32 ou bfloat16 (la somme des gradients des matrices et de la tête d'un micro-lot à l'autre ;
  medium : ≈ 0,6 Go de moins en mémoire, ≈ 1,2 Go de moins lus et écrits par micro-lot ; arrondie à chaque ajout) ;
- LAYER_UNROLL, MICRO_UNROLL : 1, un entier ou full (tours déroulés des lax.scan des blocs, des micro-lots ;
  LAYER_UNROLL=full avec REMAT ou CHEAP_REMAT : plus de boucle entre aller et retour, XLA peut confondre le
  recalcul avec l'aller (prevent_cse=False) et garder les résidus quand même : voir temp_bytes de BENCH_COST) ;
- SPLASH_BLOCK_Q, SPLASH_BLOCK_KV, SPLASH_BLOCK_KV_COMPUTE (aller), SPLASH_BLOCK_Q_BWD, SPLASH_BLOCK_KV_BWD
  (retour) : 0, soit SPLASH_BLOCK à l'aller et les tailles de l'aller au retour ; SPLASH_FUSED=1 : dq, dk et dv
  dans un seul noyau ; SPLASH_LAYOUT : la disposition de q, k et v dans le noyau, head,head,head (seq : la
  séquence en dimension mineure ; XLA copie le tableau transposé avant chacun des noyaux, et le noyau transpose
  lui-même q à l'aller et pour dq, k et v pour dk, dv à chaque bloc : à juger sur forward_backward_s de
  BENCH_PARTS, pas sur forward_s).
Hors défaut, LAYER_UNROLL, MICRO_UNROLL, CHEAP_REMAT, HEAD_CHUNKS et REMAT=dots_attn font les mêmes calculs, mais
XLA fusionne autrement et peut arrondir des intermédiaires en bfloat16 à d'autres endroits (excess precision) :
égaux aux arrondis bf16 près, pas au bit près (non vérifié sur TPU ; A/B au bit près :
XLA_FLAGS=--xla_allow_excess_precision=false).
Les flags de XLA passent par LIBTPU_INIT_ARGS (--xla_tpu_scoped_vmem_limit_kib=98304…) ou XLA_FLAGS, notés dans le
JSON dès qu'un levier ou un outil du laboratoire (BENCH_COST, BENCH_PARTS, PROFILE) est actif.
"""

import collections
import dataclasses
import functools
import gc
import glob
import json
import math
import os
import queue
import re
import sys
import tempfile
import threading
import time
import zlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.ad_checkpoint import checkpoint_name
from safetensors import safe_open
from safetensors.numpy import load_file, save_file


def _input_dir() -> Path:
    """Le dossier de gpt.py, du tokenizer et des Simpson : INPUT, sinon une entrée Kaggle, sinon celui du script."""
    if os.environ.get("INPUT"):
        return Path(os.environ["INPUT"])
    found = glob.glob("/kaggle/input/**/gpt.py", recursive=True)
    return Path(found[0]).parent if found else Path(__file__).resolve().parent


INPUT = _input_dir()
sys.path.insert(0, str(INPUT))
import gpt  # noqa: E402

OUT = Path(os.environ.get("OUT", "/kaggle/working"))
N_LAYER = int(os.environ.get("N_LAYER", 24))  # GPT-2 medium par défaut (~320 M paramètres avec 8 192 tokens)
N_EMBD = int(os.environ.get("N_EMBD", 1024))
N_HEAD = int(os.environ.get("N_HEAD", 16))
BLOCK = int(os.environ.get("BLOCK", 512))  # une scène de dialogue tient largement en 512 tokens
MICRO_BATCH = int(os.environ.get("MICRO_BATCH", 16))  # séquences par micro-lot
TOKENS_PER_STEP = int(os.environ.get("TOKENS_PER_STEP", 524_288))  # ~0,5 M tokens par pas, comme GPT-2
SESSION = int(os.environ.get("SESSION", 1))
HOURS_PRETRAIN = float(os.environ.get("HOURS_PRETRAIN", 7.6))  # sans budget en tokens : la durée de la session
TOTAL_PRETRAIN_HOURS = float(os.environ.get("TOTAL_PRETRAIN_HOURS", 15.2))
HOURS_FINETUNE = float(os.environ.get("HOURS_FINETUNE", 0.4))
FAKE_PRETRAIN = os.environ.get("FAKE_PRETRAIN") == "1"  # essai : les Simpson tiennent lieu de corpus général
MIX = dict((k, float(v)) for k, v in (item.split(":") for item in os.environ["MIX"].split(","))) if os.environ.get("MIX") else None
LR_SCALE = float(os.environ.get("LR_SCALE", 1.0))  # multiplie les learning rates de NorMuon et d'Adam
COMPUTE_DTYPE = {"bf16": "bfloat16", "f32": "float32"}.get(os.environ.get("COMPUTE_DTYPE", "bfloat16"),
                                                          os.environ.get("COMPUTE_DTYPE", "bfloat16"))
NS_DTYPE = {"bf16": "bfloat16", "f32": "float32"}.get(os.environ.get("NS_DTYPE", "bfloat16"), os.environ.get("NS_DTYPE", "bfloat16"))
ATTN = os.environ.get("ATTN", "auto")
REMAT = os.environ.get("REMAT", "none")
FT_REMAT = os.environ.get("FT_REMAT", "dots")  # l'affinage passe par l'attention explicite (dropout) : plus de mémoire
FT_MICRO_BATCH = int(os.environ.get("FT_MICRO_BATCH", MICRO_BATCH))
FT_OPTIM = os.environ.get("FT_OPTIM", "adamw")  # l'optimiseur de l'affinage : adamw (la meilleure recette mesurée) ou normuon
FT_LR = float(os.environ.get("FT_LR", 1.5e-4))
FT_SCALE = float(os.environ.get("FT_SCALE", 0.1))
FT_DROPOUT = float(os.environ.get("FT_DROPOUT", 0.1))
INIT = os.environ.get("INIT")  # affinage seul : les poids d'un modèle déjà pré-entraîné (.safetensors de gpt.py)
IMPORT_TORCH_STATE = os.environ.get("IMPORT_TORCH_STATE")  # un pretrain_state.pt de train_tpu.py, à continuer ici
FT_TOKENS = float(os.environ.get("FT_TOKENS", 0))  # affinage : un budget en tokens au lieu de HOURS_FINETUNE
PRETRAIN_TOKENS = float(os.environ.get("PRETRAIN_TOKENS", 0))  # pré-entraînement : tokens au total, reprises comprises
CORPUS_CAP = float(os.environ.get("CORPUS_CAP", 0))  # tokens du corpus général gardés (le début de chacun) ; 0 : tout
EVAL_EVERY_S = float(os.environ.get("EVAL_EVERY_S", 600))
FT_EVAL_EVERY_S = float(os.environ.get("FT_EVAL_EVERY_S", 60))  # l'affinage est court : on mesure plus souvent
FIRST_EVAL = int(os.environ.get("FIRST_EVAL", 20))  # la première mesure, en pas après le début de l'étape
SAVE_EVERY_S = float(os.environ.get("SAVE_EVERY_S", 0))  # pré-entraînement : sauvegardes au plus si souvent
PROGRESS_EVERY_S = float(os.environ.get("PROGRESS_EVERY_S", 60))
EVAL_BATCHES = int(os.environ.get("EVAL_BATCHES", 10))
PEAK_TFLOPS = float(os.environ.get("PEAK_TFLOPS", 918))  # TPU v6e, bf16
BENCH = os.environ.get("BENCH", "24x1024x16:32:none,24x1536x12:16:dots,20x2048x16:8:full")
SPLASH_BLOCK = int(os.environ.get("SPLASH_BLOCK", 512))
SPLASH_FUSED = os.environ.get("SPLASH_FUSED", "0") == "1"
# Leviers de vitesse (voir en tête du fichier) : chaque défaut reproduit le calcul habituel à l'identique.
SPLASH_BLOCK_Q = int(os.environ.get("SPLASH_BLOCK_Q", 0))  # 0 : SPLASH_BLOCK
SPLASH_BLOCK_KV = int(os.environ.get("SPLASH_BLOCK_KV", 0))
SPLASH_BLOCK_KV_COMPUTE = int(os.environ.get("SPLASH_BLOCK_KV_COMPUTE", 0))  # 0 : SPLASH_BLOCK_KV
SPLASH_BLOCK_Q_BWD = int(os.environ.get("SPLASH_BLOCK_Q_BWD", 0))  # 0 : SPLASH_BLOCK_Q
SPLASH_BLOCK_KV_BWD = int(os.environ.get("SPLASH_BLOCK_KV_BWD", 0))  # 0 : SPLASH_BLOCK_KV
SPLASH_LAYOUT = os.environ.get("SPLASH_LAYOUT", "head,head,head")  # q, k, v : head (D mineure) ou seq (T mineure)
EMBED = os.environ.get("EMBED", "gather")
LOSS_PICK = os.environ.get("LOSS_PICK", "gather")
HEAD_CHUNKS = int(os.environ.get("HEAD_CHUNKS", 1))
CHEAP_REMAT = os.environ.get("CHEAP_REMAT", "0") == "1"
ACCUM_DTYPE = {"bf16": "bfloat16", "f32": "float32"}.get(os.environ.get("ACCUM_DTYPE", "float32"),
                                                        os.environ.get("ACCUM_DTYPE", "float32"))
LAYER_UNROLL = True if os.environ.get("LAYER_UNROLL") == "full" else int(os.environ.get("LAYER_UNROLL", 1))
MICRO_UNROLL = True if os.environ.get("MICRO_UNROLL") == "full" else int(os.environ.get("MICRO_UNROLL", 1))
# Le laboratoire (mode bench) : trace, estimation de XLA, morceaux du pas chronométrés à part.
PROFILE = os.environ.get("PROFILE", "")
PROFILE_STEPS = int(os.environ.get("PROFILE_STEPS", 3))
BENCH_COST = os.environ.get("BENCH_COST", "0") == "1" or bool(PROFILE)
BENCH_PARTS = os.environ.get("BENCH_PARTS", "0") == "1"
BENCH_STEPS = max(1, int(os.environ.get("BENCH_STEPS", 10)))
BENCH_WARMUP = max(1, int(os.environ.get("BENCH_WARMUP", 3)))
SEED = 1337
SOFTCAP = 30.0
EPS = float(np.finfo(np.float32).eps)  # celui de F.rms_norm(x.float()) quand eps=None

PRETRAIN = dict(muon=True, normuon=True, muon_lr=0.05, embed_lr=0.6, head_lr=0.008, scalar_lr=0.04, warmup=0.01,
                cooldown=0.4, dropout=0.0)
# Affinage : AdamW sur tout, montée sur 3 % puis cosinus jusqu'à lr / 10, gradients bornés à 1 (train_gpt2.py).
FINETUNE = dict(lr=FT_LR, min_lr=FT_LR / 10, warmup=0.03, weight_decay=0.1, dropout=FT_DROPOUT, replay=0.25,
                remat=FT_REMAT)
# Ou NorMuon, comme le pré-entraînement : ses learning rates × FT_SCALE, montée puis descente jusqu'à 0.
FINETUNE_MODERN = dict(muon=True, normuon=True, warmup=0.05, cooldown=0.95, dropout=FT_DROPOUT, replay=0.25,
                       remat=FT_REMAT, **{k: PRETRAIN[k] * FT_SCALE for k in ("muon_lr", "embed_lr", "head_lr", "scalar_lr")})

BLOCK_KEYS = ("qkv", "attn_proj", "fc", "mlp_proj")  # les matrices des blocs, empilées (couche, sorties, entrées)
ADAM_KEYS = ("embed", "lm_head", "skip_weights")  # pré-entraînement : Adam pour le reste
TORCH_NAMES = {"qkv": "attn.qkv.weight", "attn_proj": "attn.proj.weight", "fc": "mlp.fc.weight",
               "mlp_proj": "mlp.proj.weight"}
F32 = jnp.float32


def log(*args) -> None:
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def find_input(name: str) -> Path | None:
    """Un fichier d'une session précédente ou de la préparation (en entrée) : le dernier dans l'ordre des noms."""
    dirs = ["/kaggle/input"] + ([os.environ["RESUME_DIR"]] if os.environ.get("RESUME_DIR") else [])
    found = sorted(f for d in dirs for f in glob.glob(f"{d}/**/{name}", recursive=True))
    return Path(found[-1]) if found else None


def active_levers(everything: bool = False) -> dict:
    """Les leviers de vitesse qui ne sont pas à leur valeur par défaut (pour le journal et le JSON de l'essai).
    SPLASH_BLOCK et SPLASH_FUSED, plus anciens, n'y sont ajoutés que si un nouveau levier est actif (ou avec
    `everything`, pour nommer le dossier d'une trace) : avec les seuls réglages d'avant, journal et JSON restent
    ceux d'avant."""
    levers = {"EMBED": (EMBED, "gather"), "LOSS_PICK": (LOSS_PICK, "gather"), "HEAD_CHUNKS": (HEAD_CHUNKS, 1),
              "CHEAP_REMAT": (CHEAP_REMAT, False), "ACCUM_DTYPE": (ACCUM_DTYPE, "float32"),
              "LAYER_UNROLL": (os.environ.get("LAYER_UNROLL", "1"), "1"),
              "MICRO_UNROLL": (os.environ.get("MICRO_UNROLL", "1"), "1"),
              "SPLASH_BLOCK_Q": (SPLASH_BLOCK_Q, 0), "SPLASH_BLOCK_KV": (SPLASH_BLOCK_KV, 0),
              "SPLASH_BLOCK_KV_COMPUTE": (SPLASH_BLOCK_KV_COMPUTE, 0), "SPLASH_BLOCK_Q_BWD": (SPLASH_BLOCK_Q_BWD, 0),
              "SPLASH_BLOCK_KV_BWD": (SPLASH_BLOCK_KV_BWD, 0), "SPLASH_LAYOUT": (SPLASH_LAYOUT, "head,head,head")}
    active = {name: value for name, (value, default) in levers.items() if value != default}
    if active or everything:
        older = {"SPLASH_BLOCK": (SPLASH_BLOCK, 512), "SPLASH_FUSED": (SPLASH_FUSED, False)}
        active.update({name: value for name, (value, default) in older.items() if value != default})
    return active


# --- Le modèle : gpt.ModernGPT, en fonctions pures ---------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Spec:
    """Ce qui fixe le calcul (et donc la compilation) : dimensions, précision, attention, recalcul, dropout."""

    vocab_size: int
    n_layer: int
    n_embd: int
    n_head: int
    block: int = BLOCK
    dtype: str = COMPUTE_DTYPE  # bfloat16 : poids convertis à l'usage, activations bf16 ; float32 : calcul exact
    attn: str = "xla"
    remat: str = "none"
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def name(self) -> str:
        return f"{self.n_layer}x{self.n_embd}x{self.n_head}"


def init_params(spec: Spec, seed: int = SEED) -> dict:
    """Les poids float32, comme ModernGPT : embedding N(0, 1), qkv et fc uniformes (écart-type 0,5/√entrées),
    projections de sortie et tête à zéro, poids des connexions en U à 1. Disposition de nn.Linear : (sorties, entrées)."""
    V, C, L = spec.vocab_size, spec.n_embd, spec.n_layer

    @jax.jit
    def init(key):
        k_embed, k_qkv, k_fc = jax.random.split(key, 3)
        bound = 3**0.5 * 0.5 * C**-0.5
        return {"embed": jax.random.normal(k_embed, (V, C), F32),
                "qkv": jax.random.uniform(k_qkv, (L, 3 * C, C), F32, -bound, bound),
                "attn_proj": jnp.zeros((L, C, C), F32),
                "fc": jax.random.uniform(k_fc, (L, 4 * C, C), F32, -bound, bound),
                "mlp_proj": jnp.zeros((L, C, 4 * C), F32),
                "skip_weights": jnp.ones((L // 2,), F32),
                "lm_head": jnp.zeros((V, C), F32)}

    return init(jax.random.key(seed))


def count_params(params: dict) -> int:
    return int(sum(np.prod(p.shape) for p in params.values()))


def _precision(dtype):
    """float32 : produits matriciels vraiment en float32 (sur TPU, le défaut serait une passe bf16)."""
    return jax.lax.Precision.HIGHEST if jnp.dtype(dtype) == jnp.float32 else None


def linear(x, w):
    """x @ wᵀ, comme nn.Linear (w : (sorties, entrées))."""
    return jnp.einsum("...i,oi->...o", x, w, precision=_precision(x.dtype))


def rms_norm(x):
    """F.rms_norm(x.float()) sans paramètre, rendu dans le type de x."""
    xf = x.astype(F32)
    return (xf * jax.lax.rsqrt(jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + EPS)).astype(x.dtype)


@functools.lru_cache(maxsize=None)
def rope_tables(T: int, head_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """cos et sin des angles t·θᵢ de RoPE, θᵢ = 10000^(−i / (D/2)), en float32 comme gpt.Rotary."""
    half = head_dim // 2
    theta = np.float32(10_000.0) ** (-np.arange(0, half, dtype=np.float32) / np.float32(half))
    angles = np.outer(np.arange(T, dtype=np.float32), theta)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def rotary(x, cos, sin):
    """RoPE sur x (B, T, H, D) : chaque paire (x1, x2) des deux moitiés tourne de t·θᵢ."""
    c, s = jnp.asarray(cos, x.dtype)[:, None, :], jnp.asarray(sin, x.dtype)[:, None, :]
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


def dropout(x, rate: float, key):
    if not rate or key is None:
        return x
    keep = jax.random.bernoulli(key, 1.0 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), 0).astype(x.dtype)


def attention_explicit(q, k, v, rate: float = 0.0, key=None):
    """softmax(qkᵀ/√d)·v causal, écrit à la main (scores et softmax en float32), avec le dropout de l'affinage
    sur les poids d'attention, comme F.scaled_dot_product_attention(dropout_p=…)."""
    T, D = q.shape[1], q.shape[-1]
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=F32, precision=_precision(q.dtype)) * D**-0.5
    scores = jnp.where(jnp.tril(jnp.ones((T, T), bool)), scores, -jnp.inf)  # le futur est interdit
    probs = dropout(jax.nn.softmax(scores, axis=-1), rate, key)
    return jnp.einsum("bhqk,bkhd->bqhd", probs.astype(v.dtype), v, precision=_precision(v.dtype))


def splash_kernel(T: int, n_head: int, block: int, fused: bool, residual_name: str | None = None):
    """Le noyau Pallas « splash attention » du TPU, masque causal : les blocs entièrement dans le futur sont sautés.
    Construit à chaque traçage, sans cache : ses tableaux de masque, créés pendant un traçage, n'en sortent pas.

    Blocs de `block` tokens partout, sauf SPLASH_BLOCK_Q, SPLASH_BLOCK_KV, SPLASH_BLOCK_KV_COMPUTE (aller) et
    SPLASH_BLOCK_Q_BWD, SPLASH_BLOCK_KV_BWD (retour : dk, dv, et dq s'il n'est pas fusionné) ; des blocs plus petits
    que T sautent la moitié future du masque causal. `residual_name` nomme la sortie et le logsumexp du noyau, que
    REMAT=dots_attn garde pour le backward. SPLASH_LAYOUT=seq pour un tableau : make_splash_mha le transpose
    (une copie par XLA avant chaque noyau), puis le noyau lui-même transpose chaque bloc de q (aller, dq) ou de k
    et v (dk, dv) : le backward, deux fois l'aller, peut y perdre ce que l'aller gagne."""
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as sm

    b = min(block, T)
    bq, bkv = (min(size, T) if size else b for size in (SPLASH_BLOCK_Q, SPLASH_BLOCK_KV))
    bkv_compute = min(SPLASH_BLOCK_KV_COMPUTE, bkv) if SPLASH_BLOCK_KV_COMPUTE else bkv
    bq_bwd, bkv_bwd = (min(size, T) if size else fwd for size, fwd in ((SPLASH_BLOCK_Q_BWD, bq), (SPLASH_BLOCK_KV_BWD, bkv)))
    layouts = {}
    if SPLASH_LAYOUT != "head,head,head":  # la disposition que le noyau impose à q, k, v (XLA transpose avant lui)
        kinds = {"head": sk.QKVLayout.HEAD_DIM_MINOR, "seq": sk.QKVLayout.SEQ_MINOR}
        names = [s.strip() for s in SPLASH_LAYOUT.split(",")]
        if len(names) != 3 or not set(names) <= set(kinds):
            raise ValueError(f"SPLASH_LAYOUT : trois valeurs head ou seq (q, k, v), pas {SPLASH_LAYOUT!r}")
        layouts = {f"{part}_layout": kinds[name] for part, name in zip("qkv", names)}
    sizes = sk.BlockSizes(block_q=bq, block_kv=bkv, block_kv_compute=bkv_compute, block_q_dkv=bq_bwd,
                          block_kv_dkv=bkv_bwd, block_kv_dkv_compute=bkv_bwd, use_fused_bwd_kernel=fused,
                          **({} if fused else {"block_q_dq": bq_bwd, "block_kv_dq": bkv_bwd}), **layouts)
    mask = sm.MultiHeadMask(tuple(sm.CausalMask((T, T)) for _ in range(n_head)))
    return sk.make_splash_mha(mask, head_shards=1, q_seq_shards=1, block_sizes=sizes,
                              **({"residual_checkpoint_name": residual_name} if residual_name else {}))


def attention(q, k, v, spec: Spec, key=None, rate: float = 0.0):
    """L'attention causale de q, k, v (B, T, H, D), par le chemin de `spec.attn` ; dropout : le chemin explicite."""
    scale = spec.head_dim**-0.5
    if (rate and key is not None) or spec.attn == "explicit":
        return attention_explicit(q, k, v, rate, key)
    if spec.attn == "xla":
        return jax.nn.dot_product_attention(q, k, v, scale=scale, is_causal=True, implementation="xla")
    T = q.shape[1]
    qt, kt, vt = (jnp.swapaxes(a, 1, 2) for a in (q, k, v))  # (B, H, T, D) pour les noyaux Pallas
    if spec.attn == "splash":
        residual = "attn" if spec.remat == "dots_attn" else None  # REMAT=dots_attn : sortie et logsumexp gardés
        out = jax.vmap(splash_kernel(T, spec.n_head, SPLASH_BLOCK, SPLASH_FUSED, residual))(qt * jnp.asarray(scale, q.dtype), kt, vt)
    elif spec.attn == "flash":
        from jax.experimental.pallas.ops.tpu.flash_attention import BlockSizes, flash_attention

        b = min(SPLASH_BLOCK, T)
        sizes = BlockSizes(block_q=b, block_k_major=b, block_k=b, block_b=1, block_q_major_dkv=b, block_k_major_dkv=b,
                           block_k_dkv=b, block_q_dkv=b, block_k_major_dq=b, block_k_dq=b, block_q_dq=b)
        out = flash_attention(qt, kt, vt, causal=True, sm_scale=scale, block_sizes=sizes)
    else:
        raise ValueError(f"ATTN inconnu : {spec.attn!r} (auto, splash, flash, xla ou explicit)")
    return jnp.swapaxes(out, 1, 2)


def relu2(u):
    """ReLU² : relu(u)·relu(u)."""
    h = jax.nn.relu(u)
    return h * h


def mlp_out(u, w):
    """La fin du MLP, de u (la sortie de fc) à sa sortie : relu(u)² @ mlp_projᵀ."""
    return linear(relu2(u), w)


# CHEAP_REMAT=1 : les mêmes calculs, refaits au backward à partir de leurs entrées bf16 au lieu de garder leurs
# résidus : la copie float32 de x pour rms_norm ; pour la fin du MLP, le masque relu(u) > 0, relu(u) et relu(u)²
# (B, T, 4C), que le gradient de mlp_proj relit : seule u est gardée. Rien que de l'élément par élément à refaire :
# le produit par mlp_proj n'est pas refait, sa sortie ne sert pas au backward. Seul, ReLU² sous jax.checkpoint
# n'économiserait que le masque (u remplacerait relu(u), de même taille, et relu(u)² resterait gardé).
rms_norm_remat = jax.checkpoint(rms_norm, prevent_cse=False)
mlp_out_remat = jax.checkpoint(mlp_out, prevent_cse=False)


def block_forward(x, w: dict, spec: Spec, key, rate: float):
    """Un ModernBlock : x + attn(rms_norm(x)), puis x + mlp(rms_norm(x)) ; dropout aux trois mêmes endroits."""
    B, T, C = x.shape
    H, D = spec.n_head, spec.head_dim
    keys = [jax.random.fold_in(key, i) for i in range(3)] if (rate and key is not None) else [None] * 3
    norm, mlp = (rms_norm_remat, mlp_out_remat) if CHEAP_REMAT else (rms_norm, mlp_out)
    cos, sin = rope_tables(T, D)
    qkv = linear(norm(x), w["qkv"]).reshape(B, T, 3, H, D)  # le view(B, T, 3, H, D) de PyTorch
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    q, k = rotary(norm(q), cos, sin), rotary(norm(k), cos, sin)  # QK-norm puis RoPE
    y = attention(q, k, v, spec, keys[0], rate)
    if spec.remat == "dots_attn" and spec.attn != "splash":  # splash nomme lui-même sa sortie
        y = checkpoint_name(y, "attn")
    y = y.reshape(B, T, C)
    x = x + dropout(linear(y, w["attn_proj"]), rate, keys[1])
    return x + dropout(mlp(linear(norm(x), w["fc"]), w["mlp_proj"]), rate, keys[2])  # ReLU²


# dots_attn : les résidus nommés « attn » en plus de dots ; avec splash, la sortie et le logsumexp du noyau (son
# backward ne relance pas l'aller) ; avec flash, xla ou explicit, la seule sortie y (le noyau flash est relancé quand
# même, xla et explicit recalculent leurs probabilités) : plus de mémoire pour presque rien.
REMAT_POLICIES = {"dots": jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
                  "dots_attn": jax.checkpoint_policies.save_from_both_policies(
                      jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
                      jax.checkpoint_policies.save_only_these_names("attn")),
                  "full": jax.checkpoint_policies.nothing_saveable}


def compute_params(params: dict, spec: Spec) -> dict:
    """Les poids tels que le calcul les lit : matrices converties dans le type de calcul (une fois par pas, pas à
    chaque micro-lot), blocs déjà coupés en première moitié, bloc du milieu (nombre impair) et seconde moitié.
    L'embedding et les poids des connexions en U restent en float32."""
    dt, L = jnp.dtype(spec.dtype), spec.n_layer
    half = L // 2
    cp = {"embed": params["embed"], "skip_weights": params["skip_weights"], "lm_head": params["lm_head"].astype(dt),
          "first": {k: params[k][:half].astype(dt) for k in BLOCK_KEYS},
          "second": {k: params[k][L - half:].astype(dt) for k in BLOCK_KEYS}}
    if L % 2:
        cp["middle"] = {k: params[k][half].astype(dt) for k in BLOCK_KEYS}
    return cp


def merge_grads(g: dict, spec: Spec) -> dict:
    """Les gradients de compute_params ramenés à la disposition des paramètres (blocs empilés), en float32."""
    parts = [g["first"]] + ([{k: v[None] for k, v in g["middle"].items()}] if spec.n_layer % 2 else []) + [g["second"]]
    out = {k: jnp.concatenate([p[k].astype(F32) for p in parts]) for k in BLOCK_KEYS}
    return {"embed": g["embed"].astype(F32), "skip_weights": g["skip_weights"].astype(F32),
            "lm_head": g["lm_head"].astype(F32), **out}


def embed_tokens(table, idx, spec: Spec):
    """Les lignes float32 de l'embedding pour les tokens `idx` (B, T). EMBED=gather : une lecture indexée, dont le
    backward est un scatter-add float32 dans (V, C). EMBED=onehot : one_hot(idx) @ embedding sur le MXU dans le type
    de calcul (l'« iota embed » de MaxText) ; le one-hot, recalculé au backward (jax.checkpoint), n'est jamais gardé
    en mémoire. Le gradient est un produit matriciel lui aussi, en bfloat16 : la contribution float32 de chaque token
    y est arrondie en bf16 (précision DEFAULT : une passe bf16 du MXU), sommée en float32, et le total est arrondi en
    bf16 (le type de la table convertie) avant de revenir en float32. COMPUTE_DTYPE=float32 (précision HIGHEST) :
    l'aller est exact, le gradient égal à celui du scatter à l'ordre des sommes près."""
    if EMBED == "gather":
        return table[idx]
    if EMBED != "onehot":
        raise ValueError(f"EMBED inconnu : {EMBED!r} (gather ou onehot)")
    dt = jnp.dtype(spec.dtype)

    def onehot(table, idx):
        hot = jax.nn.one_hot(idx, table.shape[0], dtype=dt)
        return jnp.einsum("btv,vc->btc", hot, table.astype(dt), preferred_element_type=F32, precision=_precision(dt))

    return jax.checkpoint(onehot, prevent_cse=False)(table, idx)


def trunk(cp: dict, idx, spec: Spec, key=None):
    """Le flux résiduel (B, T, C) à la sortie des blocs, avant la norme finale : embedding normalisé, blocs en U.

    Première moitié des blocs : un lax.scan qui empile leurs sorties ; seconde moitié : un lax.scan qui les
    reprend dans l'ordre inverse (la pile de PyTorch), chacune multipliée par son poids appris. LAYER_UNROLL : tours
    déroulés de ces deux boucles."""
    rate = spec.dropout if key is not None else 0.0
    dt, L = jnp.dtype(spec.dtype), spec.n_layer
    half = L // 2

    def blk(x, w, i, key):
        k = jax.random.fold_in(key, i) if rate else None
        return block_forward(x, w, spec, k, rate)

    if spec.remat != "none":
        blk = jax.checkpoint(blk, policy=REMAT_POLICIES[spec.remat], prevent_cse=False)
    x = rms_norm(embed_tokens(cp["embed"], idx, spec)).astype(dt)
    if half:
        def first(x, xs):
            w, i = xs
            x = blk(x, w, i, key)
            return x, x

        x, skips = jax.lax.scan(first, x, (cp["first"], jnp.arange(half)), unroll=LAYER_UNROLL)
    if L % 2:  # le bloc du milieu ne reçoit rien
        x = blk(x, cp["middle"], jnp.int32(half), key)
    if half:
        def second(x, xs):
            w, weight, skip, i = xs
            x = (x.astype(F32) + weight * skip.astype(F32)).astype(dt)  # la sortie du bloc symétrique
            return blk(x, w, i, key), None

        x, _ = jax.lax.scan(second, x, (cp["second"], cp["skip_weights"], skips[::-1], jnp.arange(L - half, L)),
                            unroll=LAYER_UNROLL)
    return x


def head(lm_head, x, spec: Spec):
    """Les logits float32 de la tête : rms_norm(x) @ lm_headᵀ, plafonnés à ±30."""
    logits = jnp.einsum("btc,vc->btv", rms_norm(x), lm_head, preferred_element_type=F32,
                        precision=_precision(jnp.dtype(spec.dtype)))
    return SOFTCAP * jnp.tanh(logits / SOFTCAP)


def forward(cp: dict, idx, spec: Spec, key=None):
    """Les logits float32 (B, T, V) de ModernGPT : embedding normalisé, blocs en U, tête plafonnée à ±30."""
    return head(cp["lm_head"], trunk(cp, idx, spec, key), spec)


def token_losses(logits, y):
    """Entropie croisée de chaque position (float32). LOSS_PICK=onehot : le logit de la cible pris par un produit
    avec one_hot(y), qu'XLA fusionne, au lieu d'une lecture indexée dont le backward est un scatter dans (B, T, V) ;
    un seul terme non nul dans la somme : le même résultat, au bit près."""
    lse = jax.nn.logsumexp(logits, axis=-1)
    if LOSS_PICK == "onehot":
        return lse - jnp.sum(logits * jax.nn.one_hot(y, logits.shape[-1], dtype=logits.dtype), axis=-1)
    if LOSS_PICK != "gather":
        raise ValueError(f"LOSS_PICK inconnu : {LOSS_PICK!r} (gather ou onehot)")
    return lse - jnp.take_along_axis(logits, y[..., None], axis=-1)[..., 0]


def chunked_loss(cp: dict, x, y, spec: Spec):
    """HEAD_CHUNKS > 1 : la loss moyenne de la tête, calculée par morceaux de la séquence l'un après l'autre
    (lax.map), chacun recalculé au backward (jax.checkpoint) : les logits float32 n'existent jamais qu'un morceau
    à la fois, (B, T / n, V) au lieu de (B, T, V). Le gradient de la tête s'additionne en float32 d'un morceau à
    l'autre ; la loss de chaque position est celle du calcul d'un seul tenant."""
    B, T, C = x.shape
    n = HEAD_CHUNKS
    if T % n:
        raise ValueError(f"HEAD_CHUNKS ({n}) doit diviser la longueur des séquences ({T})")
    dt = cp["lm_head"].dtype
    head32 = cp["lm_head"].astype(F32)

    def piece(w, xc, yc):
        return token_losses(head(w.astype(dt), xc, spec), yc)

    piece = jax.checkpoint(piece, prevent_cse=False)
    xs = jnp.swapaxes(x.reshape(B, n, T // n, C), 0, 1)  # (n, B, T / n, C)
    ys = jnp.swapaxes(y.reshape(B, n, T // n), 0, 1)
    losses = jax.lax.map(lambda xy: piece(head32, *xy), (xs, ys))  # (n, B, T / n)
    return jnp.mean(jnp.swapaxes(losses, 0, 1).reshape(B, T))


def loss_fn(cp: dict, tokens, spec: Spec, key=None):
    """Loss moyenne d'un lot de séquences (B, BLOCK + 1) : entrées tokens[:, :-1], cibles tokens[:, 1:]."""
    if HEAD_CHUNKS > 1:
        return chunked_loss(cp, trunk(cp, tokens[:, :-1], spec, key), tokens[:, 1:], spec)
    return jnp.mean(token_losses(forward(cp, tokens[:, :-1], spec, key), tokens[:, 1:]))


# --- Optimiseurs : gpt.Muon(normuon=True), gpt.AdamW -------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Opt:
    """normuon : NorMuon pour les matrices des blocs, Adam pour embedding, tête et poids des connexions en U ;
    adamw : AdamW sur tout, weight decay sur les matrices, gradients bornés (norme globale ≤ clip)."""

    kind: str
    muon_lr: float = 0.05
    embed_lr: float = 0.6
    head_lr: float = 0.008
    scalar_lr: float = 0.04
    lr: float = 0.0
    weight_decay: float = 0.0
    clip: float = 0.0
    betas: tuple[float, float] = (0.8, 0.95)
    eps: float = 1e-8
    momentum: float = 0.95
    beta2: float = 0.95
    ns_dtype: str = NS_DTYPE


def make_opt(cfg: dict) -> Opt:
    if cfg.get("muon"):
        return Opt("normuon", betas=(0.8, 0.95), **{k: cfg[k] * LR_SCALE for k in ("muon_lr", "embed_lr", "head_lr", "scalar_lr")})
    return Opt("adamw", lr=cfg["lr"], weight_decay=cfg["weight_decay"], clip=1.0, betas=(0.9, 0.95))


def lr_multiplier(cfg: dict, progress: float) -> float:
    """Le multiplicateur des learning rates : warmup-stable-decay (NorMuon), ou montée puis cosinus (AdamW)."""
    warm = cfg["warmup"]
    if cfg.get("muon"):
        return gpt.wsd_schedule(progress, warm, cfg["cooldown"])
    if progress < warm:
        return min(1.0, (progress + 1e-4) / warm)
    cos = 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - warm) / (1 - warm))))
    return (cfg["min_lr"] + cos * (cfg["lr"] - cfg["min_lr"])) / cfg["lr"]


def init_opt_state(params: dict, opt: Opt) -> dict:
    zeros = lambda a: jnp.zeros(a.shape, F32)  # noqa: E731
    if opt.kind == "normuon":
        return {"muon_buf": {k: zeros(params[k]) for k in BLOCK_KEYS},
                "muon_v": {k: jnp.zeros(params[k].shape[:2] + (1,), F32) for k in BLOCK_KEYS},  # une valeur par neurone
                "adam_m": {k: zeros(params[k]) for k in ADAM_KEYS}, "adam_v": {k: zeros(params[k]) for k in ADAM_KEYS},
                "t": jnp.zeros((), F32)}
    return {"m": {k: zeros(p) for k, p in params.items()}, "v": {k: zeros(p) for k, p in params.items()},
            "t": jnp.zeros((), F32)}


def lerp(start, end, weight: float):
    """torch.lerp, au même arrondi près : start + w·(end − start) si w < 0,5, sinon end − (end − start)·(1 − w)."""
    if weight < 0.5:
        return start + weight * (end - start)
    return end - (end - start) * (1 - weight)


def newton_schulz(G, dtype: str):
    """gpt.newton_schulz : 5 itérations quintiques, sur la matrice transposée si elle a plus de lignes que de
    colonnes, normalisée par sa norme de Frobenius. bfloat16 sur TPU (comme modded-nanoGPT), float32 pour les tests."""
    a, b, c = 3.4445, -4.7750, 2.0315
    dt, prec = jnp.dtype(dtype), _precision(dtype)
    X = G.astype(F32)
    transposed = X.shape[-2] > X.shape[-1]
    if transposed:
        X = jnp.swapaxes(X, -1, -2)
    X = (X / (jnp.sqrt(jnp.sum(X * X)) + 1e-7)).astype(dt)  # valeurs singulières ≤ 1
    mm = lambda u, w: jnp.matmul(u.astype(dt), w.astype(dt), precision=prec, preferred_element_type=F32)  # noqa: E731
    for _ in range(5):
        A = mm(X, jnp.swapaxes(X, -1, -2))
        B = b * A + mm(c * A, A)
        X = (a * X.astype(F32) + mm(B, X)).astype(dt)
    X = X.astype(F32)
    return jnp.swapaxes(X, -1, -2) if transposed else X


def normuon_matrix(p, g, buf, v, lr, opt: Opt):
    """gpt.Muon(normuon=True).step pour une matrice : Nesterov, orthogonalisation, une échelle par ligne (β₂),
    norme de Muon rétablie, pas × √max(1, lignes/colonnes)."""
    buf = lerp(buf, g, 1 - opt.momentum)
    update = newton_schulz(lerp(g, buf, opt.momentum), opt.ns_dtype)
    v = lerp(v, jnp.mean(jnp.square(update), axis=-1, keepdims=True), 1 - opt.beta2)
    scaled = update / (jnp.sqrt(v) + 1e-8)
    update = scaled * (jnp.sqrt(jnp.sum(update * update)) / (jnp.sqrt(jnp.sum(scaled * scaled)) + 1e-8))
    scale = max(1.0, p.shape[-2] / p.shape[-1]) ** 0.5
    return p - update * (lr * scale), buf, v


def adam_update(p, g, m, v, lr, weight_decay: float, b1: float, b2: float, c1, c2, eps: float):
    """gpt.AdamW.step pour un paramètre : weight decay découplé, moments, correction de biais."""
    if weight_decay:
        p = p * (1 - lr * weight_decay)
    m = lerp(m, g, 1 - b1)
    v = v * b2 + (1 - b2) * g * g
    return p - lr * (m / c1) / (jnp.sqrt(v / c2) + eps), m, v


def normuon_step(params: dict, grads: dict, state: dict, mult, opt: Opt) -> tuple[dict, dict]:
    """gpt.modern_optimizers(normuon=True) : NorMuon couche par couche (vmap sur l'axe des couches), Adam pour le
    reste, avec un seul compteur de pas."""
    params, state = dict(params), {k: (dict(v) if isinstance(v, dict) else v) for k, v in state.items()}
    lr = mult * opt.muon_lr
    for k in BLOCK_KEYS:
        step = jax.vmap(lambda p, g, b, v: normuon_matrix(p, g, b, v, lr, opt))
        params[k], state["muon_buf"][k], state["muon_v"][k] = step(params[k], grads[k], state["muon_buf"][k], state["muon_v"][k])
    t = state["t"] + 1
    b1, b2 = opt.betas
    c1, c2 = 1 - b1**t, 1 - b2**t
    for k, group_lr in (("embed", opt.embed_lr), ("lm_head", opt.head_lr), ("skip_weights", opt.scalar_lr)):
        params[k], state["adam_m"][k], state["adam_v"][k] = adam_update(
            params[k], grads[k], state["adam_m"][k], state["adam_v"][k], mult * group_lr, 0.0, b1, b2, c1, c2, opt.eps)
    state["t"] = t
    return params, state


def global_norm(grads: dict):
    return jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in grads.values()))


def adamw_step(params: dict, grads: dict, state: dict, mult, opt: Opt) -> tuple[dict, dict]:
    """L'affinage : clip_grad_norm_ (norme globale), puis gpt.AdamW sur tous les paramètres, weight decay sur les
    matrices (embedding, blocs, tête), pas sur les poids des connexions en U."""
    if opt.clip:
        coef = jnp.minimum(1.0, opt.clip / (global_norm(grads) + 1e-6))
        grads = {k: g * coef for k, g in grads.items()}
    t = state["t"] + 1
    b1, b2 = opt.betas
    c1, c2 = 1 - b1**t, 1 - b2**t
    lr = mult * opt.lr
    new_p, m, v = {}, {}, {}
    for k in params:
        wd = opt.weight_decay if k != "skip_weights" else 0.0
        new_p[k], m[k], v[k] = adam_update(params[k], grads[k], state["m"][k], state["v"][k], lr, wd, b1, b2, c1, c2, opt.eps)
    return new_p, {"m": m, "v": v, "t": t}


OPTIMIZERS = {"normuon": normuon_step, "adamw": adamw_step}


# --- Le pas d'entraînement : une seule fonction compilée -------------------------------------------------


def accumulate_grads(spec: Spec, accum: int):
    """(poids, lots (accum, MB, BLOCK + 1), pas) → (gradients moyens à la disposition des paramètres, float32 ;
    somme des loss des micro-lots).

    Les micro-lots défilent dans un lax.scan (MICRO_UNROLL tours déroulés) qui additionne leurs gradients en
    float32 (la moyenne est celle de PyTorch, qui fait backward(loss / accum) à chaque micro-lot). Avec
    ACCUM_DTYPE=bfloat16, la somme des gradients bf16 (matrices des blocs, tête : ≈ 310 M valeurs en medium) reste
    en bfloat16 : ≈ 0,6 Go de moins en mémoire (0,62 Go au lieu de 1,24) et ≈ 1,2 Go de moins relus et réécrits par
    micro-lot, au prix d'un arrondi bf16 à chaque ajout ; embedding et poids des connexions en U restent en float32."""
    if ACCUM_DTYPE not in ("float32", "bfloat16"):
        raise ValueError(f"ACCUM_DTYPE inconnu : {ACCUM_DTYPE!r} (float32 ou bfloat16)")

    def carry_dtype(a):
        return F32 if ACCUM_DTYPE == "float32" else jnp.promote_types(a.dtype, jnp.dtype(ACCUM_DTYPE))

    def grads_and_total(params, batch, step):
        cp = compute_params(params, spec)
        base = jax.random.fold_in(jax.random.key(SEED), step)

        def micro(carry, xs):
            grads, total = carry
            tokens, i = xs
            key = jax.random.fold_in(base, i) if spec.dropout else None
            loss, g = jax.value_and_grad(lambda c: loss_fn(c, tokens, spec, key))(cp)
            return (jax.tree.map(lambda a, b: a + b.astype(a.dtype), grads, g), total + loss), None

        zeros = jax.tree.map(lambda a: jnp.zeros(a.shape, carry_dtype(a)), cp)
        (grads, total), _ = jax.lax.scan(micro, (zeros, jnp.zeros((), F32)), (batch, jnp.arange(accum)),
                                         unroll=MICRO_UNROLL)
        if ACCUM_DTYPE == "float32":
            grads = merge_grads(jax.tree.map(lambda g: g / accum, grads), spec)
        else:
            grads = merge_grads(jax.tree.map(lambda g: g.astype(F32) / accum, grads), spec)
        return grads, total

    return grads_and_total


def make_train_step(spec: Spec, opt: Opt, accum: int):
    """(poids, état, lots (accum, MB, BLOCK + 1), multiplicateur, pas) → (poids, état, loss moyenne).

    Les gradients des micro-lots s'additionnent (accumulate_grads), puis l'optimiseur s'applique. Poids et état
    sont donnés (donate) : XLA les met à jour sur place. Le multiplicateur est une donnée, pas une constante :
    il change à chaque pas sans recompiler."""
    grads_and_total = accumulate_grads(spec, accum)

    def train_step(params, opt_state, batch, mult, step):
        grads, total = grads_and_total(params, batch, step)
        params, opt_state = OPTIMIZERS[opt.kind](params, grads, opt_state, jnp.asarray(mult, F32), opt)
        return params, opt_state, total / accum

    return jax.jit(train_step, donate_argnums=(0, 1))


def make_step_parts(spec: Spec, opt: Opt, accum: int) -> dict:
    """Les morceaux du pas pour BENCH_PARTS (fonctions à compiler chacune à part), avec les mêmes micro-lots en
    lax.scan que le pas complet : forward (forward et loss seuls), forward_backward (les gradients moyens, sans
    optimiseur) et optimizer (l'optimiseur seul, sur des gradients donnés)."""
    grads_and_total = accumulate_grads(spec, accum)

    def forward_only(params, batch, step):
        cp = compute_params(params, spec)
        base = jax.random.fold_in(jax.random.key(SEED), step)

        def micro(total, xs):
            tokens, i = xs
            key = jax.random.fold_in(base, i) if spec.dropout else None
            return total + loss_fn(cp, tokens, spec, key), None

        total, _ = jax.lax.scan(micro, jnp.zeros((), F32), (batch, jnp.arange(accum)), unroll=MICRO_UNROLL)
        return total / accum

    def forward_backward(params, batch, step):
        grads, total = grads_and_total(params, batch, step)
        return grads, total / accum

    def update(params, grads, opt_state, mult):
        return OPTIMIZERS[opt.kind](params, grads, opt_state, jnp.asarray(mult, F32), opt)

    return {"forward": forward_only, "forward_backward": forward_backward, "optimizer": update}


def make_eval_fn(spec: Spec):
    """Loss moyenne sur des lots (n, MB, BLOCK + 1) déjà sur le TPU, sans dropout, en un seul appel."""
    spec = dataclasses.replace(spec, dropout=0.0)

    @jax.jit
    def evaluate(params, batches):
        cp = compute_params(params, spec)
        return jnp.mean(jax.lax.map(lambda b: loss_fn(cp, b, spec), batches))

    return evaluate


def flops_per_token(spec: Spec) -> float:
    """6 N (matrices des blocs) + 6 V·C (tête) + 12 L·T·C (attention), la formule de PaLM et nanoGPT."""
    C, L = spec.n_embd, spec.n_layer
    return 6 * 12 * L * C * C + 6 * spec.vocab_size * C + 12 * L * spec.block * C


def check_attention(kind: str, n_head: int, head_dim: int, T: int) -> float:
    """Écart relatif (sorties et gradients, bf16) entre un chemin d'attention et le calcul explicite."""
    rng = np.random.default_rng(0)
    q, k, v = (jnp.asarray(rng.standard_normal((2, T, n_head, head_dim)), jnp.bfloat16) for _ in range(3))
    w = jnp.asarray(rng.standard_normal((2, T, n_head, head_dim)), F32)

    def run(path):
        spec = Spec(vocab_size=1, n_layer=1, n_embd=n_head * head_dim, n_head=n_head, block=T, dtype="bfloat16", attn=path)
        f = lambda q, k, v: jnp.sum(attention(q, k, v, spec).astype(F32) * w)  # noqa: E731
        out = jax.jit(lambda q, k, v: attention(q, k, v, spec))(q, k, v).astype(F32)
        return out, jax.jit(jax.grad(f, argnums=(0, 1, 2)))(q, k, v)

    (out, grads), (ref, ref_grads) = run(kind), run("explicit")
    errors = [float(jnp.max(jnp.abs(out - ref)) / (jnp.max(jnp.abs(ref)) + 1e-6))]
    errors += [float(jnp.max(jnp.abs(a.astype(F32) - b.astype(F32))) / (jnp.max(jnp.abs(b.astype(F32))) + 1e-6))
               for a, b in zip(grads, ref_grads)]
    return max(errors)


def resolve_attention(attn: str, n_head: int, head_dim: int, T: int = BLOCK) -> str:
    """ATTN=auto : splash sur TPU si son autotest passe (sorties et gradients ≈ le calcul explicite), sinon xla."""
    if attn != "auto":
        return attn
    if jax.default_backend() != "tpu":
        return "xla"
    try:
        gap = check_attention("splash", n_head, head_dim, T)
        if gap < 0.05:
            return "splash"
        log(f"splash attention écartée (écart {gap:.3f} avec le calcul explicite) : attention XLA")
    except Exception as problem:  # noyau non compilable sur cette puce, version de JAX…
        log(f"splash attention indisponible ({str(problem)[:200]}) : attention XLA")
    return "xla"


# --- Données ------------------------------------------------------------------------------------------------


def load_data(tokenizer) -> tuple[list[tuple[np.ndarray, float]], np.ndarray, dict, dict]:
    """Le corpus général (sous-titres, web, Wikipédia, pondérés par MIX ou par leur taille), sa validation, et les
    Simpson (texte et tokens)."""
    simpsons = {part: (INPUT / f"simpsons_{part}.txt").read_text(encoding="utf-8") for part in ("train", "val", "test")}
    tokens = {part: np.asarray(tokenizer.encode(text), dtype=np.uint16) for part, text in simpsons.items()}
    if FAKE_PRETRAIN:
        return [(tokens["train"], 1.0)], tokens["val"], tokens, simpsons
    general, vals = [], []
    for name in ("wiki", "web", "subs"):
        path = find_input(f"{name}_train.bin")
        if path is not None and (MIX is None or MIX.get(name, 0) > 0):
            data = np.memmap(path, dtype=np.uint16, mode="r")
            if CORPUS_CAP and MIX:
                data = data[: int(CORPUS_CAP * MIX[name] / sum(MIX.values()))]
            general.append((data, MIX[name] if MIX else float(len(data))))
            vals.append(np.memmap(path.with_name(f"{name}_val.bin"), dtype=np.uint16, mode="r"))
    if not general:
        raise FileNotFoundError("pas de corpus : il faut les *_train.bin de prepare_data.py sous RESUME_DIR")
    log("corpus général : " + "   ".join(f"{len(d) / 1e9:.2f} G tokens" for d, _ in general))
    return general, np.concatenate(vals), tokens, simpsons


def rows_at(data: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Les séquences de BLOCK + 1 tokens qui commencent à `starts` (entrées et cibles décalées d'un cran)."""
    return np.asarray(data[starts[:, None] + np.arange(BLOCK + 1)], dtype=np.int32)


def random_rows(data: np.ndarray, n: int, generator: np.random.Generator) -> np.ndarray:
    """`n` séquences tirées au hasard, comme batch_from de train_tpu.py."""
    return rows_at(data, generator.integers(0, len(data) - BLOCK - 1, size=n))


class Prefetcher:
    """Prépare les lots des pas suivants dans un thread et les envoie au TPU (file de `depth` lots)."""

    def __init__(self, make, depth: int = 3) -> None:
        self.queue, self.stop = queue.Queue(maxsize=depth), threading.Event()
        self.thread = threading.Thread(target=self._loop, args=(make,), daemon=True)
        self.thread.start()

    def _put(self, item) -> None:
        while not self.stop.is_set():
            try:
                self.queue.put(item, timeout=0.5)
                return
            except queue.Full:
                pass

    def _loop(self, make) -> None:
        try:
            while not self.stop.is_set():
                self._put(jax.device_put(make()))
        except BaseException as problem:  # transmis au thread principal
            self._put(problem)

    def next(self):
        item = self.queue.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self) -> None:
        self.stop.set()
        while not self.queue.empty():
            self.queue.get_nowait()
        self.thread.join(timeout=10)


class Writer:
    """Les écritures sur disque (états, modèles), dans l'ordre, dans un thread à part : le TPU n'attend pas le
    disque. Chaque fichier est écrit à côté puis renommé : jamais de fichier à moitié écrit."""

    def __init__(self) -> None:
        self.jobs, self.problem = queue.Queue(), None
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while True:
            job = self.jobs.get()
            try:
                job()
            except BaseException as problem:
                self.problem = problem
            finally:
                self.jobs.task_done()

    def submit(self, job) -> None:
        self.check()
        self.jobs.put(job)

    def wait(self) -> None:
        self.jobs.join()
        self.check()

    def check(self) -> None:
        if self.problem is not None:
            problem, self.problem = self.problem, None
            raise problem


WRITER = Writer()


# --- Fichiers : état d'entraînement, modèles au format de gpt.py, état de train_tpu.py --------------------


def flatten(tree: dict, prefix: str = "") -> dict[str, np.ndarray]:
    out = {}
    for k, v in tree.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}/"))
        else:
            out[f"{prefix}{k}"] = np.ascontiguousarray(np.asarray(v, dtype=np.float32))
    return out


def unflatten(flat: dict) -> dict:
    tree: dict = {}
    for name, value in flat.items():
        *path, leaf = name.split("/")
        node = tree
        for p in path:
            node = node.setdefault(p, {})
        node[leaf] = value
    return tree


def write_safetensors(path: Path, tensors: dict, metadata: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    save_file(tensors, str(tmp), metadata=metadata)
    os.replace(tmp, path)


def write_state(path: Path, host: dict, meta: dict) -> None:
    """Poids et état de l'optimiseur (float32, noms « params/qkv », « opt/muon_buf/qkv »…), avancement en métadonnées."""
    write_safetensors(path, flatten(host), {"format": "train_jax", "state": json.dumps(meta)})


def read_state_meta(path: Path) -> dict:
    with safe_open(str(path), framework="np") as f:
        return json.loads(f.metadata()["state"])


def read_state(path: Path, spec: Spec) -> dict:
    meta = read_state_meta(path)
    model = meta.get("model", {})
    if (model.get("n_layer"), model.get("n_embd"), model.get("n_head"), model.get("vocab_size")) != (
            spec.n_layer, spec.n_embd, spec.n_head, spec.vocab_size):
        raise ValueError(f"{path} : un autre modèle ({model}) que {spec.name}")
    tree = unflatten(load_file(str(path)))
    return {"params": tree["params"], "opt": tree["opt"], "meta": meta}


def to_torch_state(params: dict) -> dict[str, np.ndarray]:
    """Les poids sous les noms du state_dict de gpt.ModernGPT (une matrice par bloc), en float32."""
    p = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}
    out = {"embed.weight": p["embed"]}
    for i in range(p["qkv"].shape[0]):
        for k, suffix in TORCH_NAMES.items():
            out[f"blocks.{i}.{suffix}"] = np.ascontiguousarray(p[k][i])
    out["skip_weights"], out["lm_head.weight"] = p["skip_weights"], p["lm_head"]
    return out


def from_torch_state(sd: dict, n_layer: int) -> dict[str, np.ndarray]:
    """L'inverse : un state_dict de ModernGPT (tableaux numpy) → les poids empilés."""
    params = {"embed": sd["embed.weight"], "skip_weights": sd["skip_weights"], "lm_head": sd["lm_head.weight"]}
    for k, suffix in TORCH_NAMES.items():
        params[k] = np.stack([sd[f"blocks.{i}.{suffix}"] for i in range(n_layer)])
    return {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}


def write_chat_checkpoint(path: Path, params: dict, spec: Spec, tokenizer, dropout_rate: float, info: dict) -> None:
    """Un .safetensors que gpt.load_checkpoint relit (chat.py, l'interface) : les métadonnées de gpt.save_checkpoint."""
    config = dict(vocab_size=spec.vocab_size, block_size=spec.block, n_embd=spec.n_embd, n_head=spec.n_head,
                  n_layer=spec.n_layer, dropout=dropout_rate, arch="modern")
    metadata = {"config": json.dumps(config), "tokenizer": json.dumps(tokenizer.state(), ensure_ascii=False),
                "info": json.dumps(info, ensure_ascii=False)}
    write_safetensors(path, to_torch_state(params), metadata)


def load_chat_checkpoint(path: Path) -> tuple[dict, dict, dict]:
    """(poids, config, infos) d'un .safetensors de gpt.py (arch modern)."""
    with safe_open(str(path), framework="np") as f:
        metadata = f.metadata()
    config = json.loads(metadata["config"])
    if config.get("arch") != "modern":
        raise ValueError(f"{path} : pas un ModernGPT (arch={config.get('arch')!r})")
    return from_torch_state(load_file(str(path)), config["n_layer"]), config, json.loads(metadata.get("info", "{}"))


def import_torch_state(path: Path, spec: Spec) -> dict:
    """Un pretrain_state.pt de train_tpu.py (ARCH=modern, OPTIM=normuon) → l'état de ce script, pour continuer
    exactement là où PyTorch s'est arrêté. Ordre des tampons de Muon : model.blocks.named_parameters() en 2D,
    soit, bloc par bloc, attn.qkv, attn.proj, mlp.fc, mlp.proj ; groupes d'Adam : embed, lm_head, skip_weights."""
    import torch

    state = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    numpy = lambda t: t.detach().float().numpy()  # noqa: E731
    params = from_torch_state({k: numpy(v) for k, v in state["model"].items()}, spec.n_layer)
    muon, adam = state["optimizer"]
    L = spec.n_layer
    buffers, second = [numpy(b) for b in muon["buffers"]], [numpy(v) for v in muon.get("second") or []]
    opt = {"muon_buf": {k: np.stack([buffers[4 * i + j] for i in range(L)]) for j, k in enumerate(BLOCK_KEYS)},
           "muon_v": {k: (np.stack([second[4 * i + j] for i in range(L)]) if second
                          else np.zeros(params[k].shape[:2] + (1,), np.float32)) for j, k in enumerate(BLOCK_KEYS)},
           "adam_m": {k: numpy(adam["m"][g][0]) for g, k in enumerate(ADAM_KEYS)},
           "adam_v": {k: numpy(adam["v"][g][0]) for g, k in enumerate(ADAM_KEYS)},
           "t": np.asarray(float(adam["t"]), np.float32)}
    for k in BLOCK_KEYS + ADAM_KEYS:
        if k in params and params[k].shape != init_shapes(spec)[k]:
            raise ValueError(f"{path} : {k} de forme {params[k].shape}, attendu {init_shapes(spec)[k]} ({spec.name})")
    meta = {"step": int(state["step"]), "tokens": int(state["tokens"]), "best": float(state["best"]),
            "seconds": float(state["seconds"]), "optimizer": "normuon", "source": str(path)}
    return {"params": params, "opt": opt, "meta": meta}


def init_shapes(spec: Spec) -> dict:
    V, C, L = spec.vocab_size, spec.n_embd, spec.n_layer
    return {"embed": (V, C), "qkv": (L, 3 * C, C), "attn_proj": (L, C, C), "fc": (L, 4 * C, C),
            "mlp_proj": (L, C, 4 * C), "skip_weights": (L // 2,), "lm_head": (V, C)}


def find_resume_state(spec: Spec) -> dict | None:
    """L'état à reprendre : le plus avancé (en tokens) parmi OUT/pretrain_state.safetensors, ceux trouvés sous
    RESUME_DIR, et IMPORT_TORCH_STATE (converti). Une reprise après migration ne repart donc jamais d'un état
    plus ancien que le dernier sauvegardé."""
    dirs = ["/kaggle/input"] + ([os.environ["RESUME_DIR"]] if os.environ.get("RESUME_DIR") else [])
    paths = {p.resolve() for p in [OUT / "pretrain_state.safetensors"] if p.exists()}
    paths |= {Path(f).resolve() for d in dirs for f in glob.glob(f"{d}/**/pretrain_state.safetensors", recursive=True)}
    candidates = sorted((read_state_meta(p)["tokens"], p.stat().st_mtime, str(p)) for p in paths)
    if IMPORT_TORCH_STATE:
        imported = import_torch_state(Path(IMPORT_TORCH_STATE), spec)
        if not candidates or imported["meta"]["tokens"] > candidates[-1][0]:
            log(f"état de train_tpu.py importé : {IMPORT_TORCH_STATE}")
            return imported
    return read_state(Path(candidates[-1][2]), spec) if candidates else None


# --- Entraînement -----------------------------------------------------------------------------------------


def run_stage(name: str, params: dict | None, spec: Spec, parts: list[tuple[np.ndarray, float]], val_data: np.ndarray,
              cfg: dict, hours: float, tokenizer, total_hours: float | None = None, state_in: dict | None = None,
              state_out: Path | None = None, max_tokens: float | None = None, save_every: float = 0,
              eval_every: float = EVAL_EVERY_S, micro_batch: int = MICRO_BATCH, keep_final: bool = False) -> dict:
    """Entraîne jusqu'au budget (`max_tokens`, reprise comprise) ou `hours` heures ; garde le meilleur modèle dans
    OUT/checkpoints/{name}.safetensors (`keep_final` : le modèle de la fin, dans tous les cas). Renvoie les poids de
    la fin. `params` : None si `state_in` les apporte (pas de copie inutile dans la mémoire du TPU).

    Un fichier OUT/STOP arrête l'étape proprement, état sauvegardé (code de sortie 3) ; un fichier
    OUT/{name}.tokens change le budget en cours de route (colab_watch.py)."""
    spec = dataclasses.replace(spec, dropout=cfg["dropout"], remat=cfg.get("remat", spec.remat))
    opt = make_opt(cfg)
    accum = max(1, TOKENS_PER_STEP // (micro_batch * BLOCK))
    tokens_per_step = accum * micro_batch * BLOCK
    weights = np.array([w for _, w in parts], dtype=np.float64)
    weights /= weights.sum()
    checkpoint = OUT / "checkpoints" / f"{name}.safetensors"
    best, step, tokens_seen, seconds_before = float("inf"), 0, 0, 0.0
    opt_state = None
    if state_in is not None:
        params, opt_state = jax.device_put(state_in["params"]), jax.device_put(state_in["opt"])
        meta = state_in["meta"]
        best, step, tokens_seen, seconds_before = float(meta["best"]), int(meta["step"]), int(meta["tokens"]), float(meta["seconds"])
        log(f"reprise : pas {step:,}, {tokens_seen / 1e9:.2f} G tokens, {seconds_before / 3600:.2f} h faites")
    if params is None:
        raise ValueError(f"{name} : ni poids ni état à reprendre")
    if opt_state is None:
        opt_state = init_opt_state(params, opt)
    train_step = make_train_step(spec, opt, accum)
    evaluate = make_eval_fn(spec)
    generator = np.random.default_rng(SEED)  # mêmes morceaux de validation à chaque mesure, comme train_tpu.py
    val_batches = jax.device_put(np.stack([random_rows(val_data, micro_batch, generator) for _ in range(EVAL_BATCHES)]))
    generator = np.random.default_rng(SEED * 100 + 1000 * SESSION + zlib.crc32(name.encode()) % 1000 + step)

    def make_batch() -> np.ndarray:
        out = np.empty((accum, micro_batch, BLOCK + 1), np.int32)
        for a in range(accum):  # chaque micro-lot vient d'un corpus tiré selon le mélange
            out[a] = random_rows(parts[generator.choice(len(parts), p=weights)][0], micro_batch, generator)
        return out

    total = (total_hours or hours) * 3600
    start_step, start = step, time.time()
    planned = step + 10**9  # sans budget en tokens : replanifié à chaque mesure, d'après la vitesse
    if max_tokens:  # les pas qui restent pour atteindre le budget (une reprise en a déjà fait une partie)
        planned = step + int(max(0.0, max_tokens - tokens_seen) // tokens_per_step)
    log(f"=== {name} : {hours:g} h · {accum} micro-lots × {micro_batch} × {BLOCK} × 1 cœurs = {tokens_per_step:,} tokens par pas")
    loss_sum, loss_count = 0.0, 0  # depuis la dernière mesure
    in_flight: collections.deque = collections.deque()  # les loss des derniers pas, encore sur le TPU
    last_progress, progress_tokens = time.time(), tokens_seen
    last_eval_step, last_eval_time, last_save = step, time.time(), time.time()
    rate_start = None  # (heure, tokens) après la compilation : la vitesse mesurée ne la compte pas
    saved_step = state_step = None  # les pas du dernier modèle exporté, du dernier état sauvegardé
    mfu_logged = False

    def progress_now() -> float:
        if max_tokens:
            return tokens_seen / max_tokens
        return (seconds_before + time.time() - start) / total

    lr = lr_multiplier(cfg, progress_now())

    def speed() -> float:
        if rate_start is None or time.time() <= rate_start[0]:
            return 0.0
        return (tokens_seen - rate_start[1]) / (time.time() - rate_start[0])

    def collect() -> None:
        """Rapatrie les loss des pas encore en vol (attend le TPU : seulement pour journaliser)."""
        nonlocal loss_sum, loss_count
        while in_flight:
            loss_sum += float(in_flight.popleft())
            loss_count += 1

    def save_state() -> None:
        """Poids, optimiseur et avancement : copiés du TPU maintenant, écrits sur disque en arrière-plan."""
        nonlocal state_step
        if state_out is None:
            return
        state_step = step
        host = jax.device_get({"params": params, "opt": opt_state})
        meta = {"step": step, "tokens": tokens_seen, "best": best, "seconds": seconds_before + time.time() - start,
                "optimizer": opt.kind, "model": dict(vocab_size=spec.vocab_size, n_layer=spec.n_layer,
                                                     n_embd=spec.n_embd, n_head=spec.n_head, block_size=spec.block)}
        WRITER.submit(lambda: write_state(state_out, host, meta))

    def export(val: float) -> None:
        host = jax.device_get(params)
        info = dict(step=step, val_loss=val, tokens=tokens_seen,
                    hyperparameters={**cfg, "model": spec.name, "attn": spec.attn, "micro_batch": micro_batch,
                                     "tokens_per_step": tokens_per_step, "compute_dtype": spec.dtype, "framework": "jax"})
        WRITER.submit(lambda: write_chat_checkpoint(checkpoint, host, spec, tokenizer, cfg["dropout"], info))

    def measure(final: bool = False) -> None:
        """Validation ; toutes les `save_every` secondes (et à la fin), l'état, et le modèle s'il est le meilleur."""
        nonlocal best, loss_sum, loss_count, last_eval_step, last_eval_time, last_save, saved_step, mfu_logged
        collect()
        val = float(evaluate(params, val_batches))
        train_loss = loss_sum / loss_count if loss_count else float("nan")
        loss_sum, loss_count, last_eval_step, last_eval_time = 0.0, 0, step, time.time()
        note = ""
        if val < best:
            best, note = val, "  ← meilleur"
        if final or time.time() - last_save >= save_every:
            if val <= best or (final and keep_final) or not checkpoint.exists():  # pré-entraînement : la fin, toujours
                export(val)
                saved_step, note = step, note + ", sauvegardé"
            save_state()
            last_save = time.time()
        tps = speed()
        log(f"{name}  pas {step:>6,}/{planned:,}   train {train_loss:.4f}   val {val:.4f}   lr ×{lr:.3f}   "
            f"{tokens_seen / 1e9:.2f} G tokens   {tps:,.0f} tokens/s   {(time.time() - start) / 3600:.2f} h{note}")
        if tps and not mfu_logged:
            log(f"{name} : {tps:,.0f} tokens/s, soit {100 * tps * flops_per_token(spec) / (PEAK_TFLOPS * 1e12):.1f} % "
                f"des {PEAK_TFLOPS:g} TFLOPS de la puce (attention {spec.attn}, recalcul {spec.remat})")
            mfu_logged = True

    prefetcher = Prefetcher(make_batch)
    try:
        while step < planned:
            progress = progress_now()
            if not max_tokens and (progress >= 1 or time.time() - start >= hours * 3600):
                break
            lr = lr_multiplier(cfg, progress)
            params, opt_state, loss = train_step(params, opt_state, prefetcher.next(), np.float32(lr), np.int32(step))
            step += 1
            tokens_seen += tokens_per_step
            in_flight.append(loss)
            if len(in_flight) > 2:  # deux pas d'avance au plus : le TPU reste occupé, le Python ne s'emballe pas
                loss_sum += float(in_flight.popleft())
                loss_count += 1
            if step == start_step + 1:
                loss.block_until_ready()
                log(f"{name} : pas compilé et exécuté en {time.time() - start:.0f} s")
                rate_start = (time.time(), tokens_seen)
                last_progress, progress_tokens = time.time(), tokens_seen
            if time.time() - last_progress >= PROGRESS_EVERY_S:  # entre deux validations, de quoi suivre en direct
                collect()
                train_loss = loss_sum / max(1, loss_count)
                rate = (tokens_seen - progress_tokens) / max(time.time() - last_progress, 1e-9)
                log(f"{name}  avancement  pas {step:,}/{planned:,}   train {train_loss:.4f}   lr ×{lr:.3f}   "
                    f"{tokens_seen / 1e9:.3f} G tokens   {rate:,.0f} tokens/s")
                last_progress, progress_tokens = time.time(), tokens_seen
            if step % 10 == 0 and (OUT / "STOP").exists():  # changement de machine
                save_state()
                WRITER.wait()
                log(f"{name} : arrêt demandé au pas {step:,} ({tokens_seen / 1e9:.3f} G tokens), état sauvegardé")
                raise SystemExit(3)
            budget_file = OUT / f"{name}.tokens"  # le suivi peut ajuster le budget en cours de route
            if max_tokens and step % 10 == 0 and budget_file.exists():
                try:
                    new_total = float(budget_file.read_text().strip())
                except ValueError:  # fichier en cours d'écriture : on verra dans 10 pas
                    new_total = max_tokens
                if new_total != max_tokens:
                    max_tokens = new_total
                    planned = step + int(max(0.0, max_tokens - tokens_seen) // tokens_per_step)
                    log(f"{name} : budget ajusté à {max_tokens / 1e9:.2f} G tokens (fin prévue au pas {planned:,})")
            if step == start_step + FIRST_EVAL or (step > start_step + FIRST_EVAL and time.time() - last_eval_time >= eval_every):
                if not max_tokens and rate_start is not None and step > start_step + 1:
                    sec_per_step = (time.time() - rate_start[0]) / (step - start_step - 1)
                    left = min(hours * 3600 - (time.time() - start), total - (seconds_before + time.time() - start))
                    planned = step + max(0, int(left / sec_per_step))
                measure()
        if step > last_eval_step or not checkpoint.exists() or (keep_final and saved_step != step):
            measure(final=True)  # la fin du planning, learning rate au plus bas : souvent le meilleur moment
        if state_step != step:
            save_state()
        WRITER.wait()
    finally:
        prefetcher.close()
    log(f"{name} : fin au pas {step:,}, meilleure val {best:.4f}")
    return params


def full_loss(params: dict, spec: Spec, data: np.ndarray, micro_batch: int) -> float:
    """Loss sur tout un texte, en morceaux de BLOCK tokens qui se suivent (le `full_loss` du notebook) ; le
    dernier lot est complété par des morceaux de poids nul : une seule forme, une seule compilation."""
    spec = dataclasses.replace(spec, dropout=0.0)
    n = (len(data) - 1) // BLOCK
    rows = rows_at(data, np.arange(n) * BLOCK)
    pad = (-n) % micro_batch
    rows = np.concatenate([rows, np.zeros((pad, BLOCK + 1), np.int32)]).reshape(-1, micro_batch, BLOCK + 1)
    weights = np.concatenate([np.ones(n, np.float32), np.zeros(pad, np.float32)]).reshape(-1, micro_batch)

    @jax.jit
    def total(params, rows, weights):
        cp = compute_params(params, spec)
        per_row = jax.lax.map(lambda b: jnp.mean(token_losses(forward(cp, b[:, :-1], spec), b[:, 1:]), axis=-1), rows)
        return jnp.sum(per_row * weights)

    return float(total(params, rows, weights)) / n


def base_spec(vocab_size: int, n_layer: int, n_embd: int, n_head: int) -> Spec:
    if n_embd % n_head:
        raise ValueError(f"N_EMBD ({n_embd}) doit être divisible par N_HEAD ({n_head})")
    return Spec(vocab_size=vocab_size, n_layer=n_layer, n_embd=n_embd, n_head=n_head, block=BLOCK, dtype=COMPUTE_DTYPE,
                attn=resolve_attention(ATTN, n_head, n_embd // n_head), remat=REMAT)


def score(name: str, spec: Spec, tokenizer, tokens: dict, simpsons: dict) -> None:
    """Le score du meilleur modèle affiné, comme à la fin du notebook : bits par caractère sur la validation et le
    test complets des Simpson ; puis 4 répliques écrites par gpt.reply (sur CPU, depuis le fichier exporté)."""
    path = OUT / "checkpoints" / f"{name}.safetensors"
    params, config, info = load_chat_checkpoint(path)
    params = jax.device_put(params)
    result = {"session": SESSION, "model": spec.name, "arch": "modern", "n_layer": spec.n_layer, "n_embd": spec.n_embd,
              "n_head": spec.n_head, "vocab_size": spec.vocab_size, "block_size": spec.block,
              "n_params": count_params(params), "step": info.get("step"), "tokens": info.get("tokens"),
              "val_loss_estimate": info.get("val_loss")}
    for part in ("val", "test"):
        loss = full_loss(params, spec, tokens[part], MICRO_BATCH)
        result[f"{part}_loss"] = loss
        result[f"{part}_bits_per_char"] = loss / math.log(2) / (len(simpsons[part]) / len(tokens[part]))
    (OUT / "checkpoints" / f"{name}.json").write_text(json.dumps(result, indent=2))
    (OUT / "resultat.json").write_text(json.dumps(result, indent=2))
    log("résultat : " + json.dumps(result))
    model, _, _ = gpt.load_checkpoint(path, "cpu")
    lines = []
    for me, who, message in [("Bart", "Homer", "Papa, je peux avoir un poney ?"),
                             ("Marge", "Homer", "Homer, tu as encore bu chez Moe ?"),
                             ("Homer", "Moe", "Moe, une bière s'il te plaît."),
                             ("Homer", "Burns", "Monsieur Burns, je voudrais une augmentation.")]:
        text = gpt.reply(model, tokenizer, f"{me}: {message}\n{who}:", who, "cpu")
        lines.append(f"{me}: {message}\n{who}: {text}\n")
    (OUT / "exemples.txt").write_text("\n".join(lines), encoding="utf-8")
    log("exemples :\n" + "\n".join(lines))


def train_session() -> None:
    """Une session : pré-entraînement (reprise comprise), affinage sur les Simpson, score et exemples."""
    tokenizer = gpt.tokenizer_from_state(json.loads((INPUT / "tokenizer.json").read_text(encoding="utf-8")))
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    general, general_val, tokens, simpsons = load_data(tokenizer)
    state_in = None
    if INIT:  # affinage seul : les dimensions viennent du modèle
        init_np, config, _ = load_chat_checkpoint(Path(INIT))
        spec = base_spec(config["vocab_size"], config["n_layer"], config["n_embd"], config["n_head"])
        params = jax.device_put(init_np)
    else:
        spec = base_spec(tokenizer.vocab_size, N_LAYER, N_EMBD, N_HEAD)
        state_in = find_resume_state(spec)
        params = None if state_in is not None else init_params(spec)
    n_params = int(sum(np.prod(s) for s in init_shapes(spec).values()))
    device = jax.devices()[0]
    log(f"modèle {spec.name} : {n_params:,} paramètres · {jax.device_count()} × {device.device_kind} · JAX {jax.__version__}"
        f" · attention {spec.attn} · recalcul {spec.remat} · calcul {spec.dtype}")
    if active_levers():  # rien de plus au journal quand tout est par défaut
        log("leviers de vitesse : " + "   ".join(f"{k}={v}" for k, v in active_levers().items()))
    name = f"dialogues_bpe8192_jax_{round(n_params / 1e6)}M_s{SESSION}"
    if INIT:
        log(f"poids pré-entraînés : {INIT}")
    else:
        params = run_stage("pretrain_jax", params, spec, general, general_val, PRETRAIN, HOURS_PRETRAIN, tokenizer,
                           total_hours=TOTAL_PRETRAIN_HOURS, state_in=state_in,
                           state_out=OUT / "pretrain_state.safetensors", max_tokens=PRETRAIN_TOKENS or None,
                           save_every=SAVE_EVERY_S, eval_every=EVAL_EVERY_S, keep_final=True)
    finetune = FINETUNE_MODERN if FT_OPTIM == "normuon" else FINETUNE
    replay = finetune["replay"]
    general_parts = [(d, w * replay / sum(w for _, w in general)) for d, w in general]
    params = run_stage(name, params, spec, [(tokens["train"], 1 - replay)] + general_parts, tokens["val"], finetune,
                       HOURS_FINETUNE, tokenizer, max_tokens=FT_TOKENS or None, eval_every=FT_EVAL_EVERY_S,
                       micro_batch=FT_MICRO_BATCH)
    del params
    gc.collect()
    score(name, spec, tokenizer, tokens, simpsons)


# --- Essai : vitesse, MFU et mémoire de chaque configuration ---------------------------------------------


def xla_report(compiled) -> dict:
    """Ce qu'XLA dit d'un programme compilé : son estimation (flops, transcendantes, octets lus et écrits) et son
    analyse mémoire (arguments, sorties, alias, temporaires, pic). L'estimation compte le corps d'une boucle (le
    lax.scan des micro-lots, ceux des blocs) une seule fois : à comparer entre leviers, pas à la formule 6 N."""
    report: dict = {}
    try:
        cost = compiled.cost_analysis()
        if isinstance(cost, (list, tuple)):  # une liste d'un seul dictionnaire selon les versions
            cost = cost[0] if cost else None
        for key in ("flops", "transcendentals", "bytes accessed", "optimal_seconds"):
            if cost and key in cost:
                report[key.replace(" ", "_")] = float(cost[key])
    except Exception as problem:  # pas d'estimation sur ce backend
        report["cost_error"] = f"{type(problem).__name__}: {str(problem)[:200]}"
    try:
        memory = compiled.memory_analysis()
        for key in ("argument", "output", "alias", "temp", "generated_code"):
            if getattr(memory, f"{key}_size_in_bytes", None) is not None:
                report[f"{key}_bytes"] = int(getattr(memory, f"{key}_size_in_bytes"))
        if getattr(memory, "peak_memory_in_bytes", None) is not None:
            report["peak_bytes"] = int(memory.peak_memory_in_bytes)
    except Exception as problem:  # pas d'analyse mémoire sur ce backend
        report["memory_error"] = f"{type(problem).__name__}: {str(problem)[:200]}"
    return report


def time_calls(run, n: int, warmup: int = 2) -> float:
    """Durée moyenne d'un appel de `run`, en secondes : `warmup` appels de chauffe non comptés, puis `n` appels,
    chacun attendu jusqu'au bout (block_until_ready) : le temps du TPU, pas celui de l'envoi."""
    for _ in range(warmup):
        jax.block_until_ready(run())
    t0 = time.perf_counter()
    for _ in range(n):
        jax.block_until_ready(run())
    return (time.perf_counter() - t0) / n


def bench_parts(spec: Spec, opt: Opt, accum: int, params, opt_state, batch, tokens_per_step: int) -> tuple[dict, dict]:
    """BENCH_PARTS : forward + loss, forward + backward (gradients, sans optimiseur) et optimiseur seul, chacun
    compilé à part et chronométré sur BENCH_STEPS appels après 2 de chauffe, avec les poids et les lots du pas
    complet. L'optimiseur seul ne reçoit pas poids et état en don (mêmes octets lus et écrits, sorties à côté) :
    les poids restent ceux du pas complet. compile_s : les trois compilations seules (l'optimiseur est compilé sur
    la forme des gradients, sans exécuter forward + backward).

    Le forward seul est un forward d'inférence : sans value_and_grad, XLA n'écrit pas les résidus que le backward
    relira, ce que fait le forward du pas complet. fb_minus_inference_forward_s (forward_backward_s − forward_s)
    compte donc ces écritures avec le backward, et forward_mfu flatte le forward de l'entraînement.
    Renvoie (durées, appels) ; `appels` relance un morceau (pour la trace) et garde en mémoire les gradients de
    l'optimiseur seul tant qu'il en contient."""
    fns = make_step_parts(spec, opt, accum)
    step0, one = np.int32(0), np.float32(1.0)
    grads_shape = jax.tree.map(lambda p: jax.ShapeDtypeStruct(p.shape, F32), params)  # merge_grads : float32
    t0 = time.time()
    forward = jax.jit(fns["forward"]).lower(params, batch, step0).compile()
    forward_backward = jax.jit(fns["forward_backward"]).lower(params, batch, step0).compile()
    update = jax.jit(fns["optimizer"]).lower(params, grads_shape, opt_state, one).compile()
    out: dict = {"compile_s": round(time.time() - t0, 1)}
    held: dict = {}  # les gradients de l'optimiseur seul, calculés juste avant sa mesure
    runs = {"forward": lambda: forward(params, batch, step0),
            "forward_backward": lambda: forward_backward(params, batch, step0),
            "optimizer": lambda: update(params, held["grads"], opt_state, one)}
    for name, run in runs.items():
        if name == "optimizer":  # pas avant : ils ne pèsent pas sur la mémoire des deux autres mesures
            held["grads"], _ = forward_backward(params, batch, step0)
        out[f"{name}_s"] = round(time_calls(run, BENCH_STEPS), 4)
    flops = tokens_per_step * flops_per_token(spec)  # forward : le tiers des flops du pas
    out["fb_minus_inference_forward_s"] = round(out["forward_backward_s"] - out["forward_s"], 4)
    out["forward_mfu"] = round(flops / 3 / (out["forward_s"] * PEAK_TFLOPS * 1e12), 4)
    out["forward_backward_mfu"] = round(flops / (out["forward_backward_s"] * PEAK_TFLOPS * 1e12), 4)
    if BENCH_COST:
        for name, exe in (("forward", forward), ("forward_backward", forward_backward), ("optimizer", update)):
            out[f"{name}_xla"] = xla_report(exe)
    return out, runs


def profile_run(compiled, params, opt_state, batch, runs: dict | None, trace_dir: Path, entry: dict):
    """PROFILE : le HLO optimisé du pas (step.hlo.txt), puis une trace jax.profiler de chaque morceau de
    BENCH_PARTS et de PROFILE_STEPS pas complets (annotés « train »), dans `trace_dir`, à ouvrir avec XProf ou
    TensorBoard (onglets trace_viewer, op_profile, memory_viewer). Les morceaux d'abord : les pas complets
    reçoivent les poids en don. Chaque morceau est retiré de `runs` une fois tracé : ses gradients et ses
    programmes ne pèsent plus sur la mémoire des pas complets. Renvoie les poids et l'état après ces pas."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    try:  # à part : sans HLO lisible (exécutable relu du cache…), la trace se fait quand même
        text = compiled.as_text()
        if text:
            (trace_dir / "step.hlo.txt").write_text(text)
    except Exception as problem:
        entry["hlo_error"] = f"{type(problem).__name__}: {str(problem)[:200]}"
    jax.block_until_ready((params, opt_state))
    jax.profiler.start_trace(str(trace_dir))
    try:
        for name in list(runs or {}):
            with jax.profiler.TraceAnnotation(name):
                jax.block_until_ready(runs.pop(name)())
        gc.collect()
        for i in range(PROFILE_STEPS):
            with jax.profiler.StepTraceAnnotation("train", step_num=i):
                params, opt_state, loss = compiled(params, opt_state, batch, np.float32(1.0), np.int32(i))
                loss.block_until_ready()
    finally:
        jax.profiler.stop_trace()
    return params, opt_state


def bench() -> list[dict]:
    """BENCH = « couches x largeur x têtes : micro-lot : recalcul [: attention [: ft]] », séparés par des virgules.
    Pour chacun : BENCH_WARMUP (3) pas complets pour chauffer puis BENCH_STEPS (10) mesurés (NorMuon, ou AdamW +
    dropout avec « ft »), sur des tokens au hasard ; puis, s'ils sont demandés, les morceaux du pas (BENCH_PARTS)
    et la trace (PROFILE), qui ne changent rien à la mesure de cette configuration, mais font monter le pic de
    mémoire du processus (peak_hbm_bytes) vu par les suivantes : avec eux, une configuration par processus pour
    comparer la mémoire, ou compiled_peak_bytes. Une configuration qui échoue (mémoire du TPU dépassée…) n'arrête
    pas les autres."""
    path = INPUT / "tokenizer.json"
    vocab = gpt.tokenizer_from_state(json.loads(path.read_text(encoding="utf-8"))).vocab_size if path.exists() else 8192
    device = jax.devices()[0]
    levers = active_levers()
    lab = bool(levers) or BENCH_COST or BENCH_PARTS  # PROFILE implique BENCH_COST ; sinon, le JSON d'avant
    if levers:
        log("leviers de vitesse : " + "   ".join(f"{k}={v}" for k, v in levers.items()))
    items = [s.strip() for s in BENCH.split(",") if s.strip()]
    if (BENCH_PARTS or PROFILE) and len(items) > 1:
        log("BENCH_PARTS et PROFILE font monter le pic de mémoire du processus : peak_hbm_bytes des configurations "
            "suivantes peut venir d'une précédente (peak_hbm_is_process_max) ; une configuration par processus pour "
            "comparer la mémoire, ou compiled_peak_bytes")
    lab_ran = False  # BENCH_PARTS ou PROFILE a déjà tourné dans ce processus
    results = []
    for item in items:
        entry: dict = {"config": item, "device": device.device_kind}
        if levers:
            entry["levers"] = levers
        if lab:  # les flags de XLA de cet essai
            for var in ("LIBTPU_INIT_ARGS", "XLA_FLAGS"):
                if os.environ.get(var):
                    entry[var.lower()] = os.environ[var]
        params = opt_state = batch = compiled = loss = runs = None
        try:
            fields = item.split(":")
            n_layer, n_embd, n_head = (int(x) for x in fields[0].split("x"))
            micro_batch = int(fields[1]) if len(fields) > 1 else MICRO_BATCH
            remat = fields[2] if len(fields) > 2 and fields[2] else REMAT
            attn = fields[3] if len(fields) > 3 and fields[3] else ATTN
            finetune = len(fields) > 4 and fields[4] == "ft"
            if n_embd % n_head:
                raise ValueError(f"{n_embd} n'est pas divisible par {n_head}")
            attn = resolve_attention(attn, n_head, n_embd // n_head)
            cfg = FINETUNE if finetune else PRETRAIN
            spec = Spec(vocab_size=vocab, n_layer=n_layer, n_embd=n_embd, n_head=n_head, block=BLOCK, dtype=COMPUTE_DTYPE,
                        attn=attn, remat=remat, dropout=cfg["dropout"])
            opt = make_opt(cfg)
            accum = max(1, TOKENS_PER_STEP // (micro_batch * BLOCK))
            tokens_per_step = accum * micro_batch * BLOCK
            entry.update(n_layer=n_layer, n_embd=n_embd, n_head=n_head, micro_batch=micro_batch, remat=remat, attn=attn,
                         optimizer=opt.kind, dropout=spec.dropout, accum=accum, tokens_per_step=tokens_per_step)
            if remat == "dots_attn" and (attn != "splash" or spec.dropout):  # voir REMAT_POLICIES
                entry["remat_note"] = ("dots_attn sans splash (ou avec le dropout de l'affinage) : rien du noyau "
                                       "d'attention n'est gardé, au plus sa sortie, en plus de dots")
            if attn in ("splash", "flash"):
                entry["attn_check"] = round(check_attention(attn, n_head, n_embd // n_head, BLOCK), 5)
            params = init_params(spec)
            entry["params"] = count_params(params)
            opt_state = init_opt_state(params, opt)
            batch = jax.device_put(np.random.default_rng(0).integers(0, vocab, size=(accum, micro_batch, BLOCK + 1), dtype=np.int32))
            t0 = time.time()
            compiled = make_train_step(spec, opt, accum).lower(params, opt_state, batch, np.float32(1.0), np.int32(0)).compile()
            entry["compile_s"] = round(time.time() - t0, 1)
            try:
                memory = compiled.memory_analysis()
                entry["compiled_temp_bytes"] = int(memory.temp_size_in_bytes)
                entry["compiled_argument_bytes"] = int(memory.argument_size_in_bytes)
                if lab and getattr(memory, "peak_memory_in_bytes", None) is not None:  # ce programme seul
                    entry["compiled_peak_bytes"] = int(memory.peak_memory_in_bytes)
            except Exception:  # pas d'analyse mémoire sur ce backend
                pass
            if BENCH_COST:
                entry["xla"] = {**xla_report(compiled), "model_flops": flops_per_token(spec) * tokens_per_step}
            for i in range(BENCH_WARMUP + BENCH_STEPS):
                if i == BENCH_WARMUP:
                    loss.block_until_ready()
                    t3 = time.time()
                params, opt_state, loss = compiled(params, opt_state, batch, np.float32(1.0), np.int32(i))
            loss = float(loss)
            seconds = time.time() - t3
            tps = BENCH_STEPS * tokens_per_step / seconds
            entry.update(ok=True, step_s=round(seconds / BENCH_STEPS, 4), tokens_per_s=round(tps),
                         mfu=round(tps * flops_per_token(spec) / (PEAK_TFLOPS * 1e12), 4), loss=round(loss, 4))
            stats = device.memory_stats() or {}
            if "peak_bytes_in_use" in stats:
                entry["peak_hbm_bytes"] = int(stats["peak_bytes_in_use"])
                entry["hbm_limit_bytes"] = int(stats.get("bytes_limit", 0))
                if lab_ran:  # le pic du processus : peut-être celui des morceaux ou de la trace d'une précédente
                    entry["peak_hbm_is_process_max"] = True
            if BENCH_PARTS:  # après la mesure et le pic de cette configuration (celui du processus, lui, monte)
                lab_ran = True
                try:
                    parts, runs = bench_parts(spec, opt, accum, params, opt_state, batch, tokens_per_step)
                    parts["step_s"] = entry["step_s"]
                    parts["other_s"] = round(entry["step_s"] - parts["forward_backward_s"] - parts["optimizer_s"], 4)
                    entry["parts"] = parts
                except Exception as problem:
                    entry["parts_error"] = f"{type(problem).__name__}: {str(problem)[:400]}"
            if PROFILE:  # un dossier par configuration, attention, recalcul, jeu de leviers et flags de XLA
                lab_ran = True
                tag = "_".join([attn, remat] + [f"{k}-{v}" for k, v in active_levers(everything=True).items()])
                flags = [os.environ.get(var, "") for var in ("LIBTPU_INIT_ARGS", "XLA_FLAGS")]
                if any(flags):  # une empreinte courte et stable
                    digest = zlib.crc32("\n".join(flags).encode())
                    tag += f"_flags{digest:08x}"
                trace_dir = Path(PROFILE) / re.sub(r"[^A-Za-z0-9._-]+", "_", f"{item}__{tag}")
                try:
                    params, opt_state = profile_run(compiled, params, opt_state, batch, runs, trace_dir, entry)
                    entry["profile_dir"] = str(trace_dir)
                except Exception as problem:  # profileur absent de cette installation…
                    entry["profile_error"] = f"{type(problem).__name__}: {str(problem)[:400]}"
        except Exception as problem:  # mémoire du TPU dépassée, le plus souvent
            entry.update(ok=False, error=f"{type(problem).__name__}: {str(problem)[:400]}")
        del params, opt_state, batch, compiled, runs
        gc.collect()
        jax.clear_caches()
        print(json.dumps(entry), flush=True)
        results.append(entry)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bench_jax.json").write_text(json.dumps(results, indent=2))
    return results


def setup_jax() -> None:
    """Cache de compilation sur disque (JAX_CACHE ; 0 : aucun) : une relance sur la même machine ne recompile pas."""
    cache = os.environ.get("JAX_CACHE", str(Path(tempfile.gettempdir()) / "jax_cache_train_jax"))
    if cache != "0":
        jax.config.update("jax_compilation_cache_dir", cache)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 5)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    setup_jax()
    if mode == "bench":
        bench()
    elif mode == "train":
        train_session()
    else:
        sys.exit(f"mode inconnu : {mode!r} (bench ou train)")
