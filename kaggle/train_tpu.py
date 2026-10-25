"""Pré-entraînement d'un GPT taille GPT-2 sur TPU (Kaggle, TPU v5e-8), puis affinage sur les Simpson.

    python train_tpu.py bench       # essai : vitesse de chaque taille sur 1 cœur, puis mini-run sur les 8
    python train_tpu.py train       # une session : reprise, pré-entraînement, affinage, test
    python train_tpu.py speed       # laboratoire de vitesse : le pas d'entraînement réel, tokens/s, MFU, compteurs XLA
    INIT=modele.safetensors python train_tpu.py train   # affinage seul (FT_OPTIM=adamw ou normuon), test

Sur une seule puce (TPU v5e-1 ou v6e-1 de Colab), tout tourne dans un seul processus.

Même modèle (gpt.py, PyTorch), même tokenizer et même découpage des Simpson que le notebook ; le calcul passe
par PyTorch/XLA. Les 8 cœurs du TPU apprennent chacun sur leurs lots (data parallel) et moyennent leurs
gradients à chaque pas. Points propres à XLA :

- le graphe de calcul est compilé une fois, puis réutilisé : toutes les formes restent fixes, et rien ne
  doit changer de valeur « en dur » d'un pas à l'autre. Le learning rate est donc un tenseur envoyé au TPU
  à chaque pas, et l'optimiseur AdamW est écrit ici (celui de PyTorch le fige comme une constante) ;
- les décisions (mesurer, s'arrêter) se prennent en nombre de pas, identiques sur les 8 cœurs : la durée
  d'un pas est mesurée, moyennée entre les cœurs, et sert à planifier la session ;
- bf16 natif, sans GradScaler ;
- l'accumulation des gradients : par défaut (ACCUM_MODE=loop), un graphe par micro-lot, et Python retrace tout
  le modèle (avant et arrière) à chaque micro-lot ; avec ACCUM_MODE=scan, les micro-lots d'un pas tournent dans
  une boucle While d'XLA (gradient_accumulation de torch_xla) : le modèle n'est tracé qu'une fois par pas.

Une session Kaggle TPU dure 9 h au plus : SESSION=1, 2… reprennent l'état de la précédente (poids, AdamW,
heures faites) ; le learning rate suit un cosinus sur TOTAL_PRETRAIN_HOURS.

Le mode speed mesure un pas d'entraînement tel que run_stage le fait (modèle de MODEL et ARCH, optimiseurs d'OPTIM,
MICRO_BATCH, ACCUM_MODE, autocast bf16, mark_step aux mêmes endroits), sur des tokens tirés au hasard, sans corpus :
SPEED_STEPS pas complets (12) de TOKENS_PER_STEP tokens, les 3 premiers (compilation) hors mesure. Il écrit une ligne
JSON (et l'ajoute à OUT/speed.jsonl) : tokens/s, MFU (pic de PEAK_TFLOPS, 918 pour un v6e ; la formule de
train_jax.py), durée des pas hors mesure et temps de compilation XLA, temps d'un pas, temps Python (traçage), temps
passé dans mark_step et à attendre le TPU, compteurs de torch_xla (compilations, exécutions, opérations retombées sur
le CPU), mémoire du TPU. Le témoin numérique : la sonde (probe), un forward et un backward sur un lot fixe, après la
mesure (loss, normes de gradients, types des produits matriciels du graphe ; SPEED_PROBE=0 : sans). PROFILE=<dossier> :
une trace (XProf, TensorBoard) de PROFILE_STEPS pas (3), prise après la mesure.

Les leviers de vitesse, tous éteints par défaut (sans eux, rien ne change : même graphe, mêmes nombres) :
- ATTN_KERNEL=splash : splash attention de torch_xla (le noyau Pallas de MaxText, appelé à travers JAX et torchax),
  sur une seule puce ; blocs de SPLASH_BLOCK tokens (512 : un seul bloc par séquence, rien de sauté ; 256 ou 128 :
  les blocs entièrement dans le futur sautés), SPLASH_FUSED=1 : le backward en un seul noyau ;
- LAYER_SCAN=1 : les blocs de ModernGPT dans des boucles While d'XLA (scan de torch_xla) : chaque moitié du U est
  tracée une fois par micro-lot au lieu d'un bloc à la fois ; sans dropout seulement (le pré-entraînement) ;
- BF16_WEIGHTS=1 (mode speed seulement) : les poids du modèle en bf16, une copie float32 pour les optimiseurs ;
- PREFETCH=n : les lots tirés et envoyés au TPU par un thread, n d'avance (mêmes tirages, dans le même ordre).
"""

import gc
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

# INPUT_DIR : le code et les données du run, quand /kaggle/input contient aussi d'anciennes versions de gpt.py
INPUT = Path(os.environ["INPUT_DIR"]) if os.environ.get("INPUT_DIR") else Path(
    glob.glob("/kaggle/input/**/gpt.py", recursive=True)[0]).parent if glob.glob(
    "/kaggle/input/**/gpt.py", recursive=True) else Path(__file__).resolve().parent
sys.path.insert(0, str(INPUT))
import gpt  # noqa: E402

OUT = Path(os.environ.get("OUT", "/kaggle/working"))
MODELS = {"small": dict(n_embd=768, n_head=12, n_layer=12),     # GPT-2 small (124 M avec 50 257 tokens)
          "medium": dict(n_embd=1024, n_head=16, n_layer=24),   # GPT-2 medium (355 M)
          # medium avec des têtes plus larges (128, 256) : mêmes poids, l'attention et le MXU du TPU mieux remplis
          "medium_h8": dict(n_embd=1024, n_head=8, n_layer=24),
          "medium_h4": dict(n_embd=1024, n_head=4, n_layer=24),
          # ~600 M, larges (têtes de 128) : l'optimum de Chinchilla pour ~40 h de v6e, des produits matriciels plus gros
          "w1536": dict(n_embd=1536, n_head=12, n_layer=20),
          "w1792": dict(n_embd=1792, n_head=14, n_layer=16),
          "tiny": dict(n_embd=64, n_head=4, n_layer=2)}          # pour les essais
MODEL = os.environ.get("MODEL", "small")
BLOCK = 512  # GPT-2 : 1 024 ; une scène de dialogue tient largement en 512 tokens, l'attention coûte 4 fois moins
MICRO_BATCH = int(os.environ.get("MICRO_BATCH", 16))  # morceaux par cœur et par micro-lot
TOKENS_PER_STEP = int(os.environ.get("TOKENS_PER_STEP", 524_288))  # ~0,5 M tokens par pas, comme GPT-2
SESSION = int(os.environ.get("SESSION", 1))
HOURS_PRETRAIN = float(os.environ.get("HOURS_PRETRAIN", 7.6))
TOTAL_PRETRAIN_HOURS = float(os.environ.get("TOTAL_PRETRAIN_HOURS", 15.2))
HOURS_FINETUNE = float(os.environ.get("HOURS_FINETUNE", 0.4))
FAKE_PRETRAIN = os.environ.get("FAKE_PRETRAIN") == "1"  # essai : les Simpson tiennent lieu de corpus général
ARCH = os.environ.get("ARCH", "course")  # course : le GPT du cours + AdamW ; modern : modded-nanoGPT + Muon
MIX = dict((k, float(v)) for k, v in (item.split(":") for item in os.environ["MIX"].split(","))) if os.environ.get("MIX") else None
LR_SCALE = float(os.environ.get("LR_SCALE", 1.0))
PRETRAIN = dict(lr=6e-4 if MODEL == "small" else 3e-4, min_lr=6e-5 if MODEL == "small" else 3e-5,
                warmup=0.02, weight_decay=0.1, dropout=0.0)
FT_LR = float(os.environ.get("FT_LR", 1e-4))  # affinage avec AdamW
FINETUNE = dict(lr=FT_LR, min_lr=FT_LR / 10, warmup=0.05, weight_decay=0.1, dropout=0.1, replay=0.25)
PRETRAIN_MODERN = dict(muon=True, muon_lr=0.05, embed_lr=0.6, head_lr=0.008, scalar_lr=0.04, warmup=0.01,
                       cooldown=0.4, dropout=0.0, normuon=os.environ.get("OPTIM", "muon") == "normuon")  # OPTIM=normuon : NorMuon
# Affinage avec NorMuon, comme le pré-entraînement : ses learning rates × FT_SCALE, montée puis descente jusqu'à 0.
# Sans dropout par défaut (FT_DROPOUT) : pré-entraîné sans, le modèle moderne en est déréglé (mesuré le 24/10 sur
# v6-591M : loss 2,63 sans, 8,08 avec 0,1 ; l'affinage de 190 pas n'a gardé que son pas 20).
FINETUNE_MODERN = dict(muon=True, normuon=True, warmup=0.05, cooldown=0.95, replay=0.25,
                       dropout=float(os.environ.get("FT_DROPOUT", 0.0)),
                       **{k: PRETRAIN_MODERN[k] * float(os.environ.get("FT_SCALE", 0.1))
                          for k in ("muon_lr", "embed_lr", "head_lr", "scalar_lr")})
FT_OPTIM = os.environ.get("FT_OPTIM", "adamw")  # l'optimiseur de l'affinage : adamw ou normuon
INIT = os.environ.get("INIT")  # affinage seul : les poids d'un modèle déjà pré-entraîné (.safetensors de gpt.py)
FT_TOKENS = float(os.environ.get("FT_TOKENS", 0))  # affinage : un budget en tokens au lieu de HOURS_FINETUNE
# Pré-entraînement : un total de tokens (toutes sessions confondues) au lieu des heures. Le planning du learning rate
# suit alors les tokens vus, et le run peut changer de machine (plus rapide, ou après une coupure) sans être décalé.
PRETRAIN_TOKENS = float(os.environ.get("PRETRAIN_TOKENS", 0))
# Tokens du corpus général gardés (le début de chaque corpus, au prorata du mélange) ; 0 : tout. Sur une machine où
# le corpus entier ne tient pas en cache, les lots tirés au hasard relisent sans cesse le disque : avec une partie
# qui tient en mémoire, l'entraînement n'attend plus.
CORPUS_CAP = float(os.environ.get("CORPUS_CAP", 0))
EVAL_EVERY_S = float(os.environ.get("EVAL_EVERY_S", 20 * 60))
# Pré-entraînement : sauvegardes (état, meilleur modèle) au plus si souvent ; 0 : à chaque mesure. L'affinage, court,
# garde chaque meilleur modèle.
SAVE_EVERY_S = float(os.environ.get("SAVE_EVERY_S", 0))
PROGRESS_EVERY_S = float(os.environ.get("PROGRESS_EVERY_S", 60))  # une ligne d'avancement (sans validation), pour le suivi
# Montée maximale du multiplicateur de learning rate par pas, après le warmup : quand le suivi relève le budget de tokens
# pendant la descente finale (un v6e obtenu après un v5e), le planning remonte d'un coup (×0,35 → ×0,92) ; ici, par
# paliers (0,005 par pas : ~115 pas, ~10 min sur v6e).
LR_RISE = float(os.environ.get("LR_RISE", 0.005))
# L'accumulation des gradients sur les micro-lots d'un pas. loop : une boucle Python, un mark_step par micro-lot
# (le modèle est retracé à chaque micro-lot : ~90 ms de Python pour ~60 ms de calcul du TPU, en medium). scan :
# gradient_accumulation de torch_xla, une boucle While d'XLA sur les micro-lots, le modèle tracé une fois par pas,
# un seul mark_step par pas ; mêmes lots, mêmes gradients, même loss. FT_ACCUM_MODE : celle de l'affinage (par
# défaut la même) ; le dropout y reste aléatoire d'un micro-lot et d'un pas à l'autre (une graine par itération).
ACCUM_MODE = os.environ.get("ACCUM_MODE", "loop")
FT_ACCUM_MODE = os.environ.get("FT_ACCUM_MODE", ACCUM_MODE)
assert {ACCUM_MODE, FT_ACCUM_MODE} <= {"loop", "scan"}, "ACCUM_MODE et FT_ACCUM_MODE : loop ou scan"
STOP_AFTER_PRETRAIN = os.environ.get("STOP_AFTER_PRETRAIN") == "1"  # essai de vitesse : ni affinage ni score
# L'attention sans dropout. sdpa : F.scaled_dot_product_attention de PyTorch (sur XLA : la matrice T × T entière) ;
# flash : le noyau Pallas flash_attention de torch_xla (par blocs). Avec dropout (affinage) : toujours sdpa.
# splash : splash attention de torch_xla (noyau Pallas de MaxText, via JAX et torchax ; une seule puce).
ATTN_KERNEL = os.environ.get("ATTN_KERNEL", "sdpa")
assert ATTN_KERNEL in ("sdpa", "flash", "splash"), "ATTN_KERNEL : sdpa, flash ou splash"
SPLASH_BLOCK = int(os.environ.get("SPLASH_BLOCK", 512))  # splash : la taille de ses blocs (multiple de 128), comme train_jax.py
SPLASH_FUSED = os.environ.get("SPLASH_FUSED", "0") == "1"  # splash : le backward en un seul noyau (dq, dk, dv)
# Les leviers de vitesse à essayer (mode speed d'abord), éteints par défaut : voir use_layer_scan, speed et Prefetcher.
# Trois réécritures de l'attention de ModernGPT, mêmes calculs (aux arrondis bf16 près), d'après le profil du TPU :
# QKV_LAYOUT=1 : q, k, v produits directement en (B, têtes, T, D) et la sortie projetée depuis cette disposition
# (einsum), sans les copies de permute/transpose (~7 % du temps) ; QK_NORM=bf16 : la QK-norm moyenne en float32 mais
# multiplie en bf16, sans tenseur float32 (B, têtes, T, D) diffusé au backward ; ROPE=fused : x·cos + rot(x)·sin,
# cos et sin déjà convertis et doublés, au lieu de couper, multiplier quatre fois et recoller.
QKV_LAYOUT = os.environ.get("QKV_LAYOUT", "0") == "1"
QK_NORM = os.environ.get("QK_NORM", "f32")
ROPE = os.environ.get("ROPE", "split")
assert QK_NORM in ("f32", "bf16") and ROPE in ("split", "fused"), "QK_NORM : f32 ou bf16 ; ROPE : split ou fused"
LAYER_SCAN = os.environ.get("LAYER_SCAN", "0") == "1"  # les blocs de ModernGPT dans des boucles While d'XLA
BF16_WEIGHTS = os.environ.get("BF16_WEIGHTS", "0") == "1"  # mode speed : poids bf16, copie float32 pour l'optimiseur
PREFETCH = int(os.environ.get("PREFETCH", 0))  # lots préparés d'avance par un thread (0 : aucun)
SPEED_STEPS = int(os.environ.get("SPEED_STEPS", 12))  # mode speed : pas complets, les 3 premiers hors mesure
PROFILE = os.environ.get("PROFILE")  # mode speed : le dossier de la trace du profileur (XProf, TensorBoard)
PROFILE_STEPS = int(os.environ.get("PROFILE_STEPS", 3))
SPEED_PROBE = os.environ.get("SPEED_PROBE", "1") == "1"  # mode speed : la sonde numérique après la mesure (voir probe)
PROBE_DATA = os.environ.get("PROBE_DATA")  # sonde : un .bin de tokens uint16 (une validation) au lieu du motif répété
PEAK_TFLOPS = float(os.environ.get("PEAK_TFLOPS", 918))  # TPU v6e, bf16 : le dénominateur du MFU
SEED = 1337


def config() -> dict:
    return dict(vocab_size=8192, block_size=BLOCK, dropout=0.0, **MODELS[MODEL])


def same_dtype(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """q, k et v dans un même type, bf16 dès que l'un l'est. Les noyaux Pallas l'exigent (la sortie prend le type
    de q) et l'autocast ne s'applique pas à leurs appels. Sous l'autocast de train_tpu.py, q et k sortent déjà en
    bf16 (rms_norm rend le type de son entrée, RoPE aussi) : la conversion n'y coûte rien."""
    dtype = torch.bfloat16 if torch.bfloat16 in (q.dtype, k.dtype, v.dtype) else v.dtype
    return q.to(dtype), k.to(dtype), v.to(dtype)


def jax_on_tpu(switch: str) -> None:
    """Les noyaux Pallas (flash, splash) sont abaissés par JAX dans ce processus (jax.jit(...).lower()) : le
    backend de JAX doit y être le TPU. Sinon (TPU déjà pris, jax et libtpu incompatibles), JAX retombe sans bruit sur
    le CPU et l'erreur n'arrive qu'au premier forward, obscure (Pallas sur CPU). Vérifié tout de suite, après le garde
    de torch_xla (jax_import_guard : torch_xla prend le TPU avant JAX, comme le font ses noyaux)."""
    from torch_xla._internal.jax_workarounds import jax_env_context, jax_import_guard

    jax_import_guard()
    with jax_env_context():
        import jax

        backend, devices = jax.default_backend(), jax.devices()
    if backend != "tpu":
        raise RuntimeError(f"{switch} : le JAX de ce processus n'a pas le TPU (jax {jax.__version__}, backend {backend}, "
                           f"appareils {devices}) ; ses noyaux Pallas ne s'abaissent que pour le TPU")


def use_fast_attention() -> None:
    """QKV_LAYOUT, QK_NORM, ROPE : ModernAttention.forward remplacée par une version équivalente (mêmes poids, mêmes
    noms de paramètres : les checkpoints restent ceux de gpt.py)."""
    eps = torch.finfo(torch.float32).eps  # celui de F.rms_norm sur un tenseur float32

    def qk_norm(x):
        if QK_NORM == "bf16":
            return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)
        return gpt.rms_norm(x)

    tables = {}

    def rope(x, rotary):
        if ROPE == "split":
            return rotary(x)
        T, half = x.size(-2), x.size(-1) // 2
        key = (T, x.dtype, x.device)
        if key not in tables:  # cos et sin doublés (et sin signé), dans le type de x : une fois par forme
            cos, sin = rotary.cos[:T].to(x.dtype), rotary.sin[:T].to(x.dtype)
            tables[key] = (torch.cat((cos, cos), -1), torch.cat((-sin, sin), -1))
        cos2, sin2 = tables[key]
        swapped = torch.cat((x[..., half:], x[..., :half]), -1)  # (x2, x1)
        return x * cos2 + swapped * sin2  # (x1·cos − x2·sin, x2·cos + x1·sin), comme Rotary.forward

    def forward(self, x):
        B, T, C = x.shape
        H = self.n_head
        D = C // H
        if QKV_LAYOUT:
            dt = torch.get_autocast_dtype("xla") if torch.is_autocast_enabled("xla") else x.dtype
            w = self.qkv.weight.to(dt).view(3, H, D, C)
            xd = x.to(dt)
            q, k, v = (torch.einsum("btc,hdc->bhtd", xd, w[i]) for i in range(3))
        else:
            q, k, v = self.qkv(x).view(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k = rope(qk_norm(q), self.rotary), rope(qk_norm(k), self.rotary)
        p = self.attn_dropout.p if self.training else 0.0
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)
        if QKV_LAYOUT:
            out = torch.einsum("bhtd,chd->btc", y, self.proj.weight.to(y.dtype).view(C, H, D))
        else:
            out = self.proj(y.transpose(1, 2).reshape(B, T, C))
        return self.dropout(out)

    gpt.ModernAttention.forward = forward


def use_flash_attention() -> None:
    """F.scaled_dot_product_attention (que gpt.py appelle) remplacée par le noyau flash_attention de torch_xla,
    sauf avec dropout, sans masque causal ou hors TPU (la génération sur CPU de la fin d'une session) : celle de
    PyTorch, inchangée."""
    from torch_xla.experimental.custom_kernel import flash_attention

    jax_on_tpu("ATTN_KERNEL=flash")
    sdpa = torch.nn.functional.scaled_dot_product_attention

    def attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        if attn_mask is not None or dropout_p or not is_causal or kwargs or q.device.type != "xla":
            return sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)
        q, k, v = same_dtype(q, k, v)
        return flash_attention(q, k, v, causal=True, sm_scale=q.size(-1) ** -0.5)

    torch.nn.functional.scaled_dot_product_attention = attention


def use_splash_attention() -> None:
    """ATTN_KERNEL=splash : F.scaled_dot_product_attention remplacée par splash attention de torch_xla 2.9
    (torch_xla.experimental.splash_attention : le noyau Pallas de MaxText, appelé à travers JAX par call_jax, qui
    demande aussi torchax). Il saute les blocs entièrement dans le futur, s'il y en a : à SPLASH_BLOCK = 512 = BLOCK,
    une séquence tient dans un seul bloc (celui de la diagonale), tout le T × T est calculé puis masqué, et le gain ne
    peut venir que des accès à la HBM (de même pour flash, aux blocs de 512 fixés par torch_xla) ; à 256, un bloc sur
    4 est sauté, à 128, 6 sur 16. Mêmes exclusions que flash (dropout, masque, attention non causale, hors TPU :
    celle de PyTorch). Le noyau ne divise pas les scores par √d : q l'est avant (× 1/8 en medium, exact en bf16).
    torch_xla le lance dans un shard_map sur un maillage d'axes « data » et « fsdp » : ici un maillage d'une seule puce
    (JAX n'y insère alors aucune opération de partitionnement, pas besoin du mode SPMD). Ses blocs : SPLASH_BLOCK
    tokens (torch_xla les borne à la longueur des séquences ; le noyau de JAX veut des multiples de 128), comme
    SPLASH_BLOCK de train_jax.py ; SPLASH_FUSED=1 : le backward fusionné (dq, dk, dv d'un seul noyau)."""
    import importlib.util

    import torch_xla.runtime as xr
    from torch_xla.distributed.spmd import Mesh
    from torch_xla.experimental.splash_attention import SplashAttentionConfig, splash_attention

    missing = [name for name in ("jax", "torchax") if importlib.util.find_spec(name) is None]
    if missing:  # call_jax de torch_xla 2.9 s'arrête sans eux
        raise RuntimeError(f"ATTN_KERNEL=splash : {' et '.join(missing)} manquant (pip install torch_xla[pallas] torchax)")
    if WORLD > 1 or xr.global_runtime_device_count() != 1:
        raise RuntimeError("ATTN_KERNEL=splash : une seule puce (le maillage du noyau compte un seul appareil)")
    block = min(SPLASH_BLOCK, BLOCK)  # ce que torch_xla passe au noyau
    if block % 128 or BLOCK % block:  # splash_attention_kernel.py : NotImplementedError ou ValueError sinon
        raise RuntimeError(f"ATTN_KERNEL=splash : SPLASH_BLOCK={SPLASH_BLOCK}, il faut un multiple de 128 qui divise {BLOCK}")
    jax_on_tpu("ATTN_KERNEL=splash")
    b = SPLASH_BLOCK
    config = SplashAttentionConfig(
        sa_block_q=b, sa_block_kv=b, sa_block_kv_compute=b, sa_block_q_dkv=b, sa_block_kv_dkv=b,
        sa_block_kv_dkv_compute=b, sa_block_q_dq=b, sa_block_kv_dq=b, sa_use_fused_bwd_kernel=SPLASH_FUSED,
        mesh=str(Mesh(np.array([0]), (1, 1), ("data", "fsdp"))),
        qkv_partition_spec=(("data", "fsdp"), None, None, None),
        segment_ids_partition_spec=(("data", "fsdp"), None)).to_json()
    sdpa = torch.nn.functional.scaled_dot_product_attention

    def attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        if attn_mask is not None or dropout_p or not is_causal or kwargs or q.device.type != "xla":
            return sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)
        q, k, v = same_dtype(q, k, v)  # (B, n_head, T, head_dim), la disposition qu'attend le noyau
        return splash_attention(q * q.size(-1) ** -0.5, k, v, config, causal=True)

    torch.nn.functional.scaled_dot_product_attention = attention


def use_layer_scan() -> None:
    """LAYER_SCAN=1 : gpt.ModernGPT.forward fait passer ses blocs par scan de torch_xla (torch_xla.experimental.scan,
    le moteur de scan_layers) : une boucle While d'XLA dont le corps, un bloc, est tracé par AOTAutograd une seule
    fois (is_fn_pure : avant et arrière gardés en cache) ; à chaque micro-lot, Python ne retrace plus 24 blocs avant
    et arrière, seulement un corps de boucle par moitié du U, et le compilateur ne voit qu'un bloc par boucle.

    scan_layers ne rend que la sortie du dernier bloc : les connexions en U demandent la sortie de chaque bloc de la
    première moitié, puis ajoutent à l'entrée de chaque bloc de la seconde la sortie symétrique (skips.pop() : de la
    dernière à la première) × son poids appris. D'où deux scan, avec la mécanique de scan_layers (functional_call
    sur une copie d'un bloc, poids et buffers des blocs empilés sur un axe « couche ») : la première moitié rend
    aussi la sortie de chaque bloc (empilées), la seconde reçoit avec ses poids ces sorties dans l'ordre inverse et
    les poids des connexions. Le bloc du milieu (nombre de couches impair) reste hors boucle. Même calcul que
    gpt.ModernGPT.forward ; hors TPU (génération sur CPU), avec un dropout non nul (affinage : le corps mis en cache
    ne distingue pas entraînement et évaluation) ou sans gradient (evaluate, full_loss : le cache de scan ignore le
    mode du gradient, un corps tracé sous no_grad rendrait ensuite des gradients vides pour tous les blocs ; et la
    boucle d'entraînement garderait les activations pour rien) : le forward de gpt.py, inchangé. Validation : la
    sonde du mode speed (probe : loss, gradients, et les types des dot du graphe, corps des While compris, pour
    s'assurer que l'autocast bf16 tient dans le corps tracé par AOTAutograd)."""
    import copy

    from torch_xla.experimental.scan import scan

    forward = gpt.ModernGPT.forward

    def stacked(blocks) -> tuple[dict, dict]:
        """Poids et buffers des blocs, empilés sur un axe « couche » (comme scan_layers)."""
        params = [dict(b.named_parameters()) for b in blocks]
        buffers = [dict(b.named_buffers()) for b in blocks]
        return ({k: torch.stack([p[k] for p in params]) for k in params[0]},
                {k: torch.stack([b[k] for b in buffers]) for k in buffers[0]})

    def scanned(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        n = len(self.blocks)
        half = n // 2
        if (idx.device.type != "xla" or half < 1 or not torch.is_grad_enabled()
                or any(isinstance(m, nn.Dropout) and m.p > 0 for m in self.modules())):
            return forward(self, idx, targets)
        functions = self.__dict__.get("_layer_scan")  # hors de nn.Module : ni paramètre ni sous-module
        if functions is None:  # créées une fois : le cache de scan (is_fn_pure) se règle sur ces fonctions
            template = copy.deepcopy(self.blocks[0])  # la structure d'un bloc ; ses poids sont remplacés à chaque appel

            def first(x, layer):
                x = torch.func.functional_call(template, layer, (x,), strict=True)
                return x, x.clone()  # la sortie, gardée pour la connexion en U (une copie : pas deux sorties identiques)

            def second(x, layer):
                params, buffers, skip, weight = layer
                x = x + weight * skip  # la connexion en U, comme gpt.ModernGPT.forward
                return torch.func.functional_call(template, (params, buffers), (x,), strict=True), None

            functions = self.__dict__["_layer_scan"] = (first, second)
        first, second = functions
        x = gpt.rms_norm(self.embed(idx))
        x, skips = scan(first, x, stacked(self.blocks[:half]), is_fn_pure=True)
        for block in self.blocks[half : n - half]:
            x = block(x)
        params, buffers = stacked(self.blocks[n - half :])
        x, _ = scan(second, x, (params, buffers, skips.flip(0), self.skip_weights), is_fn_pure=True)
        logits = self.lm_head(gpt.rms_norm(x)).float()  # la suite : celle de gpt.ModernGPT.forward
        logits = self.softcap * torch.tanh(logits / self.softcap)
        if targets is None:
            return logits, None
        B, T = idx.shape
        return logits, torch.nn.functional.cross_entropy(logits.view(B * T, -1), targets.reshape(B * T))

    gpt.ModernGPT.forward = scanned


def make_model() -> nn.Module:
    return (gpt.ModernGPT if ARCH == "modern" else gpt.GPTLanguageModel)(**config())


def log(*args) -> None:
    if RANK == 0:
        print(time.strftime("%H:%M:%S"), *args, flush=True)


def find_input(name: str) -> Path | None:
    """Un fichier d'une session précédente ou de la préparation (en entrée du notebook) : le plus récent."""
    dirs = ["/kaggle/input"] + ([os.environ["RESUME_DIR"]] if os.environ.get("RESUME_DIR") else [])
    found = sorted(f for d in dirs for f in glob.glob(f"{d}/**/{name}", recursive=True))
    return Path(found[-1]) if found else None


# --- AdamW sans constantes figées ------------------------------------------------------------------------


class AdamW:
    """AdamW (Loshchilov & Hutter), le learning rate en tenseur sur le TPU : il change à chaque pas sans
    recompiler le graphe. Même calcul que torch.optim.AdamW (betas, eps, weight decay découplé)."""

    def __init__(self, groups: list[dict], device, betas=(0.9, 0.95), eps=1e-8) -> None:
        self.groups, self.device, self.betas, self.eps = groups, device, betas, eps
        self.params = [p for g in groups for p in g["params"]]
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = torch.zeros((), device=device)

    def step(self, lr: float) -> None:
        lr = torch.tensor(lr, dtype=torch.float32).to(self.device)  # une donnée, pas une constante du graphe
        b1, b2 = self.betas
        self.t += 1
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        i = 0
        for group in self.groups:
            for p in group["params"]:
                m, v, g = self.m[i], self.v[i], p.grad
                i += 1
                if g is None:
                    continue
                if group["weight_decay"]:
                    p.data.mul_(1 - lr * group["weight_decay"])
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                p.data.sub_(lr * (m / c1) / ((v / c2).sqrt() + self.eps))

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def state_dict(self) -> dict:
        return {"m": [x.cpu() for x in self.m], "v": [x.cpu() for x in self.v], "t": self.t.cpu()}

    def load_state_dict(self, state: dict) -> None:
        self.m = [x.to(self.device) for x in state["m"]]
        self.v = [x.to(self.device) for x in state["v"]]
        self.t = state["t"].to(self.device)


def make_optimizer(model: nn.Module, weight_decay: float, device) -> AdamW:
    """Weight decay sur les matrices seulement (pas sur les biais ni les LayerNorm), comme le notebook."""
    params = [p for p in model.parameters() if p.requires_grad]
    return AdamW([{"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
                  {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}], device)


# --- Données ----------------------------------------------------------------------------------------------


def batch_from(data: np.ndarray, n: int, generator: np.random.Generator, device) -> tuple[torch.Tensor, ...]:
    """`n` morceaux de BLOCK tokens tirés au hasard, avec leurs cibles décalées d'un cran."""
    starts = generator.integers(0, len(data) - BLOCK - 1, size=n)
    chunk = torch.from_numpy(np.stack([data[s : s + BLOCK + 1] for s in starts]).astype(np.int64))
    return chunk[:, :-1].contiguous().to(device), chunk[:, 1:].contiguous().to(device)


def step_batch(parts: list[tuple[np.ndarray, float]], weights: np.ndarray, accum: int, generator: np.random.Generator,
               device) -> tuple[torch.Tensor, ...]:
    """Les `accum` micro-lots d'un pas d'un coup, (accum, MICRO_BATCH, BLOCK), envoyés au TPU en une fois : les mêmes
    tirages, dans le même ordre, que la boucle de `run_stage` (le corpus du micro-lot selon le mélange, puis ses
    morceaux comme `batch_from`). Des données sur le TPU, pas un calcul en attente : gradient_accumulation l'exige."""
    chunks = []
    for _ in range(accum):
        data = parts[generator.choice(len(parts), p=weights)][0]
        starts = generator.integers(0, len(data) - BLOCK - 1, size=MICRO_BATCH)
        chunks.append(np.stack([data[s : s + BLOCK + 1] for s in starts]))
    chunk = torch.from_numpy(np.stack(chunks).astype(np.int64))
    return chunk[..., :-1].contiguous().to(device), chunk[..., 1:].contiguous().to(device)


class Prefetcher:
    """PREFETCH=n : les lots préparés par un thread, `n` d'avance, déjà copiés sur le TPU quand le pas les demande
    (comme le ParallelLoader de torch_xla, qui copie aussi depuis un thread, mais sans son mark_step à chaque lot).
    `make` tire un lot exactement comme la boucle de run_stage (même générateur, mêmes tirages, dans le même ordre) :
    seul le moment du tirage change, pas les lots. Le thread est le seul à se servir du générateur."""

    def __init__(self, make, depth: int) -> None:
        import queue
        import threading

        self.full = queue.Full
        self.queue, self.stop = queue.Queue(maxsize=max(1, depth)), threading.Event()
        self.thread = threading.Thread(target=self._loop, args=(make,), daemon=True)
        self.thread.start()

    def _put(self, item) -> None:
        while not self.stop.is_set():
            try:
                self.queue.put(item, timeout=0.5)
                return
            except self.full:
                pass

    def _loop(self, make) -> None:
        try:
            while not self.stop.is_set():
                self._put(make())
        except BaseException as problem:  # transmis au thread principal, au lot suivant
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


def flops_per_token() -> float:
    """6 N (matrices des blocs) + 6 V·C (tête) + 12 L·T·C (attention) : la formule de PaLM et nanoGPT, la même que
    flops_per_token de train_jax.py (les MFU des deux moteurs se comparent)."""
    C, L = MODELS[MODEL]["n_embd"], MODELS[MODEL]["n_layer"]
    return 6 * 12 * L * C * C + 6 * config()["vocab_size"] * C + 12 * L * BLOCK * C


def xla_metrics() -> dict:
    """Les compteurs de torch_xla, cumulés depuis le lancement : compilations, exécutions et transferts (nombre,
    secondes au total), opérations retombées sur le CPU (compteurs « aten:: » : nom → nombre d'appels, chacun un
    aller-retour TPU → CPU → TPU ; un .item() en est un, aten::_local_scalar_dense). speed n'en garde que la hausse
    pendant les pas mesurés. Un diagnostic : vide s'il manque."""
    out: dict = {}
    try:
        import torch_xla.debug.metrics as met

        for name in ("CompileTime", "ExecuteTime", "TransferToDeviceTime", "TransferFromDeviceTime"):
            data = met.metric_data(name)
            if data:  # (échantillons, total en nanosecondes, derniers échantillons)
                out[name] = [int(data[0]), round(float(data[1]) / 1e9, 3)]
        for name in ("CachedCompile", "UncachedCompile", "MarkStep"):
            value = met.counter_value(name)
            if value is not None:
                out[name] = int(value)
        out["cpu_fallback"] = {n: int(met.counter_value(n) or 0) for n in met.counter_names() if n.startswith("aten::")}
    except Exception:  # noqa: BLE001 — un diagnostic, jamais une raison de s'arrêter
        pass
    return out


def compile_count() -> int | None:
    """Le nombre de compilations XLA de ce processus (None si la mesure manque). En mode scan, il ne doit plus
    bouger après les premiers pas : sinon le graphe du While change d'un pas à l'autre et tout est recompilé."""
    try:
        import torch_xla.debug.metrics as met

        data = met.metric_data("CompileTime")
        return int(data[0]) if data else 0
    except Exception:  # noqa: BLE001 — un diagnostic, jamais une raison de s'arrêter
        return None


def load_data(tokenizer) -> tuple[list[tuple[np.ndarray, float]], np.ndarray, dict, dict]:
    """Le corpus général (Wikipédia + web, pondérés par leur taille), sa validation, et les Simpson."""
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
        raise FileNotFoundError("pas de corpus : ajouter la sortie de prepare_data.py en entrée du notebook")
    log("corpus général : " + "   ".join(f"{len(d) / 1e9:.2f} G tokens" for d, _ in general))
    return general, np.concatenate(vals), tokens, simpsons


# --- Entraînement -----------------------------------------------------------------------------------------


def run_stage(name: str, model: nn.Module, parts: list[tuple[np.ndarray, float]], val_data: np.ndarray, cfg: dict,
              hours: float, tokenizer, device, total_hours: float | None = None, state_in: Path | None = None,
              state_out: Path | None = None, max_tokens: float | None = None, save_every: float = 0,
              accum_mode: str = "loop") -> None:
    """Entraîne `hours` heures (planifiées en pas) ; garde le meilleur modèle dans OUT/checkpoints/{name}.

    `max_tokens` : un budget en tokens à la place des heures, reprise comprise (le planning du learning rate
    suit alors les tokens vus), pour comparer des machines de vitesses différentes ou en changer en cours de route.
    Un fichier OUT/STOP arrête l'étape proprement, état sauvegardé (pour reprendre sur une autre machine).
    `accum_mode` : l'accumulation des gradients, loop ou scan (voir ACCUM_MODE).
    """
    import torch_xla.core.xla_model as xm

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = cfg["dropout"]
    model.train()
    if cfg.get("muon"):  # modded-nanoGPT : Muon pour les matrices des blocs, Adam pour le reste
        optimizers = gpt.modern_optimizers(model, **{k: cfg[k] * LR_SCALE for k in ("muon_lr", "embed_lr", "head_lr", "scalar_lr")}, normuon=cfg["normuon"])
        clip = None
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizers = [gpt.AdamW([{"params": [p for p in params if p.dim() >= 2], "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]},
                                 {"params": [p for p in params if p.dim() < 2], "lr": cfg["lr"], "weight_decay": 0.0}])]
        clip = 1.0
    accum = max(1, TOKENS_PER_STEP // (MICRO_BATCH * BLOCK * WORLD))
    tokens_per_step = accum * MICRO_BATCH * BLOCK * WORLD
    weights = np.array([w for _, w in parts]) / sum(w for _, w in parts)
    generator = np.random.default_rng(SEED * 100 + RANK + 1000 * SESSION + hash(name) % 1000)
    checkpoint = OUT / "checkpoints" / f"{name}.safetensors"
    best, step, tokens_seen, seconds_before = float("inf"), 0, 0, 0.0
    if state_in is not None:
        state = torch.load(state_in, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        for o, o_state in zip(optimizers, state["optimizer"]):
            o.load_state_dict(o_state)
        best, step, tokens_seen, seconds_before = state["best"], state["step"], state["tokens"], state["seconds"]
        log(f"reprise : pas {step:,}, {tokens_seen / 1e9:.2f} G tokens, {seconds_before / 3600:.2f} h faites")
    total = (total_hours or hours) * 3600
    start_step, start = step, time.time()
    previous_lr = None  # le multiplicateur du pas précédent (LR_RISE)
    sec_per_step = None  # mesurée après la compilation, moyennée entre les cœurs
    planned = step + 10**9  # le dernier pas de la session, replanifié à chaque mesure
    if max_tokens:  # les pas qui restent pour atteindre le budget (une reprise en a déjà fait une partie)
        planned = step + int(max(0.0, max_tokens - tokens_seen) // tokens_per_step)
    next_eval, last_eval = step + 20, step
    scan = accum_mode == "scan"
    log(f"=== {name} : {hours:g} h · {accum} micro-lots × {MICRO_BATCH} × {BLOCK} × {WORLD} cœurs "
        f"= {tokens_per_step:,} tokens par pas" + (" · micro-lots dans une boucle While d'XLA (scan)" if scan else ""))
    loss_sum, loss_count = torch.zeros((), device=device), 0
    last_progress, progress_tokens = time.time(), tokens_seen
    compiles = None  # mode scan : le dernier nombre de compilations XLA annoncé

    if scan:
        from torch_xla.experimental.gradient_accumulation import gradient_accumulation

        def train_step(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Un micro-lot, tracé une fois puis répété par le While : la loss du micro-lot, NON divisée.
            gradient_accumulation (torch_xla 2.9) la divise elle-même par le nombre de micro-lots avant le backward :
            sa somme est la loss moyenne du pas, ses gradients ceux de la boucle Python. Fonction pure (exigé) : rien
            n'est modifié en place, ni les lots ni les poids ; les gradients sont relevés par gradient_accumulation."""
            with AUTOCAST():
                return model(x, y)[1].float()  # float32 : l'accumulateur de la loss du While est un f32[]

    prefetch = None  # PREFETCH=n : les lots (ou, en mode scan, ceux d'un pas) tirés d'avance par un thread
    if PREFETCH:
        prefetch = Prefetcher((lambda: step_batch(parts, weights, accum, generator, device)) if scan else (
            lambda: batch_from(parts[generator.choice(len(parts), p=weights)][0], MICRO_BATCH, generator, device)), PREFETCH)

    def save_state() -> None:
        """Poids, AdamW et avancement, pour la session suivante (le cœur 0 seulement : ils sont identiques)."""
        if state_out is not None and RANK == 0:  # écrit à côté puis renommé : jamais de fichier à moitié écrit
            torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}, "optimizer": [o.state_dict() for o in optimizers],
                        "best": best, "step": step, "tokens": tokens_seen,
                        "seconds": seconds_before + (step - start_step) * (sec_per_step or 0)}, f"{state_out}.tmp")
            os.replace(f"{state_out}.tmp", state_out)
        xm.rendezvous("save")

    last_save = time.time()

    def measure(final: bool = False) -> None:
        """Validation ; toutes les `save_every` secondes (et à la fin), l'état, et le modèle s'il est le meilleur."""
        nonlocal best, loss_sum, loss_count, last_eval, last_save
        val = evaluate(model, val_data, device)
        train_loss = loss_sum.item() / max(1, loss_count)
        loss_sum, loss_count, last_eval = torch.zeros((), device=device), 0, step
        note = ""
        if val < best:
            best, note = val, "  ← meilleur"
        if final or time.time() - last_save >= save_every:
            if val <= best or (final and save_every):  # pré-entraînement : le modèle de la fin, dans tous les cas
                if RANK == 0:
                    gpt.save_checkpoint(checkpoint, model, config() | {"dropout": cfg["dropout"]} | ({"arch": "modern"} if ARCH == "modern" else {}), tokenizer,
                                        step=step, val_loss=val, tokens=tokens_seen, hyperparameters={**cfg, "model": MODEL})
                xm.rendezvous("checkpoint")
                note += ", sauvegardé"
            save_state()
            last_save = time.time()
        log(f"{name}  pas {step:>6,}/{planned:,}   train {train_loss:.4f}   val {val:.4f}   lr ×{lr:.3f}   "
            f"{tokens_seen / 1e9:.2f} G tokens   {tokens_per_step / sec_per_step if sec_per_step else 0:,.0f} tokens/s   "
            f"{(time.time() - start) / 3600:.2f} h{note}")

    while step < planned:
        if max_tokens:
            progress = tokens_seen / max_tokens
        else:
            progress = (seconds_before + (step - start_step) * (sec_per_step or 0)) / total
            if progress >= 1:
                break
        warm = cfg["warmup"]
        if cfg.get("muon"):
            lr = gpt.wsd_schedule(progress, warm, cfg["cooldown"])  # un multiplicateur des learning rates
        else:
            lr = min(1.0, (progress + 1e-3) / warm) if progress < warm else (
                cfg["min_lr"] + 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - warm) / (1 - warm)))) * (cfg["lr"] - cfg["min_lr"])) / cfg["lr"]
        if previous_lr is not None and progress >= warm:  # budget relevé en cours de descente : remontée par paliers
            lr = min(lr, previous_lr + LR_RISE)
        previous_lr = lr
        if scan:  # tout le pas en un graphe : les micro-lots dans le While, puis l'optimiseur, puis un mark_step
            x, y = step_batch(parts, weights, accum, generator, device) if prefetch is None else prefetch.next()
            loss_sum += gradient_accumulation(train_step, (x, y), model)  # les gradients du pas dans p.grad
        else:
            for micro in range(accum):
                if prefetch is None:
                    data = parts[generator.choice(len(parts), p=weights)][0]
                    x, y = batch_from(data, MICRO_BATCH, generator, device)
                else:
                    x, y = prefetch.next()
                with AUTOCAST():
                    loss = model(x, y)[1] / accum
                loss.backward()
                loss_sum += loss.detach()
                xm.mark_step()
        if WORLD > 1:  # moyenne des gradients des 8 cœurs
            xm.all_reduce(xm.REDUCE_SUM, [p.grad for p in model.parameters() if p.grad is not None], scale=1.0 / WORLD)
        if clip:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        mult = torch.tensor(lr, dtype=torch.float32).to(device)  # une donnée : pas de recompilation quand il change
        for o in optimizers:
            o.step(mult)
        for o in optimizers:
            o.zero_grad()
        xm.mark_step()
        step += 1
        loss_count += 1
        tokens_seen += tokens_per_step
        if step == start_step + 5:
            t5 = time.time()  # après la compilation
        if time.time() - last_progress >= PROGRESS_EVERY_S:  # entre deux validations, de quoi suivre en direct
            speed = (tokens_seen - progress_tokens) / (time.time() - last_progress)
            log(f"{name}  avancement  pas {step:,}/{planned:,}   train {loss_sum.item() / max(1, loss_count):.4f}   "
                f"lr ×{lr:.3f}   {tokens_seen / 1e9:.3f} G tokens   {speed:,.0f} tokens/s")
            last_progress, progress_tokens = time.time(), tokens_seen
        if step % 10 == 0 and xm.mesh_reduce("stop", int((OUT / "STOP").exists()), max):  # changement de machine
            log(f"{name} : arrêt demandé au pas {step:,} ({tokens_seen / 1e9:.3f} G tokens), état sauvegardé")
            save_state()
            if prefetch is not None:  # le thread arrêté avant la sortie du processus
                prefetch.close()
            raise SystemExit(3)
        budget_file = OUT / f"{name}.tokens"  # le suivi peut ajuster le budget en cours de route (sur le plateau
        if max_tokens and step % 10 == 0 and budget_file.exists():  # du planning, ça ne change rien d'autre)
            new_total = xm.mesh_reduce("budget", float(budget_file.read_text()), max)
            if new_total != max_tokens:
                max_tokens = new_total
                planned = step + int(max(0.0, max_tokens - tokens_seen) // tokens_per_step)
                log(f"{name} : budget ajusté à {max_tokens / 1e9:.2f} G tokens (fin prévue au pas {planned:,})")
        if step == start_step + 25 or step >= next_eval:
            if step == start_step + 25:
                sec_per_step = xm.mesh_reduce("sec", (time.time() - t5) / 20, np.mean)
            else:
                sec_per_step = xm.mesh_reduce("sec", (time.time() - start) / (step - start_step), np.mean)
            if not max_tokens:  # la session : `hours` heures, compilation comprise ; le cosinus : `total_hours`
                planned = start_step + int((hours * 3600 - (time.time() - start)) / sec_per_step) + (step - start_step)
                planned = xm.mesh_reduce("planned", planned, min)
            measure()
            next_eval = step + max(1, int(EVAL_EVERY_S / sec_per_step))
            # Mode scan : les compilations XLA, stables après le pas 25 (après une mesure, le premier pas repart d'une
            # loss à zéro : une variante du graphe, compilée une fois). Au-delà, une hausse = un graphe recompilé.
            if scan and (count := compile_count()) is not None and count != compiles:
                late = compiles is not None and step > start_step + 25
                log(f"{name} : {count} compilations XLA depuis le lancement"
                    + (f" (+{count - compiles} depuis la mesure précédente : recompilation)" if late else ""))
                compiles = count
    if prefetch is not None:
        prefetch.close()
    if step > last_eval:  # la fin du planning, learning rate au plus bas : souvent le meilleur moment
        measure(final=True)
    save_state()
    log(f"{name} : fin au pas {step:,}, meilleure val {best:.4f}")


@torch.no_grad()
def evaluate(model: nn.Module, data: np.ndarray, device, n_batches: int = 10) -> float:
    """Loss moyenne sur des morceaux de validation tirés avec une graine fixe, moyennée entre les cœurs."""
    import torch_xla.core.xla_model as xm

    model.eval()
    generator = np.random.default_rng(SEED + RANK)
    total = torch.zeros((), device=device)
    for _ in range(n_batches):
        x, y = batch_from(data, MICRO_BATCH, generator, device)
        with AUTOCAST():
            total += model(x, y)[1].float()
        xm.mark_step()
    model.train()
    return xm.mesh_reduce("val", total.item() / n_batches, np.mean)


def full_loss(model: nn.Module, data: np.ndarray, device) -> float:
    """Loss sur tout un texte, en morceaux de BLOCK qui se suivent (comme `full_loss` du notebook)."""
    import torch_xla.core.xla_model as xm

    model.eval()
    n = (len(data) - 1) // BLOCK
    n -= n % MICRO_BATCH  # des lots pleins : une seule forme, une seule compilation
    x = torch.from_numpy(data[: n * BLOCK].astype(np.int64)).view(n, BLOCK)
    y = torch.from_numpy(data[1 : n * BLOCK + 1].astype(np.int64)).view(n, BLOCK)
    total = torch.zeros((), device=device)
    with torch.no_grad():
        for i in range(0, n, MICRO_BATCH):
            with AUTOCAST():
                total += model(x[i : i + MICRO_BATCH].to(device), y[i : i + MICRO_BATCH].to(device))[1].float() * MICRO_BATCH
            xm.mark_step()
    return total.item() / n


def train_session(index: int) -> None:
    """Une session : pré-entraînement (reprise comprise), affinage sur les Simpson, score et exemples."""
    init_worker()
    device = DEVICE
    torch.manual_seed(SEED)
    tokenizer = gpt.tokenizer_from_state(json.loads((INPUT / "tokenizer.json").read_text(encoding="utf-8")))
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    general, general_val, tokens, simpsons = load_data(tokenizer)
    model = make_model().to(device)
    log(f"{MODEL} : {sum(p.numel() for p in model.parameters()):,} paramètres · {WORLD} cœurs TPU")

    name = f"dialogues_bpe8192_gpt2{'' if MODEL == 'small' else '_' + MODEL}{'_modded' if ARCH == 'modern' else ''}"
    if INIT:  # affinage seul, à partir d'un modèle pré-entraîné ailleurs (par exemple sur les GPU de Kaggle)
        model.load_state_dict(gpt.load_checkpoint(Path(INIT), "cpu")[0].state_dict())
        log(f"poids pré-entraînés : {INIT}")
        name += f"_s{SESSION}_ft_tpu_{FT_OPTIM}"
    else:
        run_stage("pretrain_gpt2" + ("_modded" if ARCH == "modern" else ""), model, general, general_val,
                  PRETRAIN_MODERN if ARCH == "modern" else PRETRAIN, HOURS_PRETRAIN, tokenizer, device,
                  total_hours=TOTAL_PRETRAIN_HOURS, state_in=find_input("pretrain_state.pt"),
                  state_out=OUT / "pretrain_state.pt", max_tokens=PRETRAIN_TOKENS or None, save_every=SAVE_EVERY_S,
                  accum_mode=ACCUM_MODE)
        if STOP_AFTER_PRETRAIN:  # essai de vitesse : l'état est sauvegardé, on s'arrête là (code 0)
            log("STOP_AFTER_PRETRAIN : fin après le pré-entraînement, sans affinage ni score")
            return
        name += f"_tpu_s{SESSION}"
    finetune = FINETUNE_MODERN if FT_OPTIM == "normuon" else FINETUNE
    replay = finetune["replay"]
    general_parts = [(d, w * replay / sum(w for _, w in general)) for d, w in general]
    run_stage(name, model, [(tokens["train"], 1 - replay)] + general_parts, tokens["val"], finetune, HOURS_FINETUNE,
              tokenizer, device, max_tokens=FT_TOKENS or None, accum_mode=FT_ACCUM_MODE)

    if RANK == 0:  # le score du meilleur modèle affiné, comme à la fin du notebook
        best, _, info = gpt.load_checkpoint(OUT / "checkpoints" / f"{name}.safetensors", "cpu")
        model.load_state_dict(best.state_dict())
        result = {"session": SESSION, "model": MODEL, "step": info["step"], "params": sum(p.numel() for p in best.parameters())}
        for part in ("val", "test"):
            loss = full_loss(model, tokens[part], device)
            result[f"{part}_loss"] = loss
            result[f"{part}_bits_per_char"] = loss / math.log(2) / (len(simpsons[part]) / len(tokens[part]))
        (OUT / "checkpoints" / f"{name}.json").write_text(json.dumps(result, indent=2))
        (OUT / "resultat.json").write_text(json.dumps(result, indent=2))
        log("résultat : " + json.dumps(result))
        lines = []
        for me, who, message in [("Bart", "Homer", "Papa, je peux avoir un poney ?"),
                                 ("Marge", "Homer", "Homer, tu as encore bu chez Moe ?"),
                                 ("Homer", "Moe", "Moe, une bière s'il te plaît."),
                                 ("Homer", "Burns", "Monsieur Burns, je voudrais une augmentation.")]:
            text = gpt.reply(best, tokenizer, f"{me}: {message}\n{who}:", who, "cpu")  # génération sur CPU
            lines.append(f"{me}: {message}\n{who}: {text}\n")
        (OUT / "exemples.txt").write_text("\n".join(lines), encoding="utf-8")
        log("exemples :\n" + "\n".join(lines))


# --- Essai : vitesse de chaque taille, sur un cœur ----------------------------------------------------------


def bench() -> dict:
    """Pour chaque taille et chaque micro-lot : quelques pas d'entraînement sur un cœur, tokens/s ou erreur."""
    global MODEL, MICRO_BATCH
    import torch_xla.core.xla_model as xm

    init_worker()
    results = []
    for MODEL, MICRO_BATCH in [("small", 16), ("medium", 8), ("medium", 4)]:
        entry = {"model": MODEL, "micro_batch": MICRO_BATCH}
        try:
            torch.manual_seed(0)
            model = make_model().to(DEVICE)
            optimizer = make_optimizer(model, 0.1, DEVICE)
            data = np.random.default_rng(0).integers(0, 8192, size=2_000_000).astype(np.uint16)
            generator = np.random.default_rng(0)
            t0 = time.time()
            for i in range(13):
                if i == 3:
                    xm.wait_device_ops()
                    t3 = time.time()
                    entry["compile_s"] = round(t3 - t0, 1)
                x, y = batch_from(data, MICRO_BATCH, generator, DEVICE)
                with AUTOCAST():
                    loss = model(x, y)[1]
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(3e-4)
                optimizer.zero_grad()
                xm.mark_step()
            xm.wait_device_ops()
            entry["tokens_per_s_per_core"] = round(10 * MICRO_BATCH * BLOCK / (time.time() - t3))
            entry["loss"] = round(loss.item(), 3)
            entry["ok"] = True
            del model, optimizer
            gc.collect()
        except Exception as error:  # mémoire du TPU dépassée, le plus souvent
            entry["ok"], entry["error"] = False, f"{type(error).__name__}: {str(error)[:300]}"
        print(json.dumps(entry), flush=True)
        results.append(entry)
    return {"bench": results}


# --- Laboratoire de vitesse : le pas d'entraînement réel, mesuré ------------------------------------------------


def probe() -> dict:
    """La sonde du mode speed, son témoin numérique : un forward (sous l'autocast bf16) et un backward (hors autocast,
    comme run_stage), sans pas d'optimiseur, sur un modèle à part et un lot fixe. La loss et les poids des pas de la
    mesure n'y suffisent pas : la tête et les projections de sortie des blocs de ModernGPT partent de zéro (loss =
    ln 8192 et gradients des blocs nuls aux premiers pas), le corpus de Zipf n'a pas de contexte, et NorMuon fixe la
    norme de ses mises à jour ; un noyau d'attention faux, une connexion en U mal placée ou un autocast perdu passaient.

    Le modèle : celui de MODEL et ARCH (graine SEED), ses poids nuls au départ (proj.weight, lm_head.weight) tirés de
    N(0, 0,02) (graine fixe, sur CPU), les poids des connexions en U tous différents (0,5 → 1,5) ; ou ceux d'un vrai
    modèle, INIT=<…>.safetensors. En bf16 avec BF16_WEIGHTS, comme celui de la mesure. Le lot, 8 séquences quel que
    soit MICRO_BATCH (les lignes se comparent d'un micro-lot à l'autre) : chacune répète un motif de 32 tokens (la
    suite se lit plus haut : l'attention et le masque causal comptent), ou PROBE_DATA=<…>.bin (tokens uint16, une
    validation de prepare_data.py) : des morceaux à positions fixes.

    Rend la loss (exacte : repr du float) ; les normes des gradients du premier bloc (qkv, proj), du MLP du dernier, de
    l'embedding, de la tête, et de tous (all) ; le gradient de chaque poids des connexions en U ; les produits
    matriciels (dot) du graphe HLO, corps des boucles While compris, comptés par types (sortie←entrées). Lecture :
    loss et gradients à ~1e-2 près (relatif) de la ligne de référence (ATTN_KERNEL=sdpa, sans levier) pour flash,
    splash et LAYER_SCAN ; probe_dots : les mêmes types que la référence, et pas plus de dot float32 (avec LAYER_SCAN,
    un seul corps par boucle est écrit : moins de dot, mêmes types)."""
    import re

    import torch_xla
    import torch_xla.core.xla_builder as xb
    import torch_xla.core.xla_model as xm

    t0 = time.time()
    torch.manual_seed(SEED)
    model = make_model()
    if INIT:  # les poids d'un vrai modèle
        model.load_state_dict(gpt.load_checkpoint(Path(INIT), "cpu")[0].state_dict())
    else:
        generator = torch.Generator().manual_seed(SEED)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name.endswith("proj.weight") or name == "lm_head.weight":  # nuls au départ dans ModernGPT
                    param.copy_(torch.randn(param.shape, generator=generator) * 0.02)
            if isinstance(getattr(model, "skip_weights", None), nn.Parameter):  # un poids mal placé se voit
                model.skip_weights.copy_(torch.linspace(0.5, 1.5, model.skip_weights.numel()))
    model = model.to(DEVICE)
    if BF16_WEIGHTS:
        model = model.to(torch.bfloat16)
    model.train()
    if PROBE_DATA:
        data = np.memmap(PROBE_DATA, dtype=np.uint16, mode="r")
        starts = np.linspace(0, len(data) - BLOCK - 1, 8, endpoint=False).astype(np.int64)
        chunk = np.stack([data[s : s + BLOCK + 1] for s in starts])
    else:
        motifs = np.random.default_rng(SEED).integers(0, config()["vocab_size"], size=(8, 32))
        chunk = np.tile(motifs, (1, BLOCK // 32 + 1))[:, : BLOCK + 1]
    chunk = torch.from_numpy(chunk.astype(np.int64))
    x, y = chunk[:, :-1].contiguous().to(DEVICE), chunk[:, 1:].contiguous().to(DEVICE)
    with AUTOCAST():
        loss = model(x, y)[1]
    loss.backward()
    grads = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    out: dict = {"probe_init": INIT or "seed", "probe_data": PROBE_DATA or "motif"}
    try:  # le graphe avant son exécution (rien n'est lancé) : les dot comptés par types
        hlo = xb.get_computation_hlo(xb.computation_from_module_proto(
            "probe", torch_xla._XLAC._get_xla_tensors_hlo_proto([loss, *grads.values()])))
        dots: dict = {}
        for result, operands in re.findall(r"=\s*(\w+)\[[^\]]*\]\S*\s+dot\(([^)]*)\)", hlo):
            key = result + "←" + ",".join(re.findall(r"(\w+)\[", operands))
            dots[key] = dots.get(key, 0) + 1
        out.update(probe_dots=dict(sorted(dots.items())), probe_dots_f32=sum(n for k, n in dots.items() if "f32" in k))
    except Exception as error:  # noqa: BLE001 — un diagnostic, la suite de la sonde reste valable
        out["probe_hlo_error"] = f"{type(error).__name__}: {str(error)[:300]}"
    xm.mark_step()
    n = len(getattr(model, "blocks", ()))
    names = [k for k in ("blocks.0.attn.qkv.weight", "blocks.0.attn.proj.weight", f"blocks.{n - 1}.mlp.fc.weight",
                         "embed.weight", "lm_head.weight") if k in grads]
    with torch.no_grad():
        norms = [grads[k].float().norm() for k in names]
        norms.append(torch.stack([g.float().norm() for g in grads.values()]).norm())
        out.update(probe_loss=loss.item(), probe_grad_norm=dict(zip(names + ["all"], torch.stack(norms).cpu().tolist())))
        if "skip_weights" in grads:
            out["probe_skip_grad"] = grads["skip_weights"].float().cpu().tolist()
    out["probe_s"] = round(time.time() - t0, 1)
    return out


def speed() -> dict:
    """Mode speed : SPEED_STEPS pas complets du pré-entraînement, faits comme run_stage les fait (même modèle, mêmes
    optimiseurs, même autocast, même accumulation, mark_step aux mêmes endroits, lots tirés de la même façon), sur
    un corpus de tokens tirés au hasard (loi de Zipf). Les 3 premiers pas (compilation) sont hors mesure. Le pas est
    celui de run_stage, recopié ligne à ligne (run_stage reste intact) ; seuls ajouts : mark_step et wait_device_ops
    chronométrés, et BF16_WEIGHTS. Puis, modèle de la mesure libéré, la sonde (probe ; SPEED_PROBE=0 : sans). Une ligne
    JSON, ajoutée à OUT/speed.jsonl.

    Lecture : warmup_s, la durée des pas hors mesure (compilation comprise), xla_compile_s, le temps de compilation
    d'XLA pendant ces pas. host_trace_s, le temps Python d'un pas hors mark_step et hors attente du TPU (traçage du
    modèle, lots) ; mark_step_s, le temps passé dans mark_step (hachage du graphe, lancement, et l'attente du TPU s'il
    calcule encore le graphe précédent) ; device_wait_s, le temps passé dans xm.wait_device_ops pendant le pas (en mode
    scan, gradient_accumulation attend la fin du pas précédent à chaque pas ; 0 en mode loop). host_trace_s proche de
    step_s : le TPU attend Python ; device_wait_s ou mark_step_s proche de step_s : Python attend le TPU. xla_measured :
    les compteurs pendant les pas mesurés ; une compilation de plus y est un graphe qui change d'un pas à l'autre,
    cpu_fallback (nom → appels pendant ces pas) non vide des opérations qui font l'aller-retour par le CPU (les .item()
    de la mesure tombent hors de la fenêtre : aten::_local_scalar_dense n'y apparaît que pour un vrai .item() par pas).
    Le témoin numérique : les champs probe_* (voir probe), à comparer à la ligne de référence. loss et param_norm (pas
    de la mesure) ne servent plus qu'à PREFETCH (au bit près) : ils voient mal le reste (voir probe)."""
    import copy
    import traceback

    import torch_xla.core.xla_model as xm

    warmup = 3
    n_steps = max(SPEED_STEPS, warmup + 1)
    entry: dict = {"mode": "speed", "model": MODEL, **MODELS[MODEL], "arch": ARCH, "block": BLOCK,
                   "optim": ("normuon" if PRETRAIN_MODERN["normuon"] else "muon") if ARCH == "modern" else "adamw",
                   "micro_batch": MICRO_BATCH, "accum_mode": ACCUM_MODE, "attn_kernel": ATTN_KERNEL,
                   "qkv_layout": QKV_LAYOUT, "qk_norm": QK_NORM, "rope": ROPE,
                   "layer_scan": LAYER_SCAN, "bf16_weights": BF16_WEIGHTS, "prefetch": PREFETCH, "steps": n_steps,
                   "warmup": warmup, "accelerator": os.environ.get("TPU_ACCELERATOR_TYPE", ""),
                   "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""), "xla_flags": os.environ.get("XLA_FLAGS", "")}
    if ATTN_KERNEL == "splash":
        entry.update(splash_block=SPLASH_BLOCK, splash_fused=SPLASH_FUSED)
    prefetch = None
    try:
        init_worker()
        import torch_xla

        entry["versions"] = f"torch {torch.__version__}, torch_xla {getattr(torch_xla, '__version__', '?')}"
        device = DEVICE
        cfg = PRETRAIN_MODERN if ARCH == "modern" else PRETRAIN  # le pré-entraînement, comme train_session
        scan = ACCUM_MODE == "scan"
        accum = max(1, TOKENS_PER_STEP // (MICRO_BATCH * BLOCK * WORLD))
        tokens_per_step = accum * MICRO_BATCH * BLOCK * WORLD
        entry.update(accum=accum, tokens_per_step=tokens_per_step, world=WORLD, flops_per_token=flops_per_token(),
                     peak_tflops=PEAK_TFLOPS)
        torch.manual_seed(SEED)
        model = make_model().to(device)
        master = None  # BF16_WEIGHTS : la copie float32 des poids, celle que les optimiseurs mettent à jour
        if BF16_WEIGHTS:
            master, model = model, copy.deepcopy(model).to(torch.bfloat16)
        owner = model if master is None else master  # les poids des optimiseurs
        entry["params"] = sum(p.numel() for p in model.parameters())

        # La suite recopie run_stage : dropout, optimiseurs, lots, puis le pas.
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = cfg["dropout"]
        model.train()
        if cfg.get("muon"):  # modded-nanoGPT : Muon pour les matrices des blocs, Adam pour le reste
            optimizers = gpt.modern_optimizers(owner, **{k: cfg[k] * LR_SCALE for k in ("muon_lr", "embed_lr", "head_lr", "scalar_lr")}, normuon=cfg["normuon"])
            clip = None
        else:
            params = [p for p in owner.parameters() if p.requires_grad]
            optimizers = [gpt.AdamW([{"params": [p for p in params if p.dim() >= 2], "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]},
                                     {"params": [p for p in params if p.dim() < 2], "lr": cfg["lr"], "weight_decay": 0.0}])]
            clip = 1.0
        vocab = config()["vocab_size"]
        zipf = 1.0 / np.arange(1, vocab + 1)
        parts = [(np.random.default_rng(SEED).choice(vocab, size=1 << 24, p=zipf / zipf.sum()).astype(np.uint16), 1.0)]
        weights = np.array([w for _, w in parts]) / sum(w for _, w in parts)
        generator = np.random.default_rng(SEED * 100 + RANK)
        lr = 0.1  # le multiplicateur des learning rates : une donnée du graphe, sa valeur ne change pas la vitesse
        loss_sum = torch.zeros((), device=device)
        clock = {"trace": 0.0, "sync": 0.0, "wait": 0.0, "last": time.perf_counter()}

        def mark_step() -> None:
            """xm.mark_step, chronométré : avant lui, le temps de Python (traçage, lots) ; dedans, celui d'XLA."""
            now = time.perf_counter()
            clock["trace"] += now - clock["last"]
            xm.mark_step()
            clock["last"] = time.perf_counter()
            clock["sync"] += clock["last"] - now

        wait_device_ops = xm.wait_device_ops

        def timed_wait(*args, **kwargs):
            """xm.wait_device_ops, chronométré, à la place de l'original pendant les pas (gradient_accumulation le
            cherche dans le module à chaque appel) : l'attente du TPU va dans clock["wait"], pas dans le traçage."""
            now = time.perf_counter()
            try:
                return wait_device_ops(*args, **kwargs)
            finally:
                waited = time.perf_counter() - now
                clock["wait"] += waited
                clock["last"] += waited  # décomptée du temps de Python que mark_step relève

        if scan:
            from torch_xla.experimental.gradient_accumulation import gradient_accumulation

            def train_step(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                """Celui de run_stage : la loss du micro-lot en float32, non divisée (gradient_accumulation divise)."""
                with AUTOCAST():
                    return model(x, y)[1].float()

        if PREFETCH:
            prefetch = Prefetcher((lambda: step_batch(parts, weights, accum, generator, device)) if scan else (
                lambda: batch_from(parts[generator.choice(len(parts), p=weights)][0], MICRO_BATCH, generator, device)), PREFETCH)

        def one_step() -> None:
            """Un pas de run_stage, ligne à ligne (plus BF16_WEIGHTS) ; le planning du learning rate en moins."""
            nonlocal loss_sum
            clock["last"] = time.perf_counter()
            if scan:
                x, y = step_batch(parts, weights, accum, generator, device) if prefetch is None else prefetch.next()
                loss_sum += gradient_accumulation(train_step, (x, y), model)
            else:
                for micro in range(accum):
                    if prefetch is None:
                        data = parts[generator.choice(len(parts), p=weights)][0]
                        x, y = batch_from(data, MICRO_BATCH, generator, device)
                    else:
                        x, y = prefetch.next()
                    with AUTOCAST():
                        loss = model(x, y)[1] / accum
                    loss.backward()
                    loss_sum += loss.detach()
                    mark_step()
            if master is not None:  # BF16_WEIGHTS : les gradients (bf16) du modèle passent à la copie float32
                for pm, pw in zip(master.parameters(), model.parameters()):
                    pm.grad, pw.grad = (None if pw.grad is None else pw.grad.float()), None
            if WORLD > 1:
                xm.all_reduce(xm.REDUCE_SUM, [p.grad for p in owner.parameters() if p.grad is not None], scale=1.0 / WORLD)
            if clip:
                nn.utils.clip_grad_norm_(owner.parameters(), clip)
            mult = torch.tensor(lr, dtype=torch.float32).to(device)
            for o in optimizers:
                o.step(mult)
            for o in optimizers:
                o.zero_grad()
            if master is not None:  # puis les poids mis à jour reviennent au modèle, en bf16
                with torch.no_grad():
                    for pm, pw in zip(master.parameters(), model.parameters()):
                        pw.copy_(pm)
            mark_step()

        at_start = xla_metrics()
        t0 = time.time()
        step_ends = []
        xm.wait_device_ops = timed_wait
        try:
            for i in range(n_steps):
                if i == warmup:  # compilation faite : la mesure part d'un TPU au repos, lectures faites
                    xm.wait_device_ops()
                    entry["warmup_s"] = round(time.time() - t0, 1)  # les pas hors mesure, compilation comprise
                    loss_warm = loss_sum.item()
                    before = xla_metrics()
                    if "CompileTime" in before:  # la compilation d'XLA seule, pendant ces pas
                        entry["xla_compile_s"] = round(before["CompileTime"][1] - at_start.get("CompileTime", [0, 0.0])[1], 1)
                    clock.update(trace=0.0, sync=0.0, wait=0.0)
                    t_warm = time.time()
                one_step()
                if i >= warmup:
                    step_ends.append(time.time())
        finally:
            xm.wait_device_ops = wait_device_ops
        xm.wait_device_ops()
        seconds = time.time() - t_warm
        measured = n_steps - warmup
        after = xla_metrics()
        loss_end = loss_sum.item()
        tps = measured * tokens_per_step / seconds
        durations = np.diff([t_warm] + step_ends)  # Python seul (le TPU suit en décalé) : la dispersion d'un pas à l'autre
        entry.update(ok=True, step_s=round(seconds / measured, 4), tokens_per_s=round(tps),
                     mfu=round(tps * flops_per_token() / (PEAK_TFLOPS * 1e12), 4),
                     step_s_min=round(float(durations.min()), 4), step_s_max=round(float(durations.max()), 4),
                     host_trace_s=round(clock["trace"] / measured, 4), mark_step_s=round(clock["sync"] / measured, 4),
                     device_wait_s=round(clock["wait"] / measured, 4),
                     loss_warmup=round(loss_warm / warmup, 5), loss=round((loss_end - loss_warm) / measured, 5))
        xla = {}
        for key, value in after.items():  # les compteurs des pas mesurés seulement
            old = before.get(key)
            if key == "cpu_fallback":  # les opérations retombées sur le CPU pendant ces pas, et combien de fois
                old = old or {}
                xla[key] = {n: v - old.get(n, 0) for n, v in sorted(value.items()) if v > old.get(n, 0)}
            elif isinstance(value, list):
                old = old or [0, 0.0]
                xla[key] = [value[0] - old[0], round(value[1] - old[1], 3)]
            else:
                xla[key] = value - (old or 0)
        entry.update(xla_measured=xla, compiles_total=compile_count())
        try:
            memory = xm.get_memory_info(device)
            entry.update(hbm_peak_gb=round(int(memory.get("peak_bytes_used", 0)) / 1e9, 2),
                         hbm_limit_gb=round(int(memory.get("bytes_limit", 0)) / 1e9, 2))
        except Exception:  # noqa: BLE001 — un diagnostic
            pass
        with torch.no_grad():  # les poids des optimiseurs après les pas : le témoin de PREFETCH (au bit près)
            entry["param_norm"] = round(torch.stack([p.detach().float().norm() for p in owner.parameters()]).norm().item(), 6)

        if PROFILE:  # quelques pas de plus, sous le profileur (après la mesure : elle n'en paie pas le coût)
            import torch_xla.debug.profiler as xp

            try:
                Path(PROFILE).mkdir(parents=True, exist_ok=True)
                xm.wait_device_ops()
                xp.start_trace(PROFILE)
                try:
                    for _ in range(PROFILE_STEPS):
                        one_step()
                    xm.wait_device_ops()
                finally:
                    xp.stop_trace()
                entry.update(profile=PROFILE, profile_steps=PROFILE_STEPS)
            except Exception as error:  # noqa: BLE001 — la mesure reste valable
                entry["profile_error"] = f"{type(error).__name__}: {str(error)[:300]}"

        if SPEED_PROBE:  # le témoin numérique, sur un modèle à part : le modèle de la mesure et ses optimiseurs libérés
            if prefetch is not None:
                prefetch.close()
                prefetch = None
            model = master = owner = optimizers = params = module = None  # noqa: F841 — leurs tenseurs libérés
            gc.collect()
            try:
                entry.update(probe())
            except Exception as error:  # noqa: BLE001 — la mesure reste valable
                entry["probe_error"] = f"{type(error).__name__}: {str(error)[:400]}"
                traceback.print_exc()
    except Exception as error:  # mémoire du TPU dépassée, noyau indisponible…
        entry.update(ok=False, error=f"{type(error).__name__}: {str(error)[:400]}")
        traceback.print_exc()
    finally:
        if prefetch is not None:
            prefetch.close()
    print(json.dumps(entry), flush=True)
    try:
        with open(OUT / "speed.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass
    return entry


# --- Lancement ------------------------------------------------------------------------------------------


RANK, WORLD, DEVICE = 0, 1, None
AUTOCAST = None


def init_worker() -> None:
    """Le cœur TPU de ce processus, son rang, et l'autocast bf16."""
    global RANK, WORLD, DEVICE, AUTOCAST
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    DEVICE = torch_xla.device() if hasattr(torch_xla, "device") else xm.xla_device()
    RANK, WORLD = xr.global_ordinal(), xr.world_size()
    if QKV_LAYOUT or QK_NORM != "f32" or ROPE != "split":
        use_fast_attention()
    if ATTN_KERNEL == "flash":
        use_flash_attention()
    elif ATTN_KERNEL == "splash":
        use_splash_attention()
    if LAYER_SCAN:
        use_layer_scan()
    try:
        from torch_xla.amp import autocast as xla_autocast

        AUTOCAST = lambda: xla_autocast(DEVICE, dtype=torch.bfloat16)  # noqa: E731
    except ImportError:
        AUTOCAST = lambda: torch.autocast("xla", dtype=torch.bfloat16)  # noqa: E731


if __name__ == "__main__":
    import torch_xla.distributed.xla_multiprocessing as xmp

    mode = sys.argv[1]
    if mode == "bench":  # un seul processus : il calcule sur le premier cœur
        OUT.mkdir(parents=True, exist_ok=True)
        result = bench()
        (OUT / "bench.json").write_text(json.dumps(result, indent=2))
    elif mode == "speed":  # un seul processus, une configuration : une ligne JSON (voir speed)
        OUT.mkdir(parents=True, exist_ok=True)
        speed()
    elif mode == "train":
        if os.environ.get("TPU_ACCELERATOR_TYPE", "").endswith("-1"):  # une seule puce (v5e-1, v6e-1 de Colab) :
            train_session(0)  # xmp.spawn y lit TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 et cherche 4 puces
        else:  # le TPU v5e-8 de Kaggle : un processus par puce
            xmp.spawn(train_session, args=())
