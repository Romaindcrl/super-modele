# HomerGPT

Un GPT français entraîné *from scratch*, qui écrit des répliques dans le style des Simpson. Projet ESILV, cours
Creative Technology.

## Le modèle rendu : HomerGPT (`v6-591M-ft2`)

| | |
|---|---|
| Taille | 591 M paramètres : 20 couches × 1536, 12 têtes de 128 |
| Contexte, tokenizer | 512 tokens, BPE de 8 192 tokens |
| Architecture | modded-nanoGPT : RoPE, RMSNorm sans paramètre, QK-norm, ReLU², pas de biais, connexions en U, logits plafonnés |
| Optimiseur | NorMuon pour les matrices des blocs, Adam pour le reste, planning WSD |
| Pré-entraînement | 7,0 G tokens de français (sous-titres OpenSubtitles 50 %, FineWeb-2 30 %, Wikipédia 20 %), 13 358 pas de 524 288 tokens, sur TPU Google Colab (PyTorch/XLA, bf16) |
| Affinage | scripts des Simpson en français (`Personnage: réplique`), 25 % de corpus général rejoué, sans dropout, 100 M tokens |
| Poids | `v6-591M-ft2-bf16.safetensors` (1,2 Go), dans la *release* GitHub. La config et le tokenizer sont dans le fichier |

## Résultats

Entropie croisée sur les dialogues Simpson de test, en bits par caractère (plus bas = mieux, comparable quel que
soit le tokenizer) :

| Version | Modèle | Test |
|---|---|---|
| v1-17M | GPT du cours, Wikipédia puis dialogues | 1,420 |
| v2-98M | GPT-2 small, AdamW, GPU Kaggle | 1,257 |
| v3-98M | GPT-2 small modded-nanoGPT, NorMuon | 1,229 |
| **v6-591M-ft2** | **le modèle rendu** | **1,086** |

Les répliques sont en français correct et respectent le format, mais elles restent souvent courtes ou génériques,
et les personnages se mélangent. C'est la limite d'un modèle de cette taille entraîné sur 7 G tokens.

## Utiliser le modèle

Placer `v6-591M-ft2-bf16.safetensors` (release) dans `checkpoints/`, puis :

```bash
uv run app.py    # chat web sur http://localhost:8765
uv run chat.py   # chat dans le terminal
```

## Fichiers

- `transformer_from_scratch.ipynb` : le notebook commenté. Il va du code du cours à l'architecture du modèle
  rendu, entraînée à petite échelle sur un Mac, puis recharge v6-591M-ft2.
- `src/super_modele/gpt.py` : le modèle, les tokenizers, les optimiseurs (Muon, NorMuon) et la génération.
- `src/super_modele/interface.py`, `app.py`, `chat.py` : le chat.
- `kaggle/train_tpu.py` : l'entraînement sur TPU (pré-entraînement, affinage, score).
- `kaggle/prepare_data.py`, `kaggle/prepare_dialogue.py`, `prepare_simpsons.py` : la préparation des corpus.
- `colab_watch.py`, `kaggle_watch.py`, `dashboard.py` : le pilotage et le suivi des sessions Colab et Kaggle.

Les données (corpus, scripts des Simpson) ne sont pas dans le dépôt. Les scripts des Simpson ont servi à un usage
privé et non commercial.
