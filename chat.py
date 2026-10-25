"""Discuter avec le modèle entraîné : `uv run chat.py` (options : `uv run chat.py --help`).

Ajoute `src/` au chemin d'import lui-même : dans un dossier synchronisé par iCloud, macOS marque cachés
les fichiers de `.venv`, et Python ≥ 3.13 ignore alors le `.pth` de l'installation éditable du paquet.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from super_modele import main  # noqa: E402

if __name__ == "__main__":
    main()
