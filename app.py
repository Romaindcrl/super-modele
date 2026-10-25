"""Interface pour discuter avec le modèle : `uv run app.py`, puis ouvrir http://localhost:8765.

On choisit qui on est et à qui on parle ; options : `uv run app.py --help`. Ajoute `src/` au chemin
d'import lui-même, comme `chat.py` (voir son docstring).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from super_modele.interface import main  # noqa: E402

if __name__ == "__main__":
    main()
