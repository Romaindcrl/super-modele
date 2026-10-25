"""Le suivi de l'expérience en direct : `uv run dashboard.py`, puis ouvrir http://localhost:8766.

Chaque étape (kaggle/experience.json), l'entraînement (loss, tokens/s), les résultats et les journaux.
Ajoute `src/` au chemin d'import lui-même, comme `app.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from super_modele.dashboard import main  # noqa: E402

if __name__ == "__main__":
    main()
