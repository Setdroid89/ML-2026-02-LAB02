"""Genera datos_app/lfw_50.npz con el subconjunto LFW que usa el laboratorio.

Descarga LFW con scikit-learn (~200 MB, una sola vez), aplica la selección de
``CargadorLFW`` y guarda imágenes float32, etiquetas y nombres, comprimidos.
Después comprueba que el cargador, leyendo el .npz, devuelve lo mismo.

    python scripts/generar_cache_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RUTA_LFW_NPZ  # noqa: E402
from src.datos.lfw import CargadorLFW  # noqa: E402


def main() -> None:
    cargador = CargadorLFW()
    original = cargador.cargar_desde_sklearn()
    RUTA_LFW_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RUTA_LFW_NPZ,
        images=original.images.astype(np.float32),
        y=original.y.astype(np.int16),
        nombres=np.asarray(original.nombres_identidades),
    )
    desde_npz = cargador._desde_npz(RUTA_LFW_NPZ)

    iguales = (
        np.array_equal(original.images, desde_npz.images)
        and np.array_equal(original.X, desde_npz.X)
        and np.array_equal(original.y, desde_npz.y)
        and original.nombres_identidades == desde_npz.nombres_identidades
        and (original.alto, original.ancho) == (desde_npz.alto, desde_npz.ancho)
    )
    mb = RUTA_LFW_NPZ.stat().st_size / 1e6
    print(f"N={original.X.shape[0]} C={original.n_clases} H={original.alto} W={original.ancho}")
    print(f"images dtype={original.images.dtype}; {RUTA_LFW_NPZ} = {mb:.1f} MB")
    print(f"Idéntico al de fetch_lfw_people (array_equal en images, X, y, nombres): {iguales}")
    if not iguales:
        raise SystemExit("El .npz no coincide con el conjunto original.")
    if mb >= 50:
        raise SystemExit("El .npz pesa 50 MB o más.")


if __name__ == "__main__":
    main()
