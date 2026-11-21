"""Carga de LFW: hasta 50 identidades con al menos 20 fotos."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_lfw_people

from src.config import DIR_CACHE, RUTA_LFW_NPZ
from src.datos.base import CargadorRostros, validar_conjunto
from src.modelos import ConjuntoRostros

MIN_FOTOS_POR_PERSONA = 20
MAX_IDENTIDADES = 50
ESCALA = 0.5


class CargadorLFW(CargadorRostros):
    """Fotografías de LFW con mayor diversidad de pose, luz y fondo."""

    nombre = "lfw"

    def cargar(self) -> ConjuntoRostros:
        """Lee ``datos_app/lfw_50.npz`` si existe; si no, descarga con sklearn."""
        if RUTA_LFW_NPZ.exists():
            return self._desde_npz(RUTA_LFW_NPZ)
        return self.cargar_desde_sklearn()

    def _desde_npz(self, ruta) -> ConjuntoRostros:
        with np.load(ruta, allow_pickle=False) as archivo:
            images = archivo["images"]
            y = archivo["y"].astype(int)
            nombres = [str(n) for n in archivo["nombres"]]
        return self._armar(images, y, nombres)

    def cargar_desde_sklearn(self) -> ConjuntoRostros:
        lfw = fetch_lfw_people(
            min_faces_per_person=MIN_FOTOS_POR_PERSONA,
            resize=ESCALA,
            data_home=str(DIR_CACHE),
            download_if_missing=True,
        )
        y_original = np.asarray(lfw.target, dtype=int)

        # Las identidades con más fotos; a igual cantidad, el id original menor.
        ids, cuentas = np.unique(y_original, return_counts=True)
        orden = np.lexsort((ids, -cuentas))
        ids_elegidos = np.sort(ids[orden[:MAX_IDENTIDADES]])
        mascara = np.isin(y_original, ids_elegidos)

        images = np.asarray(lfw.images)[mascara]
        # ids_elegidos está ordenado: y queda reindexado a 0..C-1.
        y = np.searchsorted(ids_elegidos, y_original[mascara]).astype(int)
        nombres = [str(lfw.target_names[i]) for i in ids_elegidos]

        return self._armar(images, y, nombres)

    def _armar(self, images: np.ndarray, y: np.ndarray, nombres: list[str]) -> ConjuntoRostros:
        n, alto, ancho = images.shape
        conjunto = ConjuntoRostros(
            nombre=self.nombre,
            images=images,
            X=images.reshape(n, alto * ancho).astype(float),
            y=y,
            nombres_identidades=nombres,
            alto=int(alto),
            ancho=int(ancho),
        )
        validar_conjunto(conjunto)
        return conjunto
