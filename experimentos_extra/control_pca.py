"""Controles: PCA solo (tabla A) y LDA solo, sin PCA previo (tabla B).

Mismo split, FabricaClasificadores e hiperparámetros del proyecto. Un solo CSV,
con la columna ``embedding`` para distinguir las dos tablas.

    python experimentos_extra/control_pca.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clasificadores.fabrica import FabricaClasificadores  # noqa: E402
from src.config import DIR_RESULTADOS, asegurar_directorios  # noqa: E402
from src.datos.division import DivisorEstratificado  # noqa: E402
from src.datos.lfw import CargadorLFW  # noqa: E402
from src.datos.olivetti import CargadorOlivetti  # noqa: E402
from src.embeddings.lda import ExtractorLDA  # noqa: E402
from src.embeddings.pca import ExtractorPCA  # noqa: E402

D_PCA = (10, 20, 30, 50, 100, 200)
D_LDA = (10, 20, 30, "C-1")
MODELOS = ("svm", "knn")
SALIDA = DIR_RESULTADOS / "control_pca_lda.csv"


def evaluar(conjunto, particion, extractor, embedding, d, fabrica) -> list[dict]:
    t0 = time.perf_counter()
    extractor.fit(particion.X_train, particion.y_train)
    t_embedding = time.perf_counter() - t0
    Z_train = extractor.transform(particion.X_train)
    Z_test = extractor.transform(particion.X_test)
    filas = []
    for nombre in MODELOS:
        modelo = fabrica.crear(nombre)
        t0 = time.perf_counter()
        modelo.fit(Z_train, particion.y_train)
        pred_test = modelo.predict(Z_test)
        t_clasificador = time.perf_counter() - t0
        pred_train = modelo.predict(Z_train)
        filas.append(
            {
                "dataset": conjunto.nombre,
                "embedding": embedding,
                "d": d,
                "modelo": nombre,
                "accuracy_train": accuracy_score(particion.y_train, pred_train),
                "accuracy_test": accuracy_score(particion.y_test, pred_test),
                "macro_f1_test": f1_score(
                    particion.y_test, pred_test, average="macro", zero_division=0
                ),
                "tiempo_fit_embedding": t_embedding,
                "tiempo_clasificador": t_clasificador,
            }
        )
    return filas


def main() -> None:
    asegurar_directorios()
    fabrica = FabricaClasificadores()
    filas: list[dict] = []
    for cargador in (CargadorOlivetti(), CargadorLFW()):
        conjunto = cargador.cargar()
        particion = DivisorEstratificado().dividir(conjunto)
        C = conjunto.n_clases
        print(f"== {conjunto.nombre}: N_train={particion.X_train.shape[0]} C={C} ==")
        for d in D_PCA:
            filas += evaluar(conjunto, particion, ExtractorPCA(n_components=d), "pca", d, fabrica)
        for v in D_LDA:
            d = C - 1 if v == "C-1" else v
            filas += evaluar(conjunto, particion, ExtractorLDA(n_components=d), "lda", d, fabrica)
    tabla = pd.DataFrame(filas).round(4)
    tabla.to_csv(SALIDA, index=False)
    print(f"\nTabla guardada en {SALIDA}")
    with pd.option_context("display.width", 160, "display.max_rows", None):
        for emb, titulo in (("pca", "Tabla A (PCA solo)"), ("lda", "Tabla B (LDA solo)")):
            print(f"\n{titulo}")
            print(tabla[tabla.embedding == emb].to_string(index=False))


if __name__ == "__main__":
    main()
