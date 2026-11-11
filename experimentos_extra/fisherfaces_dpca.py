"""Fisherfaces con d_pca y d_lda mayores, sobre Olivetti y LFW.

Usa el mismo split, las mismas fábricas y los mismos hiperparámetros del proyecto.
Mide aparte el tiempo de ajustar el embedding, que el CSV principal no incluye.

    python experimentos_extra/fisherfaces_dpca.py
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
from src.embeddings.fisherfaces import ExtractorFisherfaces  # noqa: E402

D_PCA = (50, 100, 200)
D_LDA = (10, 20, 30, "C-1")
MODELOS = ("svm", "knn")
SALIDA = DIR_RESULTADOS / "fisherfaces_dpca.csv"


def main() -> None:
    asegurar_directorios()
    fabrica = FabricaClasificadores()
    filas = []
    for cargador in (CargadorOlivetti(), CargadorLFW()):
        conjunto = cargador.cargar()
        particion = DivisorEstratificado().dividir(conjunto)
        C = conjunto.n_clases
        n_train = particion.X_train.shape[0]
        print(f"== {conjunto.nombre}: N_train={n_train} C={C} ==")
        for d_pca in D_PCA:
            if d_pca > n_train - C:
                print(f"  OMITIDO d_pca={d_pca}: supera N_train-C={n_train - C}")
                continue
            for d_lda in sorted({C - 1 if v == "C-1" else v for v in D_LDA}):
                if d_lda > min(C - 1, d_pca):
                    print(f"  OMITIDO d_pca={d_pca} d_lda={d_lda}: supera min(C-1, d_pca)")
                    continue
                extractor = ExtractorFisherfaces(d_pca=d_pca, d_lda=d_lda)
                t0 = time.perf_counter()
                extractor.fit(particion.X_train, particion.y_train)
                t_embedding = time.perf_counter() - t0
                Z_train = extractor.transform(particion.X_train)
                Z_test = extractor.transform(particion.X_test)
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
                            "d_pca": d_pca,
                            "d_lda": d_lda,
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
    tabla = pd.DataFrame(filas).round(
        {"accuracy_train": 4, "accuracy_test": 4, "macro_f1_test": 4,
         "tiempo_fit_embedding": 4, "tiempo_clasificador": 4}
    )
    tabla.to_csv(SALIDA, index=False)
    print(f"\nTabla guardada en {SALIDA}")
    with pd.option_context("display.width", 160, "display.max_rows", None):
        print(tabla.to_string(index=False))


if __name__ == "__main__":
    main()
