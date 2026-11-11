"""Fisherfaces: PCA comprime y LDA discrimina. Lo implementa el alumno."""

from __future__ import annotations

import numpy as np

from src.embeddings.base import ExtractorEmbedding
from src.embeddings.lda import ExtractorLDA
from src.embeddings.pca import ExtractorPCA


class ExtractorFisherfaces(ExtractorEmbedding):
    """Proyección z = W_LDAᵀ W_PCAᵀ (x − μ), ajustada solo con TRAIN.

    Orden obligatorio:

    1. Ajustar PCA con ``d_pca`` solo sobre ``X_train`` (ahí se estima μ).
    2. Ajustar LDA con ``d_lda`` sobre ese PCA de TRAIN, usando ``y_train``.
    3. Transformar TEST con las dos proyecciones, sin volver a ajustar.

    Contrato: ``d_lda <= min(C - 1, d_pca)``. Revise también el rango y
    la cantidad de muestras. Puede reutilizar ``ExtractorPCA`` y ``ExtractorLDA``.
    """

    nombre = "fisherfaces"

    def __init__(self, d_pca: int, d_lda: int) -> None:
        if d_pca < 1 or d_lda < 1:
            raise ValueError("Fisherfaces requiere d_pca y d_lda positivos.")
        if d_lda > d_pca:
            raise ValueError(
                f"d_lda={d_lda} supera d_pca={d_pca}. "
                "Debe cumplirse d_lda <= min(C-1, d_pca)."
            )
        self.d_pca = int(d_pca)
        self.d_lda = int(d_lda)

    @property
    def dimension(self) -> int:
        return self.d_lda

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> ExtractorFisherfaces:
        if y is None:
            raise ValueError("Fisherfaces necesita y_train: LDA usa las identidades.")
        maximo = int(np.unique(y).size) - 1
        if self.d_lda > min(maximo, self.d_pca):
            raise ValueError(
                f"d_lda={self.d_lda} supera min(C-1={maximo}, d_pca={self.d_pca}). "
                "Reduzca d_lda."
            )
        # Ambos pasos ven solo TRAIN: PCA estima μ y W_PCA; LDA aprende sobre Z_pca.
        self._pca = ExtractorPCA(n_components=self.d_pca).fit(X)
        self._lda = ExtractorLDA(n_components=self.d_lda).fit(self._pca.transform(X), y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_lda"):
            raise RuntimeError("Fisherfaces: llame a fit antes de transform.")
        return self.comprobar(self._lda.transform(self._pca.transform(X)), X.shape[0])
