#!/usr/bin/env python3
"""Interfaz web para revisar rostros de prueba con un modelo ya ajustado.

Uso (con el entorno conda activado):

    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from src.config import DIMENSIONES
from src.interfaz.servicio import ServicioReconocimiento


st.set_page_config(page_title="Embeddings faciales", layout="wide")


@st.cache_resource(show_spinner="Cargando datos y separando TRAIN/TEST...")
def obtener_servicio(dataset: str) -> ServicioReconocimiento:
    """Un servicio por dataset: datos, split y modelos ajustados se reutilizan."""
    return ServicioReconocimiento(dataset)


@st.cache_data
def obtener_etiquetas(dataset: str) -> list[str]:
    return obtener_servicio(dataset).etiquetas_prueba()


def main() -> None:
    with st.sidebar:
        st.header("Datos")
        dataset = st.radio("Conjunto", options=["olivetti", "lfw"], format_func=_etiqueta_dataset)
    servicio = obtener_servicio(dataset)
    st.title("Reconocimiento de rostros con embeddings lineales")
    st.caption(
        f"Cada fotografía pertenece al conjunto de prueba de {_etiqueta_dataset(dataset)} "
        f"({servicio.particion.images_test.shape[0]} imágenes). "
        "No participó del ajuste de PCA, LDA, Fisherfaces ni del clasificador."
    )

    with st.sidebar:
        st.header("Modelo")
        embedding = st.radio(
            "Representación",
            options=["pca", "lda", "fisherfaces"],
            format_func=_etiqueta_embedding,
        )
        clasificador = st.radio(
            "Clasificador",
            options=["svm", "knn", "arbol"],
            format_func=_etiqueta_clasificador,
        )
        d = st.select_slider("Componentes (d)", options=list(DIMENSIONES))
        st.info(
            "PCA, LDA y Fisherfaces se ajustan solo con TRAIN; "
            "la foto de prueba solo se proyecta."
        )
        etiquetas = obtener_etiquetas(dataset)
        indice = st.selectbox(
            "Fotografía de prueba",
            options=range(len(etiquetas)),
            format_func=lambda i: etiquetas[i],
        )

    with st.spinner("Cargando el modelo ajustado en TRAIN..."):
        prediccion = servicio.predecir(indice, embedding, clasificador, int(d))

    columna_foto, columna_resultado = st.columns([1, 1.4], gap="large")
    with columna_foto:
        st.image(
            servicio.imagen_prueba(indice),
            caption=prediccion.nombre_real,
            clamp=True,
            width=280,
        )
    with columna_resultado:
        if prediccion.acierto:
            st.success("El modelo identificó la cara.")
        else:
            st.error("El modelo no identificó la cara.")
        real, predicha, probabilidad = st.columns(3)
        real.markdown(f"**Identidad real**\n\n{prediccion.nombre_real}")
        predicha.markdown(f"**Identidad predicha**\n\n{prediccion.nombre_predicho}")
        probabilidad.markdown(f"**Probabilidad**\n\n{prediccion.probabilidad:.1%}")
        st.caption(prediccion.nota_probabilidad)
        st.subheader("Cinco identidades más probables")
        tabla = pd.DataFrame(prediccion.top, columns=["identidad", "probabilidad"])
        st.bar_chart(
            tabla,
            x="identidad",
            y="probabilidad",
            horizontal=True,
            sort=False,
        )


def _etiqueta_embedding(nombre: str) -> str:
    return {"pca": "PCA", "lda": "LDA", "fisherfaces": "Fisherfaces"}[nombre]


def _etiqueta_dataset(nombre: str) -> str:
    return {"olivetti": "Olivetti", "lfw": "LFW (50 identidades)"}[nombre]


def _etiqueta_clasificador(nombre: str) -> str:
    return {"svm": "SVM", "knn": "k-NN", "arbol": "Árbol"}[nombre]


main()
