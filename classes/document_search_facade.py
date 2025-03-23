import os
import re
import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from classes.document_indexer import DocumentIndexer
from classes.pdf_processor import PDFProcessor
from sklearn.metrics.pairwise import cosine_similarity


class DocumentSearchFacade:
    """ Fachada que centraliza la extracción, preprocesamiento, indexación y búsqueda """

    def __init__(self):
        self.processor = PDFProcessor()
        self.indexer = DocumentIndexer()
        self.processed_documents = []
        self.clusters = None  # Para almacenar los clusters
        self.num_clusters = 8  # Número de clusters (puedes ajustarlo)

    def add_documents(self, processed_folder_path="./processed_files"):
        """
        Obtiene todos los archivos .txt de la carpeta processed_files,
        lee su contenido y los pasa a la función index_documents.
        """
        # Lista para almacenar el contenido de los documentos procesados
        self.processed_documents = []

        # Obtener todos los archivos .txt de la carpeta processed_files
        txt_files = [os.path.join(processed_folder_path, file) for file in os.listdir(
            processed_folder_path) if file.endswith(".txt")]

        # Leer el contenido de cada archivo .txt
        for txt_file in txt_files:
            with open(txt_file, "r", encoding="utf-8") as file:
                # Leer el contenido del archivo como cadena
                processed_text = file.read()
                # Agregar el texto procesado (cadena) a la lista de documentos procesados
                self.processed_documents.append(processed_text)

        # Verificar si hay documentos procesados
        if not self.processed_documents:
            raise ValueError(
                "No se encontraron documentos procesados para indexar.")

        # Ajustar el vectorizador TF-IDF con los documentos procesados
        self.indexer.vectorizer.fit(self.processed_documents)

        # Crear la matriz de documentos
        self.indexer.document_matrix = self.indexer.vectorizer.transform(
            self.processed_documents)

        # Guardar el vectorizador y la matriz en el estado de la sesión
        st.session_state["vectorizer"] = self.indexer.vectorizer
        st.session_state["document_matrix"] = self.indexer.document_matrix
        st.session_state["processed_documents"] = self.processed_documents

    def perform_clustering(self):
        """
        Realiza clustering de los documentos indexados usando K-Means.
        """
        tfidf_matrix = self.indexer.document_matrix
        # Ajustar el número de clusters
        n_clusters = min(self.num_clusters, max(2, tfidf_matrix.shape[0] // 2))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.clusters = kmeans.fit_predict(tfidf_matrix)

        # Asociar documentos a clusters
        self.clustered_documents = {i: [] for i in range(n_clusters)}
        for idx, cluster_id in enumerate(self.clusters):
            self.clustered_documents[cluster_id].append(
                self.processed_documents[idx]
            )

    def search_documents(self, query, threshold=0.02):
        """
        Realiza una búsqueda en los documentos más relevantes basados en clustering.
        """
        # Preprocesar la consulta
        query = self.processor.preprocess_text(query)
        # Vectorizar la consulta
        query_vector = self.indexer.vectorizer.transform([query])
        similarities = cosine_similarity(
            query_vector, self.indexer.document_matrix).flatten()

        # Calcular el puntaje acumulado por cluster
        cluster_scores = {i: 0 for i in range(len(self.clustered_documents))}
        for idx, cluster_id in enumerate(self.clusters):
            cluster_scores[cluster_id] += similarities[idx]

        # Seleccionar el cluster más relevante
        best_cluster = max(cluster_scores, key=cluster_scores.get)

        # Filtrar documentos del cluster más relevante
        cluster_docs = self.clustered_documents[best_cluster]
        cluster_vectors = self.indexer.document_matrix[self.clusters == best_cluster]

        # Calcular similitud de coseno dentro del cluster
        cluster_similarities = cosine_similarity(
            query_vector, cluster_vectors).flatten()
        top_indices = cluster_similarities.argsort()[::-1]

        # Filtrar resultados por umbral
        filtered_results = [
            (cluster_docs[i], cluster_similarities[i])
            for i in top_indices if cluster_similarities[i] >= threshold
        ]

        return filtered_results

    def generate_summary(self, document, query):
        """
        Genera un resumen completo del documento basado en la consulta.
        Selecciona las oraciones más relevantes que contienen las palabras clave de la consulta.
        """
        # Dividir el documento en oraciones
        sentences = re.split(r'(?<=[.!?]) +', document)

        # Dividir la consulta en palabras clave
        keywords = query.lower().split()

        # Calcular la relevancia de cada oración
        sentence_scores = []
        for sentence in sentences:
            score = sum(1 for word in keywords if word in sentence.lower())
            if score > 0:
                sentence_scores.append((sentence, score))

        # Ordenar las oraciones por relevancia (puntaje)
        sentence_scores = sorted(
            sentence_scores, key=lambda x: x[1], reverse=True)

        # Seleccionar las oraciones más relevantes
        summary_sentences = [sentence for sentence, _ in sentence_scores]

        # Construir el resumen completo
        summary = " ".join(summary_sentences)

        return (summary.strip())[:500]

    def highlight_keywords(self, text, query, color="yellow"):
        """
        Resalta las palabras clave de la consulta en el texto con un color personalizado.
        """
        keywords = query.split()
        for keyword in keywords:
            # Usar HTML para resaltar las palabras clave con un color
            text = re.sub(
                f"(?i)({keyword})", rf"<span style='color: {color};'>\1</span>", text)
        return text

    def recommend_similar_documents(self, selected_document, threshold=0.1):
        """
        Recomienda documentos similares al documento seleccionado.
        Calcula la similitud de coseno entre el documento seleccionado y todos los demás documentos.
        Filtra los resultados para aquellos cuya similitud sea mayor al umbral.
        """
        # Verificar si el vectorizador y la matriz están en el estado de la sesión
        if "vectorizer" not in st.session_state or "document_matrix" not in st.session_state:
            raise ValueError(
                "El vectorizador TF-IDF o la matriz de documentos no están disponibles.")

        vectorizer = st.session_state["vectorizer"]
        document_matrix = st.session_state["document_matrix"]
        processed_documents = st.session_state["processed_documents"]

        # Vectorizar el documento seleccionado
        selected_vector = vectorizer.transform([selected_document])

        # Calcular la similitud de coseno con todos los documentos
        similarities = cosine_similarity(
            selected_vector, document_matrix).flatten()

        # Ordenar los documentos por puntaje de similitud (de mayor a menor)
        similar_indices = similarities.argsort()[::-1]

        # Excluir el documento seleccionado y filtrar por umbral
        similar_documents = [
            (processed_documents[i], similarities[i])
            for i in similar_indices
            if processed_documents[i] != selected_document and similarities[i] >= threshold
        ]

        # Retornar todos los documentos similares que cumplen con el umbral
        return similar_documents
