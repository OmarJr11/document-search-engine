import os
import re
import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from classes.document_indexer import DocumentIndexer
from classes.pdf_processor import PDFProcessor
from functions import process_text


class DocumentSearchFacade:
    """ Fachada que centraliza la extracción, preprocesamiento, indexación y búsqueda """

    def __init__(self):
        self.processor = PDFProcessor()
        self.indexer = DocumentIndexer()
        self.processed_documents = []
        self.clusters = None  # Para almacenar los clusters
        self.clusters_scores = []
        self.pdf_files = []
        self.pdf_titles = {}
        
        self.eps = 0.816 # Radio de vecindad para DBSCAN
        self.min_samples = 2  # Número mínimo de puntos para formar un cluster
   
        
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
        Realiza clustering de los documentos indexados usando DBSCAN.
        """
        if not self.processed_documents:
            raise ValueError(
                "No hay documentos procesados para realizar clustering.")

        tfidf_matrix = self.indexer.document_matrix
        
        self.pdf_files = process_text.get_pdf_files()

        # Configurar y ajustar DBSCAN
        dbscan = DBSCAN(
            eps=self.eps, min_samples=self.min_samples, metric="cosine")
        self.clusters = dbscan.fit_predict(tfidf_matrix)

        # Asociar documentos a clusters
        self.clustered_documents = {}
        self.clustered_pdfs = {}
        self.pdf_titles = {}
        null_docs = 0
        for idx, cluster_id in enumerate(self.clusters):
            if cluster_id == -1:
                null_docs += 1  # Contar documentos de ruido
                # Ignorar puntos etiquetados como ruidodocument_vectors
                continue
            if cluster_id not in self.clustered_documents:
                self.clustered_documents[cluster_id] = []
            self.clustered_documents[cluster_id].append(
                self.processed_documents[idx])
            if cluster_id not in self.clustered_pdfs:
                self.clustered_pdfs[cluster_id] = []
            self.clustered_pdfs[cluster_id].append(
                self.pdf_files[idx])
            if cluster_id not in self.pdf_titles:
                self.pdf_titles[cluster_id] = []
            self.pdf_titles[cluster_id].append(
               self.processor.extract_title(self.pdf_files[idx]))
        
        # Guardar los clusters en el estado de la sesión
        st.session_state["clusters"] = self.clusters
        st.session_state["clustered_documents"] = self.clustered_documents

    def search_documents(self, query, threshold=0.009):
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
        cluster_scores = {i: 0 for i in self.clustered_documents.keys()}
        cluster_counts = {i: 0 for i in self.clustered_documents.keys()} 
        for idx, cluster_id in enumerate(self.clusters):
            if cluster_id != -1:  # Ignorar puntos de ruido
                cluster_scores[cluster_id] += similarities[idx]
                cluster_counts[cluster_id] += 1 # Contar documentos en cada cluster
                
        for cluster_id in cluster_scores.keys():
            if cluster_counts[cluster_id] > 0:  # Evitar división por cero
                cluster_scores[cluster_id] /= cluster_counts[cluster_id]
                
        # Verificar si hay clusters válidos
        if not cluster_scores:
            raise ValueError(
                "No se generaron clusters válidos. Verifica los parámetros de DBSCAN.")

        # Seleccionar el cluster más relevante
        best_cluster = max(cluster_scores, key=cluster_scores.get)
        
        self.clusters_scores = cluster_scores
        
        # Filtrar documentos del cluster más relevante
        cluster_docs = self.clustered_documents[best_cluster]
        cluster_pdfs = self.clustered_pdfs[best_cluster]
        cluster_titles = self.pdf_titles[best_cluster]
        cluster_vectors = self.indexer.document_matrix[self.clusters == best_cluster]
        
        # Calcular similitud de coseno dentro del cluster
        cluster_similarities = cosine_similarity(
            query_vector, cluster_vectors).flatten()
        top_indices = cluster_similarities.argsort()[::-1]

        # Filtrar resultados por umbral
        filtered_results = [
            (cluster_docs[i], cluster_similarities[i], cluster_pdfs[i], cluster_titles[i])
            for i in top_indices if cluster_similarities[i] >= threshold
        ]

        return filtered_results

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
        pdf_files = process_text.get_pdf_files()

        # Vectorizar el documento seleccionado
        selected_vector = vectorizer.transform([selected_document])

        # Calcular la similitud de coseno con todos los documentos
        similarities = cosine_similarity(
            selected_vector, document_matrix).flatten()

        # Ordenar los documentos por puntaje de similitud (de mayor a menor)
        similar_indices = similarities.argsort()[::-1]

        # Excluir el documento seleccionado y filtrar por umbral
        similar_documents = [
            (processed_documents[i], similarities[i], pdf_files[i])
            for i in similar_indices
            if processed_documents[i] != selected_document and similarities[i] >= threshold
        ]

        # Retornar todos los documentos similares que cumplen con el umbral
        return similar_documents
