import os
from sklearn.cluster import KMeans
from document_indexer import DocumentIndexer
from pdf_processor import PDFProcessor
from sklearn.metrics.pairwise import cosine_similarity


class DocumentSearchFacade:
    """ Fachada que centraliza la extracción, preprocesamiento, indexación y búsqueda """

    def __init__(self):
        self.processor = PDFProcessor()
        self.indexer = DocumentIndexer()
        self.processed_documents = []
        self.clusters = None  # Para almacenar los clusters
        self.num_clusters = 2  # Número de clusters (puedes ajustarlo)

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

        # Indexar los documentos procesados
        self.indexer.index_documents(self.processed_documents)
        # Realizar clustering
        self.perform_clustering()

    def perform_clustering(self):
        """
        Realiza clustering de los documentos indexados usando K-Means.
        """
        # Usar la matriz TF-IDF para clustering
        tfidf_matrix = self.indexer.document_matrix
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        self.clusters = kmeans.fit_predict(tfidf_matrix)

        # Asociar documentos a clusters
        self.clustered_documents = {i: [] for i in range(self.num_clusters)}
        for idx, cluster_id in enumerate(self.clusters):
            self.clustered_documents[cluster_id].append(
                self.processed_documents[idx])

    def search_documents(self, query):
        """
        Realiza una búsqueda en los documentos más relevantes basados en clustering.
        """
        # Vectorizar la consulta
        query_vector = self.indexer.vectorizer.transform([query])

        # Calcular similitud de coseno con todos los documentos
        similarities = cosine_similarity(
            query_vector, self.indexer.document_matrix).flatten()

        # Encontrar el cluster más relevante
        cluster_scores = {i: 0 for i in range(self.num_clusters)}
        for idx, cluster_id in enumerate(self.clusters):
            cluster_scores[cluster_id] += similarities[idx]

        # Seleccionar el cluster con mayor puntaje
        best_cluster = max(cluster_scores, key=cluster_scores.get)

        # Filtrar documentos del cluster más relevante
        cluster_docs = self.clustered_documents[best_cluster]
        cluster_vectors = self.indexer.document_matrix[self.clusters == best_cluster]

        # Calcular similitud de coseno dentro del cluster
        cluster_similarities = cosine_similarity(
            query_vector, cluster_vectors).flatten()
        top_indices = cluster_similarities.argsort()[::-1]

        # Devolver los documentos más relevantes del cluster
        return [(cluster_docs[i], cluster_similarities[i]) for i in top_indices]
