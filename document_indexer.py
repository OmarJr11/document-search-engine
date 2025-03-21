import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DocumentIndexer:
    """ Clase que maneja la representación vectorial de los documentos y su búsqueda """

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.document_matrix = None
        self.documents = []

    def index_documents(self, processed_texts):
        """ Convierte los documentos en vectores TF-IDF """
        self.documents = processed_texts
        self.document_matrix = self.vectorizer.fit_transform(processed_texts)

    def search(self, query, top_k=5):
        """ Busca documentos similares a la consulta usando similitud de coseno """
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(
            query_vector, self.document_matrix).flatten()
        top_indices = np.argsort(similarities)[
            ::-1][:top_k]  # Ordena por similitud
        return [(self.documents[i], similarities[i]) for i in top_indices]
