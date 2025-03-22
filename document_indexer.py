import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class DocumentIndexer:
    """ Clase que maneja la representación vectorial de los documentos y su búsqueda """

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.document_matrix = None
        self.documents = []
        self.documents_as_arrays = []

    def index_documents(self, processed_texts):
        """ Convierte los documentos en vectores TF-IDF """
        self.documents_as_arrays = [doc.split() for doc in processed_texts]
        self.documents = processed_texts
        self.document_matrix = self.vectorizer.fit_transform(processed_texts)
