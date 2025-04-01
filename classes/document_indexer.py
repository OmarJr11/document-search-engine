from sklearn.feature_extraction.text import TfidfVectorizer


class DocumentIndexer:
    """ Clase que maneja la representación vectorial de los documentos y su búsqueda """

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.document_matrix = None
        self.documents = []
        self.documents_as_arrays = []
