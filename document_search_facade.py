class DocumentSearchFacade:
    """ Fachada que centraliza la extracción, preprocesamiento, indexación y búsqueda """

    def __init__(self):
        self.processor = PDFProcessor()
        self.indexer = DocumentIndexer()
        self.processed_documents = []

    def add_documents(self, pdf_paths):
        """ Extrae, limpia e indexa los documentos """
        texts = [self.processor.extract_text(pdf) for pdf in pdf_paths]
        self.processed_documents = [
            self.processor.preprocess_text(text) for text in texts]
        self.indexer.index_documents(self.processed_documents)

    def search_documents(self, query):
        """ Realiza una búsqueda con la consulta dada """
        query_processed = self.processor.preprocess_text(query)
        return self.indexer.search(query_processed)
