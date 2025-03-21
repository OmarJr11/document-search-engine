# PyMuPDF para extracción de texto de PDFs.
import fitz
# Para procesamiento de texto, incluyendo tokenización y lematización.
import spacy

""" Clase encargada de extraer y preprocesar texto de PDFs """


class PDFProcessor:
    """ Este método inicializa la clase cargando el modelo de lenguaje en español (es_core_news_sm) de spaCy. Este modelo incluye herramientas para analizar texto en español, como lematización y detección de palabras vacías."""

    def __init__(self):
        self.nlp = spacy.load("es_core_news_sm")

    """ 
        Este método extrae el texto de un archivo PDF utilizando la biblioteca PyMuPDF (fitz). Abre el PDF, lee cada página y concatena el texto extraído en una sola cadena. Finalmente, devuelve el texto completo del PDF. 
    """
    # Entrada:
    # Recibe la ruta de un archivo PDF (pdf_path).
    # Proceso:
    # Abre el archivo PDF utilizando fitz.open(pdf_path).
    # Itera sobre todas las páginas del PDF y extrae el texto de cada una con page.get_text().
    # Une el texto de todas las páginas en una sola cadena.
    # Salida:
    # Devuelve el texto completo del PDF como una cadena.

    def extract_text(self, pdf_path):
        doc = fitz.open(pdf_path)
        text = " ".join([page.get_text() for page in doc])
        return text

    """
        Este método procesa el texto extraído del PDF. Convierte el texto a minúsculas, elimina las palabras vacías (stopwords) y aplica lematización para reducir las palabras a su forma base.
    """
    # Entrada:
    # Recibe una cadena de texto (text).
    # Proceso:
    # Convierte el texto a minúsculas.
    # Utiliza el modelo de spaCy para procesar el texto.
    # Filtra las palabras para eliminar stopwords y puntuación.
    # Aplica lematización a las palabras restantes.
    # Salida:
    # Devuelve el texto procesado como una cadena.

    def preprocess_text(self, text):
        doc = self.nlp(text.lower())
        return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
