from classes.pdf_processor import PDFProcessor
import os

# Rutas de las carpetas
folder_path = "./files"  # Carpeta donde están los PDFs
# Carpeta donde se guardarán los archivos procesados
processed_folder_path = "./processed_files"

# Crear la carpeta de salida si no existe
os.makedirs(processed_folder_path, exist_ok=True)

# Instanciar el procesador de PDFs
process = PDFProcessor()


def get_pdf_files():
    """
    Obtiene todos los archivos PDF de la carpeta especificada.
    """
    return [os.path.join(folder_path, file).replace(
        "\\", "/") for file in os.listdir(folder_path) if file.endswith(".pdf")]


def process_pdfs(pdf_files):
    """
    Procesa una lista de archivos PDF y guarda los textos procesados en archivos .txt.
    """
    for pdf_file in pdf_files:
        # Extraer el texto del PDF y preprocesarlo
        processed_text = process.preprocess_text(
            process.extract_text(pdf_file)
        )

        # Nombre del archivo de salida
        output_file_name = os.path.splitext(
            os.path.basename(pdf_file))[0] + ".txt"
        output_file_path = os.path.join(
            processed_folder_path, output_file_name)

        # Guardar el texto procesado en un archivo .txt
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(processed_text)
