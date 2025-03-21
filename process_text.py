from pdf_processor import PDFProcessor
import os

if __name__ == "__main__":
    # Instanciamos la Fachada
    process = PDFProcessor()

    # Ruta de la carpeta donde están los PDFs
    folder_path = "./files"

    # Ruta de la carpeta donde se guardarán los archivos procesados
    processed_folder_path = "./processed_files"
    # Crear la carpeta si no existe
    os.makedirs(processed_folder_path, exist_ok=True)

    # Obtenemos todos los archivos PDF de la carpeta
    pdf_files = [os.path.join(folder_path, file).replace(
        "\\", "/") for file in os.listdir(folder_path) if file.endswith(".pdf")]

    # Procesamos e indexamos los documentos
    for pdf_file in pdf_files:
        # Extraer el texto del PDF y preprocesarlo
        processed_text = process.preprocess_text(
            process.extract_text(pdf_file)
        )

        # Obtener el texto procesado
        # Nombre del archivo de salida
        output_file_name = os.path.splitext(
            os.path.basename(pdf_file))[0] + ".txt"
        output_file_path = os.path.join(
            processed_folder_path, output_file_name)

        # Guardar el texto procesado en un archivo .txt
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(processed_text)
