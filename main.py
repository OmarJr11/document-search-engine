# Importa la clase desde pdf-processor.py
from pdf_processor import PDFProcessor

# Crear una instancia de PDFProcessor
processor = PDFProcessor()

# Ruta del archivo PDF (asegúrate de reemplazar esto con la ruta real de tu archivo PDF)
pdf_path = "./archivos/001.pdf"

# Extraer texto del PDF
texto_pdf = processor.extract_text(pdf_path)

# Guardar el texto extraído sin procesar en un archivo .txt
# Ruta del archivo de entrada sin procesar
raw_output_path = "./archivos/entrada_sin_procesar.txt"
with open(raw_output_path, "w", encoding="utf-8") as file:
    file.write(texto_pdf)

# Preprocesar el texto extraído
texto_procesado = processor.preprocess_text(texto_pdf)

# Guardar el texto procesado en un archivo .txt
# Ruta del archivo de salida procesada
processed_output_path = "./archivos/salida_procesada.txt"
with open(processed_output_path, "w", encoding="utf-8") as file:
    file.write(texto_procesado)

# Imprimir mensajes de confirmación
print(f"El texto sin procesar se ha guardado en: {raw_output_path}")
print(f"El texto procesado se ha guardado en: {processed_output_path}")
