if __name__ == "__main__":
    # Instanciamos la Fachada
    facade = DocumentSearchFacade()

    # Lista de documentos PDF (Asegúrate de tener estos archivos en el mismo directorio)
    pdf_files = ["documento1.pdf", "documento2.pdf"]

    # Procesamos e indexamos los documentos
    facade.add_documents(pdf_files)

    # Realizamos una consulta
    consulta = "inteligencia artificial en PYMEs"
    resultados = facade.search_documents(consulta)

    # Mostramos los documentos más relevantes
    print("\nDocumentos relevantes:")
    for doc, score in resultados:
        print(f"Score: {score:.4f} - {doc[:200]}...")
