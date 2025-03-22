from document_search_facade import DocumentSearchFacade

if __name__ == "__main__":
    # Instanciamos la Fachada
    facade = DocumentSearchFacade()
    # Procesamos e indexamos los documentos
    facade.add_documents()

    # Realizamos una consulta
    consulta = "implmentar buscador"
    resultados = facade.search_documents(consulta)

    # Mostramos los documentos más relevantes

    print(f"Resultados para la consulta: '{consulta}'")
    print("\nDocumentos relevantes:")
    for doc, score in resultados:
        print(f"Score: {score:.4f} - {doc[:200]}...")
