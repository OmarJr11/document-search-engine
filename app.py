import streamlit as st
from classes.document_search_facade import DocumentSearchFacade
from classes.pdf_processor import PDFProcessor
from functions import process_text

# Instanciar la fachada y el procesador de PDFs
facade = DocumentSearchFacade()
process = PDFProcessor()

# Título de la aplicación
st.title("Buscador y Procesador de Documentos PDF")

# Sección para procesar PDFs
st.header("Procesar documentos PDF")
if st.button("Procesar PDFs"):
    pdf_files = process_text.get_pdf_files()  # Obtener los archivos PDF
    if pdf_files:
        st.write("Procesando documentos...")
        # Llamar a la función para procesar los PDFs
        process_text.process_pdfs(pdf_files)
        st.success(
            "¡Todos los PDFs han sido procesados y guardados en la carpeta 'processed_files'!")
    else:
        st.warning("No se encontraron archivos PDF en la carpeta 'files'.")

# Sección para realizar búsquedas
st.header("Realizar una búsqueda")
consulta = st.text_input("Introduce tu consulta:", "")

if st.button("Buscar"):
    if consulta.strip():
        # Procesar e indexar los documentos antes de buscar
        st.write("Procesando e indexando documentos...")
        facade.add_documents()
        st.success("Documentos procesados e indexados con éxito.")

        # Realizar la búsqueda
        resultados = facade.search_documents(consulta)

        # Mostrar los resultados
        st.subheader(f"Resultados para la consulta: '{consulta}'")
        st.subheader(f"{len(resultados)} documentos encontrados.")
        if resultados:
            for doc, score in resultados:
                resumen = facade.generate_summary(doc, consulta)
                resumen_resaltado = facade.highlight_keywords(
                    resumen, consulta, color="blue")

                st.write(f"**Relevancia:** {score:.4f}")
                st.markdown(resumen_resaltado, unsafe_allow_html=True)
                st.write("---")
        else:
            st.warning("No se encontraron documentos relevantes.")
    else:
        st.error("Por favor, introduce una consulta válida.")

# Sección para visualizar los clusters
st.header("Visualización de Clusters")
if st.button("Mostrar Clusters"):
    if hasattr(facade, "clusters") and facade.clusters is not None:
        st.write("Documentos agrupados en clusters:")
        for cluster_id, docs in facade.clustered_documents.items():
            st.subheader(f"Cluster {cluster_id}")
            for doc in docs[:3]:  # Mostrar los primeros 3 documentos de cada cluster
                # Mostrar los primeros 200 caracteres de cada documento
                st.write(f"- {doc[:200]}...")
            st.write("---")
    else:
        st.warning(
            "No se han generado clusters. Procesa e indexa los documentos primero.")

# Sección para mostrar estadísticas de clusters
st.header("Estadísticas de Clusters")
if hasattr(facade, "clusters") and facade.clusters is not None:
    for cluster_id, docs in facade.clustered_documents.items():
        st.write(f"Cluster {cluster_id}: {len(docs)} documentos")
else:
    st.warning(
        "No se han generado clusters. Procesa e indexa los documentos primero.")
