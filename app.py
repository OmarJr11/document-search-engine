import streamlit as st
from classes.document_search_facade import DocumentSearchFacade
from classes.pdf_processor import PDFProcessor
from functions import process_text
from functions.evaluation import evaluate_recommendations, ndcg

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

# Inicializar el estado de la sesión para resultados, opciones y selección
if "resultados" not in st.session_state:
    st.session_state["resultados"] = []
if "opciones" not in st.session_state:
    st.session_state["opciones"] = []
if "selected_option" not in st.session_state:
    st.session_state["selected_option"] = "Selecciona un documento"
if "vectorizer" not in st.session_state:
    st.session_state["vectorizer"] = None
if "document_matrix" not in st.session_state:
    st.session_state["document_matrix"] = None
if "processed_documents" not in st.session_state:
    st.session_state["processed_documents"] = []
if "clusters" not in st.session_state:
    st.session_state["clusters"] = None
if "clustered_documents" not in st.session_state:
    st.session_state["clustered_documents"] = {}
if "recomendaciones" not in st.session_state:
    st.session_state["recomendaciones"] = []
if "recommended_docs" not in st.session_state:
    st.session_state["recommended_docs"] = []
if "relevant_docs" not in st.session_state:
    st.session_state["relevant_docs"] = []

if st.button("Buscar"):
    if consulta.strip():
        # Procesar e indexar los documentos antes de buscar
        st.write("Procesando e indexando documentos...")
        facade.add_documents()
        st.success("Documentos procesados e indexados con éxito.")

        # Realizar clustering de los documentos
        st.write("Generando clusters de documentos...")
        facade.perform_clustering()
        st.success("Clusters generados con éxito.")

        # Realizar la búsqueda
        resultados = facade.search_documents(consulta)
        # Guardar resultados en el estado de la sesión
        st.session_state["resultados"] = resultados
        # Guardar los clusters en el estado de la sesión
        st.session_state["clusters"] = facade.clusters
        st.session_state["clustered_documents"] = facade.clustered_documents

        # Restablecer la selección actual
        st.session_state["selected_option"] = "Selecciona un documento"

        # Crear una lista de opciones para el radio con relevancia y resumen
        opciones = ["Selecciona un documento"] + [
            f"Documento {i+1}: Relevancia {score:.4f} - {doc[:100]}..."
            for i, (doc, score) in enumerate(resultados)
        ]
        # Guardar opciones en el estado de la sesión
        st.session_state["opciones"] = opciones

        # Mostrar los resultados
        st.subheader(f"Resultados para la consulta: '{consulta}'")
        st.subheader(f"{len(resultados)} documentos encontrados.")

# Usar los resultados y opciones almacenados en el estado de la sesión
if st.session_state["opciones"]:
    # Permitir al usuario seleccionar un documento
    selected_option = st.radio(
        "Selecciona un documento para recomendar documentos:",
        st.session_state["opciones"],
        index=st.session_state["opciones"].index(
            st.session_state["selected_option"])
    )

    # Guardar la selección en el estado de la sesión
    st.session_state["selected_option"] = selected_option

    # Verificar si el usuario seleccionó un documento válido
    if selected_option != "Selecciona un documento":
        # Obtener el índice del documento seleccionado
        selected_index = st.session_state["opciones"].index(
            selected_option) - 1
        selected_doc, selected_score = st.session_state["resultados"][selected_index]

        # Mostrar información del documento seleccionado
        st.success(f"Has seleccionado el documento {selected_index + 1}.")
        st.write(f"**Relevancia:** {selected_score:.4f}")
        st.write(f"**Documento:** {selected_doc[:300]}...")

        # Botón para buscar documentos similares
        if st.button("Buscar documentos similares"):
            # Verificar si los datos necesarios están en el estado de la sesión
            if "vectorizer" in st.session_state and "document_matrix" in st.session_state:
                # Verificar si la matriz de documentos no está vacía
                if st.session_state["document_matrix"].shape[0] > 0:
                    recomendaciones = facade.recommend_similar_documents(
                        selected_doc)
                    # Guardar las recomendaciones en el estado de la sesión
                    st.session_state["recomendaciones"] = recomendaciones

                    st.subheader("Documentos similares:")
                    if recomendaciones:
                        for similar_doc, similarity_score in recomendaciones:
                            st.write(f"**Similitud:** {similarity_score:.4f}")
                            st.write(f"**Documento:** {similar_doc[:300]}...")
                            st.write("---")
                    else:
                        st.warning("No se encontraron documentos similares.")
                else:
                    st.error(
                        "La matriz de documentos está vacía. Por favor, realiza una búsqueda primero.")
            else:
                st.error(
                    "Los datos necesarios no están disponibles. Por favor, realiza una búsqueda primero.")

        # Evaluar las recomendaciones
        if st.button("Evaluar recomendaciones"):
            # Verificar si hay recomendaciones disponibles
            if "recomendaciones" in st.session_state:
                recomendaciones = st.session_state["recomendaciones"]

                # Obtener los documentos procesados
                processed_documents = st.session_state["processed_documents"]

                # Consulta actual
                query = consulta.lower()

                # Definir documentos relevantes basados en reglas
                relevant_docs = [
                    doc for doc in processed_documents if query in doc.lower()]
                # Guardar en session_state
                st.session_state["relevant_docs"] = relevant_docs

                # Extraer solo los textos de los documentos recomendados
                recommended_docs = [doc for doc, _ in recomendaciones]
                # Guardar en session_state
                st.session_state["recommended_docs"] = recommended_docs

                # Evaluar las recomendaciones
                metrics = evaluate_recommendations(
                    recommended_docs, relevant_docs)

                # Mostrar las métricas
                st.subheader("Evaluación de las recomendaciones:")
                st.write(f"**Precisión:** {metrics['precision']:.4f}")
                st.write(f"**Cobertura:** {metrics['recall']:.4f}")
                st.write(f"**F1-Score:** {metrics['f1_score']:.4f}")
            else:
                st.warning(
                    "No hay recomendaciones disponibles para evaluar. Por favor, busca documentos similares primero.")

        # Evaluar NDCG
        if st.button("Evaluar NDCG"):
            if "recommended_docs" in st.session_state and "relevant_docs" in st.session_state:
                recommended_docs = st.session_state["recommended_docs"]
                relevant_docs = st.session_state["relevant_docs"]

                # Calcular NDCG
                ndcg_score = ndcg(recommended_docs, relevant_docs, k=10)
                st.write(f"**NDCG:** {ndcg_score:.4f}")
            else:
                st.warning(
                    "No hay datos disponibles para calcular NDCG. Por favor, evalúa las recomendaciones primero.")

# Sección para mostrar estadísticas de clusters
st.header("Estadísticas de Clusters")
if "clusters" in st.session_state and "clustered_documents" in st.session_state:
    clustered_documents = st.session_state["clustered_documents"]
    for cluster_id, docs in clustered_documents.items():
        st.write(f"Cluster {cluster_id}: {len(docs)} documentos")
else:
    st.warning(
        "No se han generado clusters. Procesa e indexa los documentos primero."
    )
