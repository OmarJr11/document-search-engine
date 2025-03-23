import numpy as np


def evaluate_recommendations(recommended_docs, relevant_docs):
    """
    Evalúa la calidad de las recomendaciones usando precisión, cobertura y F1-Score.

    Args:
        recommended_docs (list): Lista de documentos recomendados.
        relevant_docs (list): Lista de documentos relevantes.

    Returns:
        dict: Diccionario con las métricas de evaluación.
    """
    # Convertir a conjuntos para facilitar las operaciones
    recommended_set = set(recommended_docs)
    relevant_set = set(relevant_docs)

    # Calcular intersección
    true_positives = recommended_set.intersection(relevant_set)

    # Calcular precisión
    precision = len(true_positives) / \
        len(recommended_set) if recommended_set else 0.0

    # Calcular cobertura
    recall = len(true_positives) / len(relevant_set) if relevant_set else 0.0

    # Calcular F1-Score
    f1_score = (2 * precision * recall) / (precision +
                                           recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }


def ndcg(recommended_docs, relevant_docs, k=10):
    """
    Calcula el NDCG (Normalized Discounted Cumulative Gain) para las recomendaciones.

    Args:
        recommended_docs (list): Lista de documentos recomendados en orden.
        relevant_docs (list): Lista de documentos relevantes.
        k (int): Número máximo de documentos a considerar.

    Returns:
        float: NDCG para las recomendaciones.
    """
    # Convertir a conjuntos para facilitar las operaciones
    relevant_set = set(relevant_docs)

    # Calcular DCG
    dcg = 0.0
    for i, doc in enumerate(recommended_docs[:k]):
        if doc in relevant_set:
            dcg += 1 / np.log2(i + 2)  # i+2 porque log2(1) no está definido

    # Calcular IDCG (DCG ideal)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_set), k)))

    # Calcular NDCG
    return dcg / idcg if idcg > 0 else 0.0
