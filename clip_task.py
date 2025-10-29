"""
CLIP (Contrastive Language-Image Pre-training) - Análisis de Similitud Multimodal

Este módulo implementa un pipeline completo para evaluar el modelo CLIP de OpenAI,
que aprende representaciones visuales y textuales mediante entrenamiento contrastivo
en un espacio latente compartido.

Características del Modelo CLIP:
    - Arquitectura dual: Vision Transformer (ViT-B/32) + Text Transformer
    - Entrenamiento contrastivo en 400M pares imagen-texto de internet
    - Embeddings de 512 dimensiones para ambas modalidades
"""

import json
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import CLIPProcessor, CLIPModel

# Configuración de threading para estabilidad en contenedores Docker
# Previene conflictos de paralelización y consumo excesivo de recursos
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)


def load_dataset(path="dataset"):
    """
    Carga el dataset de imágenes y descripciones desde metadata.json.
    
    Lee la configuración del dataset que contiene las rutas relativas de las imágenes,
    sus descripciones textuales (captions) y categorías temáticas asociadas.
    
    Args:
        path (str): Ruta al directorio raíz que contiene metadata.json.
                   Por defecto "dataset".
    
    Returns:
        tuple: Tupla de tres listas:
            - images (list[str]): Rutas absolutas a los archivos de imagen
            - captions (list[str]): Descripciones textuales de cada imagen
            - categories (list[str]): Categorías temáticas de cada imagen
    
    Raises:
        FileNotFoundError: Si metadata.json no existe en la ruta especificada
        json.JSONDecodeError: Si el archivo JSON tiene formato inválido
    
    Example:
        imgs, caps, cats = load_dataset("dataset")
        print(f"Cargadas {len(imgs)} imágenes de {len(set(cats))} categorías")
    """
    with open(Path(path) / "metadata.json", 'r') as f:
        data = json.load(f)
    
    images = [str(Path(path) / item['image']) for item in data]
    captions = [item['caption'] for item in data]
    categories = [item['category'] for item in data]
    return images, captions, categories


class CLIPEmbedder:
    """
    Wrapper para el modelo CLIP que proporciona embeddings unificados imagen-texto.
    
    Esta clase encapsula el modelo CLIP preentrenado (ViT-B/32) y su procesador,
    ofreciendo una interfaz simplificada para generar embeddings en un espacio
    latente compartido de 512 dimensiones.
    """
    
    def __init__(self):
        """
        Inicializa el modelo CLIP y su procesador con pesos preentrenados.
        
        Descarga automáticamente los pesos desde Hugging Face Hub si no están
        en caché local. Configura el modelo en CPU para evitar problemas de
        memoria en contenedores Docker con recursos limitados.
        """
        # Forzar CPU para evitar segmentation faults en Docker con GPU limitada
        self.device = torch.device("cpu")
        print(f"Usando dispositivo: {self.device}")
        
        print("Cargando CLIP...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()
        print("CLIP cargado correctamente")
    
    def encode_images(self, paths):
        """
        Genera embeddings visuales mediante el Vision Transformer de CLIP.
        
        Proceso de encoding:
            1. Carga y conversión a RGB
            2. Redimensionamiento a 224x224
            3. Normalización según estadísticas de CLIP
            4. Patchificación en bloques de 32x32
            5. Encoding mediante ViT (12 capas transformer)
            6. Proyección a espacio latente de 512D
        
        Args:
            paths (list[str]): Lista de rutas a archivos de imagen
        
        Returns:
            np.ndarray: Matriz de embeddings con forma (N, 512) donde N es el
                       número de imágenes. Los embeddings están normalizados.
        
        Note:
            Las imágenes se procesan individualmente para minimizar uso de memoria.
            En producción, se recomienda procesamiento por lotes (batch).
        """
        embs = []
        with torch.no_grad():
            for p in paths:
                img = Image.open(p).convert('RGB')
                inputs = self.processor(images=img, return_tensors="pt")
                embs.append(self.model.get_image_features(**inputs).squeeze().numpy())
        return np.array(embs)
    
    def encode_text(self, texts):
        """
        Genera embeddings textuales mediante el Text Transformer de CLIP.
        
        Args:
            texts (list[str]): Lista de descripciones textuales
        
        Returns:
            np.ndarray: Matriz de embeddings con forma (N, 512) donde N es el
                       número de textos. Los embeddings están normalizados.
        
        Note:
            Los embeddings de texto e imagen residen en el MISMO espacio latente,
            permitiendo comparaciones directas sin necesidad de proyección adicional.
        """
        embs = []
        with torch.no_grad():
            for txt in texts:
                inputs = self.processor(text=txt, return_tensors="pt", padding=True)
                embs.append(self.model.get_text_features(**inputs).squeeze().numpy())
        return np.array(embs)


def compute_metrics(img_emb, txt_emb):
    """
    Calcula métricas de similitud entre embeddings CLIP de imágenes y texto.
    
    Esta función evalúa la capacidad del modelo CLIP para asociar correctamente
    cada imagen con su descripción textual correspondiente mediante similitud coseno.
    
    Ventaja de CLIP:
        A diferencia de Pre-CLIP, los embeddings ya están en el mismo espacio latente
        (512D) gracias al entrenamiento contrastivo conjunto, por lo que NO se requiere
        proyección adicional.
    
    Proceso:
        1. Normalización L2 de embeddings (img y txt)
        2. Cálculo de matriz de similitud mediante producto punto
        3. Extracción de diagonal (similitudes correctas)
        4. Evaluación de precisión Recall@1
    
    Args:
        img_emb (np.ndarray): Embeddings de imágenes con forma (N, 512)
        txt_emb (np.ndarray): Embeddings de texto con forma (N, 512)
    
    Returns:
        tuple: Tupla conteniendo:
            - sim (np.ndarray): Matriz de similitud (N, N) donde sim[i,j] es la
                               similitud coseno entre imagen i y texto j
            - acc (float): Precisión de recuperación (0-1). Fracción de imágenes
                          cuyo texto más similar es el correcto
            - diag (np.ndarray): Similitudes diagonales (N,) representando los
                                pares correctos imagen-texto
    
    Note:
        Recall@1: Para cada imagen, verifica si su texto correspondiente es el
        más similar entre todos los textos disponibles
    """
    # Normalización L2: convierte vectores a norma unitaria
    # Permite usar producto punto como similitud coseno
    img_norm = img_emb / np.linalg.norm(img_emb, axis=1, keepdims=True)
    txt_norm = txt_emb / np.linalg.norm(txt_emb, axis=1, keepdims=True)
    
    # Matriz de similitud coseno: sim[i,j] = cos(theta) entre imagen i y texto j
    # Valores en [-1, 1] donde 1 = máxima similitud, -1 = máxima disimilitud
    sim = img_norm @ txt_norm.T
    
    # Diagonal: similitudes de pares correctos (imagen_i con texto_i)
    diag = np.diag(sim)
    
    # Precisión Recall@1: fracción de imágenes cuyo texto más similar es el correcto
    # np.argmax(sim, axis=1) encuentra el índice del texto más similar para cada imagen
    acc = sum(np.argmax(sim, axis=1) == np.arange(len(sim))) / len(sim)
    
    return sim, acc, diag


def plot_matrix(sim, title, path, cats):
    """
    Genera visualización de matriz de similitud como mapa de calor.
    
    Crea un heatmap que muestra la similitud coseno entre todas las combinaciones
    de imágenes y textos del dataset, facilitando el análisis visual de patrones
    de asociación multimodal y detección de errores de matching.
    
    Args:
        sim (np.ndarray): Matriz de similitud (N, N) con valores en [-1, 1]
        title (str): Título descriptivo del gráfico
        path (str): Ruta de destino para guardar la visualización (formato PNG)
        cats (list[str]): Categorías temáticas para etiquetar ejes
    
    Output:
        Archivo PNG guardado en la ruta especificada.
    
    Note:
        - Diagonal: similitudes correctas (imagen_i con texto_i)
        - Off-diagonal: confusiones o similitudes cruzadas
        - Valores típicos en CLIP: 0.2-0.4 (alta similitud debido a entrenamiento)
    """
    plt.figure(figsize=(10, 8))
    
    # Etiquetas compactas: índice + primeros 8 caracteres de categoría
    labels = [f"{i}-{c[:8]}" for i, c in enumerate(cats)]
    
    # Heatmap con anotaciones numéricas y escala de colores
    sns.heatmap(sim, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=labels, yticklabels=labels)
    
    plt.title(title, pad=15)
    plt.xlabel('Captions')
    plt.ylabel('Imágenes')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """
    Función principal que ejecuta el pipeline completo de evaluación CLIP.
    
    Pipeline de Ejecución:
        1. Carga de dataset (imágenes, captions, categorías)
        2. Inicialización de CLIP y generación de embeddings multimodales
        3. Cálculo de métricas de similitud y precisión
        4. Generación de visualizaciones y persistencia de resultados
    
    Outputs Generados:
        - results/clip_sim.png: Mapa de calor de matriz de similitud
        - results/clip.json: Métricas cuantitativas en formato JSON
    
    Returns:
        None. Los resultados se persisten en el directorio 'results/'.
    
    Raises:
        FileNotFoundError: Si el dataset o metadata.json no existe
        RuntimeError: Si hay errores en carga del modelo o procesamiento
    """
    print("=" * 50)
    print("CLIP: Modelo Multimodal")
    print("=" * 50)
    
    # Crear directorio de salida si no existe
    Path("results").mkdir(exist_ok=True)
    
    # ========== FASE 1: Carga de Datos ==========
    print("\n[1/4] Cargando dataset...")
    imgs, caps, cats = load_dataset()
    print(f"   Cargadas {len(imgs)} imágenes, categorías: {set(cats)}")
    
    # ========== FASE 2: Generación de Embeddings ==========
    print("\n[2/4] Calculando embeddings...")
    embedder = CLIPEmbedder()
    img_emb = embedder.encode_images(imgs)
    txt_emb = embedder.encode_text(caps)
    print(f"   Imágenes: {img_emb.shape}, Texto: {txt_emb.shape}")
    
    # ========== FASE 3: Análisis de Similitud ==========
    print("\n[3/4] Analizando similitud...")
    sim, acc, diag = compute_metrics(img_emb, txt_emb)
    print(f"   Precisión: {acc:.1%}")
    print(f"   Similitud diagonal: {diag.mean():.3f} +/- {diag.std():.3f}")
    
    # ========== FASE 4: Persistencia de Resultados ==========
    print("\n[4/4] Guardando resultados...")
    
    # Generar visualización
    plot_matrix(sim, "CLIP Similarity", "results/clip_sim.png", cats)
    
    # Estructurar resultados para análisis comparativo posterior
    results = {
        "method": "CLIP",
        "accuracy": float(acc),
        "diagonal_mean": float(diag.mean()),
        "diagonal_std": float(diag.std()),
        "diagonal_scores": diag.tolist(),
        "similarity_matrix": sim.tolist()
    }
    
    # Guardar métricas en formato JSON
    with open("results/clip.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Resumen final
    print(f"\n{'='*50}\nCompletado | Precisión: {acc:.1%} | Sim: {diag.mean():.3f}\n{'='*50}")


if __name__ == "__main__":
    main()