"""
Pre-CLIP: Baseline para Similitud Imagen-Texto

Este módulo implementa un sistema de baseline para comparación con CLIP, utilizando
arquitecturas independientes para procesar imágenes y texto sin entrenamiento conjunto.

El enfoque Pre-CLIP combina:
    - ResNet-50 (preentrenado en ImageNet) para embeddings visuales
    - Sentence-BERT (all-MiniLM-L6-v2) para embeddings textuales
    - Proyección lineal para alinear espacios de características

Este baseline permite evaluar las mejoras del entrenamiento contrastivo de CLIP
frente a modelos preentrenados independientemente.

Características:
    - Embeddings de imagen: 2048D (ResNet-50) proyectados a 384D
    - Embeddings de texto: 384D (Sentence-BERT)
    - Similitud coseno para matching imagen-texto
    - Configuración optimizada para ejecución en Docker/CPU
"""

import json
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models, transforms
from sentence_transformers import SentenceTransformer

# Configuración de threading para evitar conflictos de memoria en contenedores
# Limita los threads de bibliotecas numéricas para prevenir overconsumption de CPU
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)


def load_dataset(path="dataset"):
    """
    Carga el dataset de imágenes y descripciones desde metadata.json.
    
    Lee la configuración del dataset que contiene las rutas relativas de las imágenes,
    sus descripciones textuales (captions) y categorías temáticas.
    
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
    """
    with open(Path(path) / "metadata.json", 'r') as f:
        data = json.load(f)
    
    images = [str(Path(path) / item['image']) for item in data]
    captions = [item['caption'] for item in data]
    categories = [item['category'] for item in data]
    return images, captions, categories


class PreCLIPEmbedder:
    """
    Generador de embeddings independientes para imágenes y texto (baseline Pre-CLIP).
    
    Esta clase implementa un enfoque de doble encoder sin entrenamiento conjunto,
    combinando arquitecturas preentrenadas de forma independiente:
    
    Encoder de Imagen:
        - ResNet-50 preentrenado en ImageNet (sin capa de clasificación)
        - Genera embeddings de 2048 dimensiones
        - Normalización según estadísticas de ImageNet
    
    Encoder de Texto:
        - Sentence-BERT: all-MiniLM-L6-v2
        - Genera embeddings de 384 dimensiones
        - Optimizado para similitud semántica de oraciones
    
    Limitación Principal:
        Los espacios latentes de imagen (2048D) y texto (384D) son incompatibles.
        Se requiere proyección lineal para alineamiento, lo cual es subóptimo
        comparado con el entrenamiento conjunto de CLIP.
    
    Attributes:
        device (torch.device): Dispositivo de cómputo (CUDA si disponible, sino CPU)
        img_model (torch.nn.Sequential): ResNet-50 sin clasificador final
        transform (transforms.Compose): Pipeline de preprocesamiento de imágenes
        txt_model (SentenceTransformer): Modelo Sentence-BERT para texto
    """
    
    def __init__(self):
        """
        Inicializa los modelos de imagen y texto con pesos preentrenados.
        
        Descarga automáticamente los pesos desde repositorios oficiales si no
        están en caché local. Configura los modelos en modo evaluación para
        deshabilitar dropout y batch normalization.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ResNet-50: Extraer features antes de la capa de clasificación
        # Salida: (batch_size, 2048, 1, 1) -> squeeze -> (batch_size, 2048)
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        self.img_model = torch.nn.Sequential(*list(model.children())[:-1])
        self.img_model.to(self.device).eval()

        # Pipeline de transformación estándar de ImageNet
        self.transform = transforms.Compose([
            transforms.Resize(256),           # Redimensionar lado corto a 256
            transforms.CenterCrop(224),       # Crop central de 224x224
            transforms.ToTensor(),            # Convertir a tensor [0,1]
            transforms.Normalize(             # Normalizar con estadísticas ImageNet
                [0.485, 0.456, 0.406],       # Media RGB
                [0.229, 0.224, 0.225]        # Desviación estándar RGB
            )
        ])
        
        # Sentence-BERT: Modelo ligero optimizado para similitud semántica
        self.txt_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def encode_images(self, paths):
        """
        Genera embeddings visuales para un conjunto de imágenes.
        
        Procesa cada imagen a través de ResNet-50, aplicando:
            1. Transformaciones de preprocesamiento (resize, crop, normalización)
            2. Forward pass por la red convolucional
            3. Global average pooling (implícito en ResNet)
            4. Extracción del vector de características final
        
        Args:
            paths (list[str]): Lista de rutas a archivos de imagen
        
        Returns:
            np.ndarray: Matriz de embeddings con forma (N, 2048) donde N es el
                       número de imágenes procesadas.
        
        Note:
            Las imágenes se convierten a RGB automáticamente para garantizar
            3 canales de entrada consistentes con el entrenamiento de ResNet.
        """
        embs = []
        with torch.no_grad():
            for p in paths:
                img = self.transform(Image.open(p).convert('RGB')).unsqueeze(0)
                embs.append(self.img_model(img).squeeze().numpy())
        return np.array(embs)
    
    def encode_text(self, texts):
        """
        Genera embeddings semánticos para un conjunto de textos.
        
        Procesa cada texto mediante Sentence-BERT, aplicando:
            1. Tokenización con vocabulario WordPiece
            2. Encoding mediante Transformer (6 capas, 384 dimensiones)
            3. Mean pooling de los estados ocultos
            4. Normalización L2 del embedding resultante
        
        Args:
            texts (list[str]): Lista de descripciones textuales
        
        Returns:
            np.ndarray: Matriz de embeddings con forma (N, 384) donde N es el
                       número de textos procesados.
        
        Note:
            Los embeddings están optimizados para capturar similitud semántica
            de oraciones completas, no solo palabras individuales.
        """
        return self.txt_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def compute_metrics(img_emb, txt_emb):
    """
    Calcula métricas de similitud con proyección de espacio latente cuando es necesario.
    
    Esta función maneja la incompatibilidad dimensional entre los embeddings de
    ResNet-50 (2048D) y Sentence-BERT (384D) mediante proyección lineal aleatoria.
    
    Proceso:
        1. Detección de incompatibilidad dimensional
        2. Proyección lineal de imágenes al espacio de texto (si necesario)
        3. Normalización L2 de ambos conjuntos de embeddings
        4. Cálculo de matriz de similitud coseno
        5. Evaluación de precisión de recuperación
    
    Args:
        img_emb (np.ndarray): Embeddings de imágenes con forma (N, D_img)
        txt_emb (np.ndarray): Embeddings de texto con forma (N, D_txt)
    
    Returns:
        tuple: Tupla conteniendo:
            - sim (np.ndarray): Matriz de similitud (N, N) donde sim[i,j]
                               representa la similitud coseno entre imagen i y texto j
            - acc (float): Precisión de recuperación (0-1). Fracción de imágenes
                          cuyo texto más similar es el correcto
            - diag (np.ndarray): Similitudes diagonales (N,) correspondientes
                                a los pares correctos imagen-texto
    
    Note:
        La proyección aleatoria es subóptima comparada con proyecciones aprendidas,
        pero sirve como baseline razonable para comparación con CLIP.
    """
    # Alineamiento dimensional mediante proyección lineal si es necesario
    if img_emb.shape[1] != txt_emb.shape[1]:
        target_dim = txt_emb.shape[1]
        print(f"   Proyectando imágenes: {img_emb.shape[1]}D -> {target_dim}D (proyección lineal)")
        
        # Mantiene varianza aproximadamente constante durante la proyección
        np.random.seed(42)  # Reproducibilidad
        projection = np.random.randn(img_emb.shape[1], target_dim) / np.sqrt(img_emb.shape[1])
        img_emb = img_emb @ projection
    
    # Normalización L2: convierte cada vector a norma unitaria
    # Permite usar producto punto como similitud coseno eficientemente
    img_norm = img_emb / np.linalg.norm(img_emb, axis=1, keepdims=True)
    txt_norm = txt_emb / np.linalg.norm(txt_emb, axis=1, keepdims=True)
    
    # Matriz de similitud: sim[i,j] = cos(theta) entre imagen i y texto j
    # Valores en [-1, 1] donde 1 = idénticos, 0 = ortogonales, -1 = opuestos
    sim = img_norm @ txt_norm.T
    
    # Diagonal: similitudes de pares correctos (imagen_i con texto_i)
    diag = np.diag(sim)
    
    # Precisión Recall@1: fracción de imágenes cuyo texto más similar es el correcto
    acc = sum(np.argmax(sim, axis=1) == np.arange(len(sim))) / len(sim)
    
    return sim, acc, diag


def plot_matrix(sim, title, path, cats):
    """
    Genera visualización de matriz de similitud como mapa de calor.
    
    Crea un heatmap que muestra la similitud coseno entre todas las combinaciones
    de imágenes y textos del dataset, facilitando el análisis visual de patrones
    de asociación y errores de matching.
    
    Args:
        sim (np.ndarray): Matriz de similitud (N, N) con valores en [-1, 1]
        title (str): Título descriptivo del gráfico
        path (str): Ruta de destino para guardar la visualización (formato PNG)
        cats (list[str]): Categorías temáticas para etiquetar ejes
    
    Output:
        Archivo PNG guardado en la ruta especificada.
    
    Note:
        - Diagonal: similitudes correctas (imagen_i con texto_i)
        - Off-diagonal: posibles confusiones o similitudes cruzadas
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
    Función principal que ejecuta el pipeline completo de Pre-CLIP.
    
    Pipeline de Ejecución:
        1. Carga de dataset (imágenes, captions, categorías)
        2. Generación de embeddings independientes (ResNet-50 + Sentence-BERT)
        3. Proyección a espacio común y cálculo de métricas
        4. Visualización y guardado de resultados
    
    Outputs Generados:
        - results/preclip_sim.png: Mapa de calor de matriz de similitud
        - results/preclip.json: Métricas cuantitativas (precisión, similitudes)
    
    Returns:
        None. Los resultados se persisten en el directorio 'results/'.
    
    Raises:
        FileNotFoundError: Si el dataset o metadata.json no existe
        RuntimeError: Si hay errores en carga de modelos o procesamiento
    """
    print("=" * 50)
    print("PRE-CLIP: ResNet-50 + Sentence-Transformers")
    print("=" * 50)
    
    # Crear directorio de salida si no existe
    Path("results").mkdir(exist_ok=True)
    
    # ========== FASE 1: Carga de Datos ==========
    print("\n[1/4] Cargando dataset...")
    imgs, caps, cats = load_dataset()
    print(f"   Cargadas {len(imgs)} imágenes, categorías: {set(cats)}")
    
    # ========== FASE 2: Generación de Embeddings ==========
    print("\n[2/4] Calculando embeddings...")
    embedder = PreCLIPEmbedder()
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
    plot_matrix(sim, "Pre-CLIP Similarity", "results/preclip_sim.png", cats)
    
    # Estructurar resultados para análisis posterior
    results = {
        "method": "Pre-CLIP",
        "accuracy": float(acc),
        "diagonal_mean": float(diag.mean()),
        "diagonal_std": float(diag.std()),
        "diagonal_scores": diag.tolist(),
        "similarity_matrix": sim.tolist()
    }
    
    # Guardar métricas en formato JSON
    with open("results/preclip.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Resumen final
    print(f"\n{'='*50}\nCompletado | Precisión: {acc:.1%} | Sim: {diag.mean():.3f}\n{'='*50}")


if __name__ == "__main__":
    main()