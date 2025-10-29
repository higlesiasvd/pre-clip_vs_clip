import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def main():
    print("=" * 50)
    print("COMPARACIÓN: Pre-CLIP vs CLIP")
    print("=" * 50)
    
    # Cargar
    pre = load_json("results/preclip.json")
    clip = load_json("results/clip.json")
    
    # Comparar
    print(f"\n{'Métrica':<25} {'Pre-CLIP':>10} {'CLIP':>10} {'Mejora':>10}")
    print("-" * 50)
    print(f"{'Precisión':<25} {pre['accuracy']:>9.1%} {clip['accuracy']:>9.1%} {clip['accuracy']-pre['accuracy']:>+9.1%}")
    print(f"{'Similitud diagonal':<25} {pre['diagonal_mean']:>10.3f} {clip['diagonal_mean']:>10.3f} {clip['diagonal_mean']-pre['diagonal_mean']:>+10.3f}")
    
    # Gráfica
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # Precisión
    methods = ['Pre-CLIP', 'CLIP']
    accs = [pre['accuracy']*100, clip['accuracy']*100]
    bars = ax[0].bar(methods, accs, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    ax[0].set_ylabel('Precisión (%)')
    ax[0].set_title('Precisión de Matching')
    ax[0].set_ylim([0, 100])
    for bar, acc in zip(bars, accs):
        ax[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{acc:.1f}%', ha='center', fontweight='bold')
    
    # Por imagen
    pre_scores = np.array(pre['diagonal_scores'])
    clip_scores = np.array(clip['diagonal_scores'])
    x = np.arange(len(pre_scores))
    w = 0.35
    ax[1].bar(x - w/2, pre_scores, w, label='Pre-CLIP', color='#FF6B6B', alpha=0.8)
    ax[1].bar(x + w/2, clip_scores, w, label='CLIP', color='#4ECDC4', alpha=0.8)
    ax[1].set_xlabel('Imagen')
    ax[1].set_ylabel('Similitud')
    ax[1].set_title('Similitud por Imagen')
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig("results/comparison.png", dpi=300, bbox_inches='tight')
    
    # Guardar comparación
    comp = {
        "preclip": {"accuracy": pre['accuracy'], "similarity": pre['diagonal_mean']},
        "clip": {"accuracy": clip['accuracy'], "similarity": clip['diagonal_mean']},
        "improvement": {
            "accuracy": clip['accuracy'] - pre['accuracy'],
            "similarity": clip['diagonal_mean'] - pre['diagonal_mean']
        }
    }
    
    with open("results/comparison.json", 'w') as f:
        json.dump(comp, f, indent=2)
    
    better = sum(clip_scores > pre_scores)
    print(f"\nCLIP mejoró en {better}/{len(clip_scores)} imágenes ({better/len(clip_scores):.0%})")
    print(f"Guardado: results/comparison.png, results/comparison.json")
    print("=" * 50)


if __name__ == "__main__":
    main()