"""
Script per visualizzare gli embedding del dataset con UMAP e Gaussian KDE.
Genera grafici per esperimenti specifici, replicando il layout a cerchio.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import umap
from scipy.stats import gaussian_kde

# Colori definiti per ciascun encoder
ENCODER_COLORS: Dict[str, str] = {
    "audioclip": "#1f77b4",
    "vggish": "#ff7f0e",
    "clip": "#2ca02c",
    "vit": "#9467bd",
    "minilm": "#e377c2",
}

# Esperimenti di default
DEFAULT_EXPERIMENTS: List[Dict[str, object]] = [
    # MovieLens 1M
    {"name": "LATTICE_text_minilm", "dataset": "movielens_1m", "embeddings": ["text_minilm"]},
    {"name": "LATTICE_audio_vggish", "dataset": "movielens_1m", "embeddings": ["audio_vggish"]},
    {"name": "LATTICE_image_vit", "dataset": "movielens_1m", "embeddings": ["image_vit"]},
    {
        "name": "LATTICE_text_minilm_image_vit",
        "dataset": "movielens_1m",
        "embeddings": ["text_minilm", "image_vit"],
    },
    {
        "name": "LATTICE_text_clip_image_clip",
        "dataset": "movielens_1m",
        "embeddings": ["text_clip", "image_clip"],
    },
    {
        "name": "LATTICE_text_minilm_image_vit_audio_vggish",
        "dataset": "movielens_1m",
        "embeddings": ["text_minilm", "image_vit", "audio_vggish"],
    },
    {
        "name": "LATTICE_text_clip_image_clip_audio_vggish",
        "dataset": "movielens_1m",
        "embeddings": ["audio_vggish", "image_clip", "text_clip"],
    },
    {
        "name": "LATTICE_audioclip_full",
        "dataset": "movielens_1m",
        "embeddings": ["audio_audioclip", "image_audioclip", "text_audioclip"],
    },
    # LastFM
    {"name": "LATTICE_text_minilm", "dataset": "lastfm", "embeddings": ["text_minilm"]},
    {"name": "LATTICE_image_vit", "dataset": "lastfm", "embeddings": ["image_vit"]},
    {
        "name": "LATTICE_text_minilm_image_vit",
        "dataset": "lastfm",
        "embeddings": ["text_minilm", "image_vit"],
    },
    {
        "name": "LATTICE_text_clip_image_clip",
        "dataset": "lastfm",
        "embeddings": ["text_clip", "image_clip"],
    },
    {
        "name": "LATTICE_audioclip_full",
        "dataset": "lastfm",
        "embeddings": ["audio_audioclip", "image_audioclip", "text_audioclip"],
    },
]

DATA_ROOT = Path(__file__).resolve().parent


def normalize_dataset_name(dataset: Optional[str]) -> Optional[str]:
    """Normalizza il nome del dataset usando alias predefiniti."""
    aliases = {
        "movielens": "movielens_1m",
        "lastfm": "lastfm",
    }
    if dataset is None:
        return None
    return aliases.get(dataset.lower(), dataset)


def select_experiments(dataset_filter: Optional[str], experiment_name: Optional[str]):
    """Filtra gli esperimenti predefiniti in base a dataset e/o nome esperimento."""
    dataset_norm = normalize_dataset_name(dataset_filter)

    selected = []
    for exp in DEFAULT_EXPERIMENTS:
        if dataset_norm and exp["dataset"] != dataset_norm:
            continue
        if experiment_name and exp["name"] != experiment_name:
            continue
        selected.append(exp)
    return selected


def load_embeddings_for_experiment(dataset_path: Path, embedding_names, cache):
    """Carica file .npy degli embeddings, usando cache per evitare ricaricamenti."""
    loaded = {}
    for emb_name in embedding_names:
        file_path = dataset_path / f"{emb_name}.npy"
        if not file_path.exists():
            print(f" ✗ {emb_name} non trovato ({file_path})")
            continue

        key = str(file_path.resolve())
        if key not in cache:
            cache[key] = np.load(file_path)

        loaded[emb_name] = cache[key]
        print(f" ✓ {emb_name}: {loaded[emb_name].shape}")
    return loaded


def reduce_embeddings_to_2d(
    embeddings: Dict[str, np.ndarray],
    *,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
    metric: str,
) -> Dict[str, np.ndarray]:
    """Riduce embeddings ad alta dimensione a 2D usando UMAP.
    
    Applica UMAP per riduzione dimensionale e normalizza i punti risultanti
    sul cerchio unitario S1 (centra e normalizza per norma euclidea).
    Questo permette l'analisi della distribuzione angolare.
    
    Args:
        embeddings: Dizionario {nome: matrice_embedding}
        n_neighbors: Numero di vicini per UMAP
        min_dist: Distanza minima per UMAP
        random_state: Seed per riproducibilità
        metric: Metrica di distanza (euclidean raccomandato con normalizzazione S1)
    
    Returns:
        Dizionario {nome: punti_2D_normalizzati_su_S1}
    """
    reduced = {}

    for name, matrix in embeddings.items():
        print(f"  UMAP -> {name}")
        # Use random init to avoid spectral initialisation failures on small eigengaps.
        # Note: setting random_state forces n_jobs to 1 (UMAP limitation).
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            n_components=2,
            metric=metric,
            init="random",
        )

        reduced_data = reducer.fit_transform(matrix)

        # ✅ MODIFICATO: normalizzazione S1 obbligatoria
        centered = reduced_data - reduced_data.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        reduced_data = centered / norms

        reduced[name] = reduced_data

    return reduced


def plot_experiment_kde(
    reduced_embeddings: Dict[str, np.ndarray],
    exp_name: str,
    dataset_name: str,
    output_dir: Path,
) -> None:
    """Crea visualizzazione con scatter plot, KDE 2D e KDE angolare.
    
    Genera un grafico a griglia 2×N dove:
    - Riga superiore: scatter plot con contorni KDE 2D
    - Riga inferiore: KDE della distribuzione angolare su S1
    
    Args:
        reduced_embeddings: Dizionario {nome: punti_2D_normalizzati}
        exp_name: Nome dell'esperimento
        dataset_name: Nome del dataset
        output_dir: Directory di output per salvare il grafico
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_items = sorted(reduced_embeddings.items())
    if not embedding_items:
        print(" ⊘ Nessun embedding ridotto, salto")
        return

    n_cols = len(embedding_items)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 6))
    if n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for idx, (full_name, reduced_data) in enumerate(embedding_items):
        ax_scatter = axes[0, idx]
        ax_density = axes[1, idx]

        # ✅ Colore robusto
        encoder = full_name.split("_")[-1]
        color = ENCODER_COLORS.get(encoder, "#7f7f7f")

        # Scatter
        ax_scatter.scatter(
            reduced_data[:, 0],
            reduced_data[:, 1],
            alpha=0.6,
            s=16,
            c=color,
            edgecolors="none",
        )

        # ✅ KDE 2D migliorata (contourf)
        if len(reduced_data) > 1:
            try:
                kde = gaussian_kde(reduced_data.T)
                x_min, x_max = reduced_data[:, 0].min(), reduced_data[:, 0].max()
                y_min, y_max = reduced_data[:, 1].min(), reduced_data[:, 1].max()

                xx, yy = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                f = np.reshape(kde(positions).T, xx.shape)

                ax_scatter.contourf(xx, yy, f, levels=50, alpha=0.4, cmap="Greens")

            except Exception as exc:
                print(f" ⚠ KDE 2D non calcolato per {full_name}: {exc}")
                ax_scatter.text(0, 0, "KDE non disponibile", ha="center", va="center",
                               fontsize=10, alpha=0.5, color="red")

        ax_scatter.set_title(full_name, fontsize=12, fontweight="bold")
        ax_scatter.set_xlabel("Feature 1")
        ax_scatter.set_ylabel("Feature 2")
        ax_scatter.set_aspect("equal", "box")
        ax_scatter.grid(True, alpha=0.15)

        # KDE sugli angoli (distribuzione angolare su S1)
        if len(reduced_data) > 1:
            angles = np.arctan2(reduced_data[:, 1], reduced_data[:, 0])

            try:
                kde_angles = gaussian_kde(angles)
                angle_range = np.linspace(-np.pi, np.pi, 300)
                density = kde_angles(angle_range)
                ax_density.plot(angle_range, density, color=color, linewidth=2.0)
                ax_density.fill_between(angle_range, density, alpha=0.25, color=color)
            except Exception as exc:
                print(f" ⚠ KDE angolare non calcolato per {full_name}: {exc}")
                ax_density.text(0, 0.5, "KDE non disponibile", ha="center", va="center",
                               fontsize=10, alpha=0.5, color="red", transform=ax_density.transAxes)

        ax_density.set_xlabel("Angles (rad)")
        ax_density.set_ylabel("Density")
        ax_density.set_xlim(-np.pi, np.pi)
        ax_density.grid(True, alpha=0.2)

    # Rimuovi "LATTICE_" dal titolo
    title = exp_name.replace("LATTICE_", "")
    plt.suptitle(f"{dataset_name.upper()} - {title}", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])

    output_path = output_dir / f"{dataset_name}_{exp_name}_kde.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f" ✓ Grafico salvato: {output_path}")
    plt.close(fig)


# ------------------------------------------------------------------- #
# MAIN
# ------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualizza embedding con UMAP + KDE"
    )
    parser.add_argument("--dataset")
    parser.add_argument("--experiment")
    parser.add_argument("--dataset-path")
    parser.add_argument(
        "--output-dir",
        default="./reports/experiment_plots",
    )
    parser.add_argument(
        "--metric",
        default="euclidean",
        choices=["euclidean", "cosine", "manhattan"],
        help="Distance metric for UMAP (default: euclidean, recommended with S1 normalization)",
    )
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    selected = select_experiments(args.dataset, args.experiment)
    if not selected:
        print(" ✗ Nessun esperimento trovato.")
        return

    data_root = Path(args.dataset_path) if args.dataset_path else DATA_ROOT
    output_dir = Path(args.output_dir)

    cache = {}

    for exp in selected:
        print(f"\n==> {exp['dataset']} - {exp['name']}")

        dataset_path = data_root / exp["dataset"]
        embeddings = load_embeddings_for_experiment(dataset_path, exp["embeddings"], cache)
        reduced = reduce_embeddings_to_2d(
            embeddings,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            random_state=args.random_state,
            metric=args.metric,
        )

        plot_experiment_kde(reduced, exp["name"], exp["dataset"], output_dir)


if __name__ == "__main__":
    main()
