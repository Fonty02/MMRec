# MMRec

MMRec is a multimodal recommendation framework that supports various recommendation models and datasets, incorporating vision, text, and audio features.

## Features

- Support for multiple recommendation models (e.g., BPR, FREEDOM, LightGCN, etc.)
- Multimodal features: vision (images), text, audio
- Datasets: MovieLens 1M, LastFM
- Hyperparameter tuning and experiment logging
- Results saved to CSV for analysis

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd MMRec
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Datasets

The framework uses preprocessed datasets. Ensure the data files are in the `data/` directory.

- **MovieLens 1M**: Located in `data/movielens_1m/`
- **LastFM**: Located in `data/lastfm/`

Feature files (e.g., `image_audioclip.npy`, `text_minilm.npy`) should be present in the dataset folders.

## Usage

### Running a Single Experiment

Use `src/main.py` to run a single model on a dataset:

```bash
python src/main.py --model BPR --dataset movielens_1m
```

Available models: BPR, FREEDOM, LightGCN, etc.  
Available datasets: movielens_1m, lastfm

### Running Multiple Experiments

Use `src/run_experiments.py` to run predefined experiments:

```bash
python src/run_experiments.py
```

This script runs experiments with different feature combinations and saves results.

### Configuration

Models and datasets are configured in YAML files in `src/configs/`.

- `configs/model/`: Model-specific configurations
- `configs/dataset/`: Dataset-specific configurations

## Models

- **BPR**: Bayesian Personalized Ranking
- **FREEDOM**: Multimodal recommendation model
- **LightGCN**: Graph Convolutional Network for recommendations
- And many more in `src/models/`

## Evaluation

Results are logged and saved to CSV files in the `reports/` directory.

Metrics include precision, recall, NDCG, etc.

## Preprocessing

Preprocessing scripts are in `preprocessing/`. For example:

- `0rating2inter.ipynb`: Convert ratings to interactions
- `3feat-encoder.ipynb`: Encode features

### Mapper scripts (mapping and feature reconstruction) 🔧

There are two repository-local mapper scripts that prepare multimodal datasets by filtering, remapping IDs and rebuilding embedding files:

- `data/movielens_1m/features_mmrec/mapper.py`
   - Purpose: filter `movielens_1m.inter` to keep only items with multimodal features, apply 5-core filtering, remap users and items to 0..n-1, rebuild `.npy` embedding files and save mapping CSVs.
   - Required inputs: `movielens_1m.inter`, `item_features.csv`, and embeddings (e.g. `image_audioclip.npy`, `audio_audioclip.npy`, `text_audioclip.npy`, `image_clip.npy`, `text_clip.npy`, `text_minilm.npy`, `image_vit.npy`, `audio_vggish.npy`).
   - Outputs (saved in the parent directory): `movielens_1m.inter`, `item_features.csv`, `user_mapping.csv`, `item_mapping.csv`, and remapped `.npy` embeddings.
   - Run example:
      ```bash
      cd data/movielens_1m/features_mmrec
      python mapper.py
      ```

- `data/lastfm/lastfm_features/mapper.py`
   - Purpose: map remapped `itemID` → original `artistID`, filter artists with available features, expand user-artist interactions into user-item interactions (one per variant), apply 5-core (on artists), rebuild embeddings preserving all variants and save mappings.
   - Required inputs: `lastfm.inter`, `user_artists.dat`, `item_features.csv`, and embeddings (e.g. `image_audioclip.npy`, `audio_audioclip.npy`, `text_audioclip.npy`, `image_clip.npy`, `text_clip.npy`, `text_minilm.npy`, `image_vit.npy`).
   - Outputs (saved in the parent directory): `lastfm.inter`, `item_features.csv`, `user_mapping.csv`, `item_mapping.csv`, and remapped `.npy` embeddings.
   - Run example:
      ```bash
      cd data/lastfm/lastfm_features
      python mapper.py
      ```

Note: mapper scripts print progress and perform basic checks (e.g., consistent number of rows across embedding files); they raise errors if required files are missing.

### Analysis script (visualize embeddings)

- `data/analyze.py`
   - Purpose: visualize embeddings using UMAP reduction and Gaussian KDE; produces scatter + density plots for selected experiments.
   - Default experiments are configured inside the script (see `DEFAULT_EXPERIMENTS`) for both MovieLens and LastFM.
   - Required inputs: `.npy` embedding files in each dataset folder (e.g. `text_minilm.npy`, `image_vit.npy`, `audio_vggish.npy`, `text_clip.npy`, etc.).
   - Outputs: PNG images saved to `--output-dir` (default `./reports/experiment_plots`).
   - Dependencies: `umap-learn`, `matplotlib`, `scipy`, `numpy`.
   - Run example:
      ```bash
      python data/analyze.py --dataset movielens_1m --experiment LATTICE_audioclip_full
      ```

The analysis script supports command-line options to select datasets/experiments and UMAP parameters (`--n-neighbors`, `--min-dist`, `--metric`, `--random-state`).

## Contributing

Please follow the existing code style and add tests for new features.

## License

See LICENSE file.
