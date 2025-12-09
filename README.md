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

## Contributing

Please follow the existing code style and add tests for new features.

## License

See LICENSE file.
