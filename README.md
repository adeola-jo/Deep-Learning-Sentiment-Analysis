
# Deep Learning for Sentiment Analysis

This project implements a comprehensive sentiment analysis system using various deep learning architectures. The system analyzes movie reviews from the Stanford Sentiment Treebank (SST) dataset and classifies them as positive or negative.

## Project Overview

The project explores multiple neural network architectures including:
- Baseline fully-connected models
- Recurrent Neural Networks (RNNs)
- Gated Recurrent Units (GRUs)
- Long Short-Term Memory networks (LSTMs)

Each model is evaluated for performance in sentiment classification, with extensive hyperparameter tuning and comparative analysis between architectures.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Models](#models)
- [Training & Evaluation](#training--evaluation)
- [Experiments](#experiments)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

```
.
├── data/                           # Dataset files
│   ├── sst_glove_6b_300d.txt       # Pre-trained GloVe word embeddings
│   ├── sst_train_raw.csv           # Training dataset
│   ├── sst_valid_raw.csv           # Validation dataset
│   ├── sst_test_raw.csv            # Test dataset
├── plots/                          # Generated plots and results
│   ├── *.png                       # Performance visualizations
│   ├── hyperparams_*.txt           # Hyperparameter configurations
├── networks.py                     # Neural network architecture implementations
├── nlp.py                          # NLP utilities (vocabulary, tokenization, embeddings)
├── task2.py                        # Baseline model implementation
├── task3.py                        # RNN model implementation
├── task4a.py                       # Hyperparameter tuning for RNN models
├── task4b_a.py                     # Comparative analysis with/without pre-trained embeddings
├── task4b_b.py                     # Advanced hyperparameter optimization
├── task4c.py                       # RNN variant comparison (RNN vs GRU vs LSTM)
├── test.ipynb                      # Jupyter notebook for testing and visualization
└── utils.py                        # Utility functions for data processing and visualization
```

## Installation

1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install the required packages:
   ```sh
   pip install torch numpy pandas matplotlib seaborn wordcloud nltk tqdm scikit-learn tabulate wandb
   ```

3. Download NLTK resources (if needed):
   ```python
   import nltk
   nltk.download('punkt')
   ```

## Dataset

The Stanford Sentiment Treebank (SST) is used in this project. It contains movie reviews labeled as positive or negative. The dataset is split into:

- Training set: Used to train the models
- Validation set: Used for hyperparameter tuning
- Test set: Used for final evaluation

The data is provided in CSV format with text and sentiment labels.

## Models

### Baseline Model (`task2.py`)
- A feed-forward neural network with mean pooling
- Takes word embeddings as input
- Uses fully connected layers with dropout regularization

### RNN Implementation (`task3.py`)
- Recurrent Neural Network for sequence processing
- Supports different recurrent cell types (RNN, GRU, LSTM)
- Configurable for bidirectional processing
- Multiple layer support with dropout

### Model Components (`networks.py`)
- `BaseLineModel`: The basic feed-forward neural network
- `RNN`: Recurrent neural network with configurable cell types
- `MeanPoolingLayer`: Layer for averaging word embeddings
- Support for various activation functions and attention mechanisms

## Training & Evaluation

### Training Process
- Models are trained using the Adam optimizer
- Binary Cross-Entropy loss function
- Gradient clipping to prevent exploding gradients
- Learning rate scheduling options
- Training progress visualization

### Evaluation Metrics
- Accuracy: Percentage of correct predictions
- F1 Score: Harmonic mean of precision and recall
- Confusion Matrix: Visualization of true vs. predicted labels
- Loss curves: Training and validation loss over epochs

## Experiments

The project includes several experiments:

### Experiment 1: Baseline Model Evaluation (`task2.py`)
- Performance of feed-forward networks on sentiment analysis
- Effect of word embeddings and dropout regularization

### Experiment 2: RNN Variants (`task3.py`)
- Implementation of standard RNN cells
- Comparison with baseline model performance

### Experiment 3: Hyperparameter Optimization (`task4a.py`)
- Grid search and random search for optimal hyperparameters
- Exploration of hidden layer size, number of layers, dropout rate
- Analysis of bidirectional vs. unidirectional models

### Experiment 4: Embedding Strategy Comparison (`task4b_a.py`)
- Pre-trained GloVe embeddings vs. randomly initialized embeddings
- Impact of freezing vs. fine-tuning embeddings during training

### Experiment 5: Advanced Optimization (`task4b_b.py`)
- Systematic comparison of optimizer types
- Gradient clipping values
- Vocabulary size effects
- Activation function performance

### Experiment 6: RNN Cell Type Comparison (`task4c.py`)
- Standard RNN vs. GRU vs. LSTM
- Performance metrics across cell types
- Analysis of sequence length handling

## Results

Experiment results show:

1. GRU generally outperforms standard RNN and sometimes LSTM for this task
2. Optimal hyperparameters include:
   - Hidden size: 150-200
   - Dropout rate: 0.5-0.7
   - Layers: 2-4
3. Pre-trained embeddings significantly improve performance
4. Bidirectional models show mixed results depending on configuration

Detailed results with performance metrics are saved in the `plots/` directory and include:
- Learning curves (training/validation loss and accuracy)
- Hyperparameter configurations
- Model comparison tables

## Usage

### Run Baseline Model
```sh
python task2.py
```

### Run RNN Model
```sh
python task3.py
```

### Run Hyperparameter Tuning
```sh
python task4a.py
```

### Run Embedding Comparison
```sh
python task4b_a.py
```

### Run Advanced Optimization
```sh
python task4b_b.py
```

### Run RNN Cell Type Comparison
```sh
python task4c.py
```

### Configuration
Each script has a configurable `Args` class that can be modified to change:
- Model hyperparameters
- Training settings
- Data paths
- Device selection (CPU/GPU)
- Logging options (WandB integration)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
