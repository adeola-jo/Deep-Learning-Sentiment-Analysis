# Deep Learning Project

This project is a deep learning-based sentiment analysis system that uses various neural network architectures to classify movie reviews as positive or negative. The project includes data preprocessing, model training, evaluation, and visualization of results.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Models](#models)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Project Structure
```
.
├── data/
│   ├── sst_glove_6b_300d.txt
│   ├── sst_test_raw.csv
│   ├── sst_train_raw.csv
│   ├── sst_valid_raw.csv
├── plots/
│   ├── 1.png
│   ├── ...
│   ├── hyperparams_0.txt
│   ├── hyperparams_1.txt
│   ├── hyperparams_2.txt
├── networks.py
├── nlp.py
├── task2.py
├── task3.py
├── task4a.py
├── task4b_a.py
├── task4b_b.py
├── task4c.py
├── test.ipynb
├── utils.py
└── __pycache__/
```

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/your-repo.git
    cd your-repo
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
### Data Preparation
Ensure that the data files (`sst_train_raw.csv`, `sst_valid_raw.csv`, `sst_test_raw.csv`, `sst_glove_6b_300d.txt`) are placed in the `data/` directory.

### Running the Scripts
You can run the different tasks using the provided scripts. For example:
```sh
python task2.py
python task3.py
python task4a.py
python task4b_a.py
python task4b_b.py
python task4c.py
```

## Data
The dataset used in this project is the Stanford Sentiment Treebank (SST). It contains movie reviews labeled as positive or negative.

## Models
The project includes various neural network architectures:
- Baseline Model
- RNN
- GRU
- LSTM

## Training
Training scripts are provided for different tasks. Each script initializes the model, loads the data, and trains the model. For example, to train the model using `task2.py`:
```sh
python task2.py
```

## Evaluation
The evaluation metrics include accuracy, F1 score, and confusion matrix. The evaluation results are printed to the console and logged using Weights & Biases (if enabled).

## Visualization
The project includes visualization tools to plot training progress, token frequency distribution, sentence length distribution, and word clouds. These visualizations help in understanding the data and the model's performance.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
