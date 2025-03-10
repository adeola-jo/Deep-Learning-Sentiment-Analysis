import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import wandb
import os
import matplotlib.pyplot as plt
import seaborn as sns
import string
import csv
import io
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import Counter
from nltk.tokenize import word_tokenize
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from utils import install_requirements, inspect_dataset, TextDataInspector, read_csv, plot_training_progress
from torch.nn.utils import rnn as rnn_utils
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tabulate import tabulate
from nlp import Vocab, NLPDataset, Instance
from networks import BaseLineModel, RNN


torch.manual_seed(7052020)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(7052020)
    
    
def calculate_frequencies(text_column):
    """
    Calculate the frequencies of words in a given text column.

    Args:
        text_column (pandas.Series): A pandas Series containing text data.

    Returns:
        dict: A dictionary containing the word frequencies.
    """
    frequencies = Counter()
    for text in text_column:
        #split the text into tokens
        tokens = text.split()
        # Update counts
        frequencies.update(tokens)
    return dict(frequencies)


def load_glove_embeddings(glove_file, vocab, embedding_dim=300, use_pretrained_embeddings=True, padding_idx=0, freeze=True):
    """
    Load GloVe word embeddings from a file and create an embedding matrix for the given vocabulary.

    Args:
        glove_file (str): Path to the GloVe file.
        vocab (Vocabulary): Vocabulary object containing word-to-index mapping.
        embedding_dim (int, optional): Dimensionality of the word embeddings. Defaults to 300.
        use_embeddings (bool, optional): Whether to use the loaded embeddings or Normal(0, 1) initialization. Defaults to True.

    Returns:
        torch.nn.Embedding: Embedding layer initialized with the GloVe word embeddings.
    """
    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector

    embedding_matrix = np.random.normal(0, 1, (len(vocab), embedding_dim))
    # embedding_matrix = np.random.normal(scale=0.6, size= (len(vocab.word2idx), embedding_dim))
    if use_pretrained_embeddings:
        for i, word in enumerate(vocab.word2idx.keys()):
            if word == '<PAD>':
                embedding_matrix[i] = np.zeros(embedding_dim)
            elif word in embeddings_index.keys():
                embedding_matrix[i] = embeddings_index[word]
        #set requires_grad to False to prevent the embedding layer from being updated during training
        return torch.nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), padding_idx=padding_idx, freeze=freeze)
    else:
            embedding_matrix[0] = np.zeros(embedding_dim)
            return torch.tensor(embedding_matrix, dtype=torch.float32)

def pad_collate_fn(batch, pad_idx=0):
    """
    Collate function for padding and batching text data.

    Args:
        batch (list): List of tuples containing texts and labels.
        pad_idx (int, optional): Index used for padding. Defaults to 0.

    Returns:
        tuple: Tuple containing the padded text embeddings, labels, and lengths.
    """
    texts, labels = zip(*batch)
    #get the length of each text in the batch
    lengths = [len(text) for text in texts]
    #convert the lengths to a tensor of size (batch_size)
    lengths = torch.tensor(lengths).view(-1)
    texts = [text.clone().detach() for text in texts]  # if texts are already tensors
    texts = torch.nn.utils.rnn.pad_sequence(texts, padding_value=pad_idx, batch_first=True)
    text_embeddings = get_embedding_matrix(texts) #get the embedding matrix for each text in the batch
    labels = torch.tensor(labels, dtype=torch.long)
    return text_embeddings, labels, lengths


def get_embedding_matrix(text_indices):
    """
    Retrieves the embedding matrix for the given text indices.

    Args:
        text_indices (list): A list of text indices.

    Returns:
        torch.Tensor: The embedding matrix for the given text indices.
    """
    global args #get the args object from the global namespace
    if args.use_pretrained_embeddings:
        return args.glove_embeddings(text_indices)
    else:
        #extract the text indices for the current batch from the embedding matrix tensor
        return args.glove_embeddings[text_indices]

def setup_wandb(args):
    """
    Set up WandB for logging experiment metrics and visualizations.

    Args:
        args: An object containing all the arguments for the entire program.

    Raises:
        ValueError: If the user is not logged in to WandB.

    Returns:
        None
    """
    try:
        # Attempt to access WandB user information
        if not wandb.api.api_key:
            raise ValueError("User not logged in")
    except (AttributeError, ValueError):
        # If user is not logged in, then login
        try:
            wandb.login(key=str(os.environ['WANDB_API_KEY']), relogin=False)
        except KeyError:
            wandb.login(key=str(args.wandb_api_key), relogin=False)
    # Initialize the wandb project
    wandb.init(project=args.wandb_projectname, entity=args.wandb_entity, config=vars(args)) 
    return wandb


def train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, args, pbar=None, wandb=None):
    """
    Trains the model for one epoch using the provided data loader.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): The data loader for training data.
        optimizer (Optimizer): The optimizer used for training.
        loss_fn (Loss): The loss function used for training.
        device (torch.device): The device to be used for training.
        epoch (int): The current epoch number.
        args (object): The arguments object containing all the hyperparameters.
        pbar (tqdm.tqdm, optional): The progress bar for tracking training progress. Defaults to None.
        wandb (wandb.Run, optional): The wandb object for logging training metrics. Defaults to None.

    Returns:
        float: The average loss per batch.
        float: The accuracy of the model on the training data.
        list: A list of text lengths for each batch.
    """
    model.train()
    # sigmoid = nn.Sigmoid()

    total_correct, total_loss, batch_count = 0, 0, 0
    text_length_list = []
    all_predictions = []
    all_labels = []
    for batch_num, (batch_text, batch_label, batch_text_length) in enumerate(train_loader, 1):
        batch_text, batch_label = batch_text.to(device), batch_label.to(device)
        optimizer.zero_grad()
        logits = model(batch_text, batch_text_length)
        loss = loss_fn(logits.squeeze(1), batch_label.float())# Adjusted for BCEWithLogitsLoss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.item()
        total_correct += ((torch.sigmoid(logits.squeeze(1)) >= 0.5) == batch_label).sum().item()

        predictions = torch.sigmoid(logits.squeeze(1)).round()

        all_predictions.extend(predictions.tolist())
        all_labels.extend(batch_label.tolist())
                          
        batch_count += len(batch_text) #batch_text.size(0)
        text_length_list.append(batch_text_length)
        if pbar is not None:
            pbar.set_postfix({'Loss': total_loss / batch_count, 'Accuracy': total_correct / batch_count * 100})
            pbar.update(1)

    avg_loss = total_loss / batch_count
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    f1 = f1_score(all_labels, all_predictions)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    print('-----------------------------------')
    print('train epoch:', epoch)
    print('train total correct:', total_correct)
    print('train total examples:', batch_count)
    return avg_loss, accuracy, text_length_list

def evaluate(model, data, loss_fn, args):
    """
    Evaluate the performance of a model on a given dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data (torch.utils.data.Dataset): The dataset to evaluate on.
        loss_fn (torch.nn.Module): The loss function used for evaluation.
        args (object): The arguments object containing all the hyperparameters.

    Returns:
        tuple: A tuple containing the average loss, accuracy, and a list of text lengths.
    """
    model.eval()
    data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_fn)
    text_length_list = []
    total_loss, total_correct, batch_count = 0, 0, 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_num, (batch_text, batch_label, batch_text_length) in enumerate(data_loader, 1):
            batch_text, batch_label = batch_text.to(args.device), batch_label.to(args.device)
            logits = model(batch_text, batch_text_length)
            loss = loss_fn(logits.squeeze(1), batch_label.float())
            total_loss += loss.item()
            total_correct += ((torch.sigmoid(logits.squeeze(1)) >= 0.5) == batch_label).sum().item()
            batch_count += len(batch_text) #batch_text.size(0)
            text_length_list.append(batch_text_length)

            predictions = torch.sigmoid(logits.squeeze(1)).round()
            all_predictions.extend(predictions.tolist())
            all_labels.extend(batch_label.tolist())
    
    avg_loss = total_loss / batch_count
    # accuracy = total_correct / batch_count * 100

    accuracy = accuracy_score(all_labels, all_predictions) * 100
    f1 = f1_score(all_labels, all_predictions)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    print('val total correct:', total_correct)
    print('val total examples:', batch_count)
    print('-----------------------------------')
    return avg_loss, accuracy, text_length_list


def train(model, train_data, val_data, optimizer, criterion, args, wandb=None):
    """
    Trains the model on the training data and evaluates it on the validation data for each epoch.

    Args:
        model (nn.Module): The model to be trained.
        train_data (Dataset): The training dataset.
        val_data (Dataset): The validation dataset.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss function.
        args (object): The arguments object containing all the hyperparameters.
        wandb (wandb.Run, optional): Wandb run object for logging. Defaults to None.
    """
    plot_data = {}
    plot_data['train_loss'] = []
    plot_data['valid_loss'] = []
    plot_data['train_acc'] = []
    plot_data['valid_acc'] = []
    plot_data['lr'] = []

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_fn)
    for epoch in range(args.epochs):
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{args.epochs}', unit='batch') as pbar:
            train_loss, train_acc, _ = train_epoch(model, train_loader, optimizer, criterion, args.device, epoch + 1, args, pbar=pbar, wandb=wandb)
            val_loss, val_acc, _ = evaluate(model, val_data, criterion, args)

            plot_data['train_loss'] += [train_loss]
            plot_data['valid_loss'] += [val_loss]
            plot_data['train_acc'] += [train_acc]
            plot_data['valid_acc'] += [val_acc]
            plot_data['lr'] += [optimizer.param_groups[0]['lr']]

            print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')
            # Log to wandb
            if wandb is not None:
                wandb.log({'Epoch': epoch + 1, 'Train/Loss': train_loss, 'Train/Accuracy': train_acc, 'Val/Loss': val_loss, 'Val/Accuracy': val_acc})
    plot_training_progress(plot_data, show=True, use_wandb=args.setup_wandb, wandb=wandb)
    return train_loss, train_acc, val_loss, val_acc


def main(args):
    """
    Main function for running the deep learning model.

    Args:
        args (object): contain all the arguments for the model

    Returns:
        None
    """
    # Set random seed for reproducibility
    # np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    if args.setup_wandb:
        # Setup wandb
        wandb = setup_wandb(args)
    else:
        wandb = None

    # Load and preprocess the dataset
    try:
        #get the absolute path of the file
        train_data = read_csv(os.path.abspath(args.train_filepath))
        val_data = read_csv(os.path.abspath(args.val_filepath))
        test_data = read_csv(os.path.abspath(args.test_filepath))
    except FileNotFoundError:
        print('File not found')
        return
    
    # Inspect the dataset
    if args.inspect_dataset:
        print('Inspecting dataset')
        train_inspector = TextDataInspector(train_data)
        train_inspector.run()
        return
    
    # Initialize the vocabulary
    train_text_frequencies = calculate_frequencies([row[0] for row in train_data])
    train_text_vocab = Vocab(train_text_frequencies, max_size=args.max_size, min_freq=args.min_freq)

        # Load GloVe embeddings. We only need the embeddings for the training text vocab
    glove_embeddings = load_glove_embeddings(args.glove_file, train_text_vocab, embedding_dim=args.embedding_dim, use_pretrained_embeddings=args.use_pretrained_embeddings, freeze=args.freeze_embeddings)

    args.glove_embeddings = glove_embeddings  # Add the glove embeddings to the args object

    # Initialize datasets
    train_dataset = NLPDataset(train_data, train_text_frequencies, glove_embeddings, args)  
    valid_dataset = NLPDataset(val_data,  train_text_frequencies, glove_embeddings, args)  
    test_dataset = NLPDataset(test_data,  train_text_frequencies, glove_embeddings, args)

    # Train the baseline model
    baseline_model = BaseLineModel(embedding_dim=args.embedding_dim, fc1_width=args.fc1_width, fc2_width=args.fc2_width, output_dim=args.num_classes, dropout_rate=args.dropout_rate, device=args.device)
    baseline_model.to(args.device)  # Send the model to the device
    baseline_criterion = nn.BCEWithLogitsLoss()
    baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=args.lr)

    baseline_train_loss, baseline_train_acc, baseline_val_loss, baseline_val_acc = train(baseline_model, train_dataset, valid_dataset, baseline_optimizer, baseline_criterion, args, wandb=wandb)
    # Evaluate the baseline model on the test data
    baseline_test_loss, baseline_test_acc, _ = evaluate(baseline_model, test_dataset, baseline_criterion, args)

    # Train the RNN model
    rnn_model = RNN(args.rnn_cell_type, embedding_dim=args.embedding_dim, hidden_size=args.hidden_size, num_classes=args.num_classes, num_layers=args.num_layers, bidirectional=args.bidirectional, dropout_rate=args.dropout_rate, device=args.device)

    rnn_model.to(args.device)  # Send the model to the device
    rnn_criterion = nn.BCEWithLogitsLoss()
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=args.lr)

    rnn_train_loss, rnn_train_acc, rnn_val_loss, rnn_val_acc = train(rnn_model, train_dataset, valid_dataset, rnn_optimizer, rnn_criterion, args, wandb=wandb)
    # Evaluate the RNN model on the test data
    rnn_test_loss, rnn_test_acc, _ = evaluate(rnn_model, test_dataset, rnn_criterion, args)

    # tabulate the final results of the baseline and RNN models
    headers = ["Model", "Train Loss", "Train Accuracy", "Validation Loss", "Validation Accuracy", "Test Loss", "Test Accuracy"]
    table_data = [
        ["Baseline", baseline_train_loss, baseline_train_acc, baseline_val_loss, baseline_val_acc, baseline_test_loss, baseline_test_acc],
        [f"{args.rnn_cell_type}", rnn_train_loss, rnn_train_acc, rnn_val_loss, rnn_val_acc, rnn_test_loss, rnn_test_acc],
    ]

    print(tabulate(table_data, headers=headers))



if __name__ == '__main__':
    class Args:
        """
        A class used to store the arguments for the training script.

        Attributes
        ----------
        seed : int
            The seed for the random number generator.
        train_filepath : str
            The path to the training data file.
        val_filepath : str
            The path to the validation data file.
        test_filepath : str
            The path to the test data file.
        glove_file : str
            The path to the GloVe embeddings file.
        rnn_cell_type : str
            The type of RNN cell to use.
        embedding_dim : int
            The dimension of the embeddings.
        hidden_size : int
            The size of the hidden layer.
        num_classes : int
            The number of classes in the output.
        num_layers : int
            The number of layers in the RNN.
        bidirectional : bool
            Whether to use a bidirectional RNN.
        epochs : int
            The number of epochs to train for.
        batch_size : int
            The size of the batches for training.
        lr : float
            The learning rate for the optimizer.
        clip : float
            The value to clip gradients at.
        min_freq : int
            The minimum frequency for words to be included in the vocabulary.
        max_size : int
            The maximum size of the vocabulary.
        dropout_rate : float
            The dropout rate to use.
        fc1_width : int
            The width of the first fully connected layer.
        fc2_width : int
            The width of the second fully connected layer.
        use_pretrained_embeddings : bool
            Whether to use pretrained embeddings.
        freeze_embeddings : bool
            Whether to freeze the embeddings during training.
        device : torch.device
            The device to run the training on.
        setup_wandb : bool
            Whether to set up Weights & Biases for logging.
        wandb_projectname : str
            The name of the Weights & Biases project.
        wandb_entity : str
            The entity (user or team) in Weights & Biases.
        wandb_api_key : str
            The API key for Weights & Biases.
        log_interval : int
            The interval (in batches) at which to log training information.
        inspect_dataset : bool
            Whether to inspect the dataset before training.
        install_requirements : bool
            Whether to install the required packages.
        """
        seed = 7052020
        # seed = 8200
        train_filepath = './data/sst_train_raw.csv'
        val_filepath = './data/sst_valid_raw.csv'
        test_filepath = './data/sst_test_raw.csv'
        glove_file = './data/sst_glove_6b_300d.txt'
        rnn_cell_type = 'GRU'
        embedding_dim = 300
        hidden_size = 150
        num_classes = 1
        num_layers = 2
        bidirectional = False
        epochs = 10
        batch_size = 10
        lr = 1e-4
        clip = 0.25
        min_freq = 1
        max_size = -1
        dropout_rate = 0.7
        #parameters for the FCN:
        fc1_width = 150
        fc2_width = 150
        use_pretrained_embeddings = True
        freeze_embeddings = True
        # device = torch.device("cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        setup_wandb = False
        wandb_projectname = 'lab3_task3'
        wandb_entity = 'adeolajosepholoruntoba'
        wandb_api_key = None
        log_interval = 1
        inspect_dataset = False
        install_requirements = False

    args = Args()
    if args.install_requirements:
        # Install required packages
        install_requirements(os.path.abspath(__file__))
    main(args)











