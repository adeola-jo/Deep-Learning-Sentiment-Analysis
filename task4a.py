import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import wandb
import os
import matplotlib.pyplot as plt
import csv
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import Counter
from collections import Counter
from utils import install_requirements, DummyContextManager, TextDataInspector, read_csv, plot_training_progress
from torch.nn.utils import rnn as rnn_utils
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from nlp import Vocab, NLPDataset, Instance
from networks import BaseLineModel, RNN
from utils import inspect_dataset, TextDataInspector, read_csv



    
# seed = 10000
seed = 7000000
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

    
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


def load_glove_embeddings(glove_file, vocab, embedding_dim=300, use_embeddings=True):
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
    # with open(glove_file, 'r') as file:

        for line in file:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector

    embedding_matrix = np.random.normal(0, 1, (len(vocab), embedding_dim))
    # embedding_matrix = np.random.normal(scale=0.6, size= (len(vocab.word2idx), embedding_dim))
    if use_embeddings:
        for i, word in enumerate(vocab.word2idx.keys()):
            if word == '<PAD>':
                embedding_matrix[i] = np.zeros(embedding_dim)
            elif word in embeddings_index.keys():
                embedding_matrix[i] = embeddings_index[word]
    #set requires_grad to False to prevent the embedding layer from being updated during training
    return torch.nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), padding_idx=0, freeze=True)

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
    return args.glove_embeddings.weight[text_indices]

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
    # accuracy = total_correct / batch_count * 100

    accuracy = accuracy_score(all_labels, all_predictions) * 100
    f1 = f1_score(all_labels, all_predictions)
    # conf_matrix = confusion_matrix(all_labels, all_preds)
    
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
    # conf_matrix = confusion_matrix(all_labels, all_preds)


    print('val total correct:', total_correct)
    print('val total examples:', batch_count)
    print('-----------------------------------')
    return avg_loss, accuracy, text_length_list

def train_and_validate(model, train_data, val_data, optimizer, criterion, args, wandb=None, use_desc=False):
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
    args.plot_counter += 1
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
    if use_desc:
        plot_training_progress(plot_data, show=args.show_plots, use_wandb=args.setup_wandb, wandb=wandb, save_dir=f'./plots/{args.plot_counter}.png', description = args.wandb_config)#dict(list(args.wandb_config.items())[:5]))
    else:
        plot_training_progress(plot_data, show=args.show_plots, use_wandb=args.setup_wandb, wandb=wandb, save_dir=f'./plots/{args.plot_counter}.png')
    return train_acc, train_loss, val_acc, val_loss


def find_best_rnn_cell_type(args, wandb, train_dataset, valid_dataset, test_dataset):
    all_runs_dir = []
    best_metrics = {'val accuracy': 0.0, 'val loss': float('inf'), 'rnn_cell_type': None}
    for rnn_cell_type in args.rnn_cell_types:

        print('=========================================')
        print('RUNNING EXPERIMENT FOR RNN CELL TYPE:', rnn_cell_type)
        print('=========================================')
        # Run experiment for each RNN cell type
        all_runs_dir, best_metrics = run_experiment(args, wandb, rnn_cell_type, args.hidden_sizes[0],
                                                args.num_layers_values[0], args.dropout_values[0],
                                                args.bidirectional_values[0], train_dataset, valid_dataset,
                                                test_dataset, all_runs_dir, best_metrics)
    best_rnn_cell_type = best_metrics['rnn_cell_type']
    return best_rnn_cell_type, best_metrics, all_runs_dir


def main(args):
    """
    Main function for running the deep learning model.

    Args:
        args (object): contain all the arguments for the model

    Returns:
        None
    """
    # Set random seed for reproducibility
    # torch.manual_seed(args.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(args.seed)
    
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
    
    # Initialize the vocabular
    train_text_frequencies = calculate_frequencies([row[0] for row in train_data])
    train_text_vocab = Vocab(train_text_frequencies, max_size=args.max_size, min_freq=args.min_freq)
    # # Load GloVe embeddings. We only need the embeddings for the training text vocab
    glove_embeddings = load_glove_embeddings(args.glove_file, train_text_vocab, embedding_dim=args.embedding_dim)
    args.glove_embeddings = glove_embeddings #add the glove embeddings to the args object

    # # # Initialize datasets
    train_dataset = NLPDataset(train_data, train_text_frequencies, glove_embeddings, args)  
    valid_dataset = NLPDataset(val_data,  train_text_frequencies, glove_embeddings, args) 
    test_dataset = NLPDataset(test_data,  train_text_frequencies, glove_embeddings, args)

    # Find the best RNN cell type using default hyperparameters
    if args.find_best_rnn_cell_type:
        all_runs_dir = []
        best_metrics = {'accuracy': 0.0, 'loss': float('inf')}
        best_rnn_cell_type, best_metrics, all_runs_dir = find_best_rnn_cell_type(args, wandb, train_dataset, valid_dataset, test_dataset)
        # Print the results
        print('=========================================')
        print('Best RNN Cell Type:', best_rnn_cell_type)
        print('Best Metrics:', best_metrics)
        print('All Runs Dir:', all_runs_dir)
        print('=========================================')

    # Reset the best metrics and all runs dir
    best_metrics = {'val accuracy': 0.0, 'val loss': float('inf')}
    all_runs_dir = []
    chosen_hyperparameters = {} 

    # ========================= USING RANDOM CHOICE ===========================
    # # # Randomized search
    # Assuming args.num_repeat is the number of samples you want
    random_hidden_sizes = random.sample(args.hidden_sizes, k=args.num_repeat)
    random_num_layers_values = random.sample(args.num_layers_values, k=args.num_repeat)
    random_dropout_values = random.sample(args.dropout_values, k=args.num_repeat)
    random_bidirectional_values = random.sample(args.bidirectional_values, k=args.num_repeat)


    for i in range(args.num_repeat):
        chosen_hyperparameters[f'{i}'] = {
            'hidden_size': random_hidden_sizes[i], 
            'num_layers': random_num_layers_values[i], 
            'dropout': random_dropout_values[i], 
            'bidirectional': random_bidirectional_values[i]
        }
        for rnn_cell_type in args.rnn_cell_types:
            all_runs_dir, best_metrics = run_experiment(args, wandb, rnn_cell_type, random_hidden_sizes[i], random_num_layers_values[i], random_dropout_values[i], random_bidirectional_values[i], train_dataset, valid_dataset, test_dataset, all_runs_dir, best_metrics)
            print('\n \n ===============================NEW RUN======================================')
        #save the chosen hyperparameters for each run to a text file
        with open(f'./plots/hyperparams_{i}.txt', 'w') as file:
            file.write(str(chosen_hyperparameters[f'{i}']))
    print('All runs dir:', all_runs_dir)
    print('Best metrics:', best_metrics)
    print('Chosen hyperparameters:', chosen_hyperparameters)



def initialize_model(args, rnn_cell_type, hidden_size, num_layers, dropout, bidirectional):
    model = RNN(rnn_cell_type, args.embedding_dim, hidden_size, args.num_classes, num_layers,
                bidirectional=bidirectional, dropout_rate=dropout, device=args.device)
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    return model, optimizer, criterion

def train_and_evaluate(args, model, train_dataset, valid_dataset, test_dataset, optimizer, criterion, wandb=None):
    train_acc, train_loss, val_acc, val_loss = train_and_validate(model, train_dataset, valid_dataset, optimizer, criterion, args, wandb=wandb, use_desc=True)

    test_loss, test_acc, _ = evaluate(model, test_dataset, criterion, args)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

    if wandb is not None:
        wandb.log({'Test/Loss': test_loss, 'Test/Accuracy': test_acc})

    return train_acc, train_loss, val_acc, val_loss

def initialize_run(args, wandb, rnn_cell_type, hidden_size, num_layers, dropout, bidirectional):
    config = {
        'seed': seed,
        'rnn_cell_type': rnn_cell_type,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'dropout': dropout,
        'bidirectional': bidirectional,
        'embedding_dim': args.embedding_dim,
        'num_classes': args.num_classes,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'clip': args.clip,
        'min_freq': args.min_freq,
        'max_size': args.max_size,
        'device': args.device,
        'setup_wandb': args.setup_wandb,
        'wandb_projectname': args.wandb_projectname,
        'log_interval': args.log_interval,
        'inspect_dataset': args.inspect_dataset,
        'install_requirements': args.install_requirements
    }
    args.wandb_config = config

    if args.setup_wandb:
        return wandb.init(project=args.wandb_projectname, entity=args.wandb_entity, config=config)
    else:
        return DummyContextManager()

def run_experiment(args, wandb, rnn_cell_type, hidden_size, num_layers, dropout, bidirectional, train_dataset, valid_dataset, test_dataset, all_runs_dir=None, best_metrics=None):
    with initialize_run(args, wandb, rnn_cell_type, hidden_size, num_layers, dropout, bidirectional) as run:
        if run:
            run_dir = run.dir
            #only append the last part of the run directory
            all_runs_dir.append(run_dir.split('/')[-1] if run_dir else run_dir)

        model, optimizer, criterion = initialize_model(args, rnn_cell_type, hidden_size, num_layers, dropout, bidirectional)
        with run:
            train_acc, train_loss, val_acc, val_loss = train_and_evaluate(args, model, train_dataset, valid_dataset, test_dataset, optimizer, criterion, wandb=run if args.setup_wandb else None)
        
        if val_acc > best_metrics['val accuracy']:
            best_metrics['train accuracy'] = train_acc
            best_metrics['train loss'] = train_loss
            best_metrics['val accuracy'] = val_acc
            best_metrics['val loss'] = val_loss
            best_metrics['rnn_cell_type'] = rnn_cell_type
            best_metrics['hidden_size'] = hidden_size
            best_metrics['num_layers'] = num_layers
            best_metrics['dropout'] = dropout
            best_metrics['bidirectional'] = bidirectional
    return all_runs_dir, best_metrics


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
        rnn_cell_types : list of str
            The types of RNN cells to use.
        hidden_sizes : list of int
            The sizes of the hidden layers.
        num_layers_values : list of int
            The number of layers in the RNN.
        dropout_values : list of float
            The dropout rates to use.
        bidirectional_values : list of bool
            Whether to use bidirectional RNNs.
        embedding_dim : int
            The dimension of the embeddings.
        num_classes : int
            The number of classes in the output.
        num_repeat : int
            The number of times to repeat the randomized search without replacement for hyperparameters.
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
        find_best_rnn_cell_type : bool
            Whether to find the best RNN cell type.
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
        rnn_cell_types = ['RNN', 'GRU', 'LSTM']
        hidden_sizes = [150, 200, 250]
        num_layers_values = [2, 4, 6]
        dropout_values = [0.5, 0.3, 0.7]
        bidirectional_values = [False, True, False]
        embedding_dim = 300
        num_classes = 1
        num_repeat = 3 # Number of times to repeat the randomized search without replacement for hyperparameters
        # num_layers = 2
        epochs = 10
        batch_size = 10
        lr = 1e-4
        clip = 0.25
        min_freq = 1
        max_size = -1
        find_best_rnn_cell_type = True
        # device = torch.device("cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        setup_wandb = False
        wandb_projectname = 'lab3_task4'
        wandb_entity = 'adeolajosepholoruntoba'
        wandb_api_key = None
        log_interval = 1
        inspect_dataset = False
        install_requirements = False
        plot_counter = 0
        show_plots = False

    args = Args()
    if args.install_requirements:
        # Install required packages
        install_requirements(os.path.abspath(__file__))
    main(args)



# RESULTS:
    # SEED = 7052020
    # Best metrics: {'accuracy': 81.32894014277869, 'loss': 0.04424905191487103, 'rnn_cell_type': 'GRU', 'hidden_size': 150, 'num_layers': 2, 'dropout': 0.7, 'bidirectional': False}


    # TRY THIS FOR 4 MORE SEEDS












