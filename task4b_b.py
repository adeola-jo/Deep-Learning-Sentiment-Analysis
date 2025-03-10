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
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import Counter
from nltk.tokenize import word_tokenize
from collections import Counter
from utils import install_requirements, inspect_dataset, TextDataInspector, read_csv, DummyContextManager, plot_training_progress, setup_wandb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from nlp import Vocab, NLPDataset, calculate_frequencies, load_glove_embeddings
from networks import BaseLineModel, RNN
from tabulate import tabulate


torch.manual_seed(7052020)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(7052020)


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

def train_and_validate(model, train_data, val_data, optimizer, criterion, args, wandb=None):
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


def get_activation_function(name):
    activation_functions = {
        'ReLU': nn.ReLU(),
        'LeakyReLU': nn.LeakyReLU(),
        'ELU': nn.ELU(),
        'SELU': nn.SELU(),
        'Tanh': nn.Tanh(),
        'Sigmoid': nn.Sigmoid(),
        'Softplus': nn.Softplus(),
        'Softsign': nn.Softsign(),
        'Softmax': nn.Softmax(),
        'LogSoftmax': nn.LogSoftmax(),
        'ReLU6': nn.ReLU6()
    }
    return activation_functions[name]

def get_optimizer(name, parameters, lr):
    optimizers = {
        'Adam': optim.Adam(parameters, lr=lr),
        'SGD': optim.SGD(parameters, lr=lr),
        'RMSprop': optim.RMSprop(parameters, lr=lr),
        'Adagrad': optim.Adagrad(parameters, lr=lr),
        'Adadelta': optim.Adadelta(parameters, lr=lr),
        'AdamW': optim.AdamW(parameters, lr=lr),
        'SparseAdam': optim.SparseAdam(parameters, lr=lr)
    }
    return optimizers.get(name)

def initialize_run(args, wandb, vocab_size, lr, dropout_rate, optimizer, grad_clip_value, freeze_vector_rep, activation_function):
    config = {
        'seed': args.seed,
        'rnn_cell_type': args.rnn_cell_type,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': dropout_rate,  # updated
        'bidirectional': args.bidirectional,
        'embedding_dim': args.embedding_dim,
        'num_classes': args.num_classes,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': lr,  # updated
        'clip': grad_clip_value,  # updated
        'min_freq': args.min_freq,
        'max_size': vocab_size,  # updated
        'device': args.device,
        'setup_wandb': args.setup_wandb,
        'wandb_projectname': args.wandb_projectname,
        'log_interval': args.log_interval,
        'inspect_dataset': args.inspect_dataset,
        'install_requirements': args.install_requirements,
        'optimizer': optimizer,  # new
        'freeze_vector_rep': freeze_vector_rep,  # new
        'activation_function': activation_function  # new
    }

    if args.setup_wandb:
        return wandb.init(project=args.wandb_projectname, entity=args.wandb_entity, config=config)
    else:
        return DummyContextManager()
    
def create_model_and_optimizer(ModelClass, model_params, optimizer_name, lr):
    model = ModelClass(**model_params)
       # Check if the model has learnable parameters
    if not any(param.requires_grad for param in model.parameters()):
        raise ValueError("Model has no learnable parameters.")
        
    model.to(model_params['device'])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = get_optimizer(optimizer_name, list(model.parameters()), lr)
    return model, criterion, optimizer


def train_and_evaluate_model(model, train_dataset, valid_dataset, test_dataset, optimizer, criterion, args, wandb):
    train_loss, train_acc, val_loss, val_acc = train_and_validate(model, train_dataset, valid_dataset, optimizer, criterion, args, wandb=wandb)
    test_loss, test_acc, _ = evaluate(model, test_dataset, criterion, args)

    return {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc
    }

def create_model_params(args, random_dropout, random_activation_function):
    model_params = {
        'embedding_dim': args.embedding_dim,
        'dropout_rate': random_dropout,
        'device': args.device,
        'activation_function': get_activation_function(random_activation_function)
    }
    return model_params


def print_results(chosen_hyperparameters):
    # Define the keys for the hyperparameters and metrics
    hyperparameter_keys = ['vocab_size', 'lr', 'dropout', 'optimizer', 'grad_clip_value', 'freeze_vector_rep', 'activation_function']
    metric_keys = ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc']

    # Prepare the headers
    headers = ["Run"] + hyperparameter_keys + ["Baseline " + key for key in metric_keys] + ["RNN " + key for key in metric_keys]
    table_data = []

    for i, params in enumerate(chosen_hyperparameters):
        hyperparameters = [params[key] for key in hyperparameter_keys]
        baseline_metrics = [params['baseline_metrics'][key] for key in metric_keys]
        rnn_metrics = [params['rnn_metrics'][key] for key in metric_keys]
        table_data.append([i + 1] + hyperparameters + baseline_metrics + rnn_metrics)

    print(tabulate(table_data, headers=headers))


def setup(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    if args.setup_wandb:
        wandb = setup_wandb(args)
    else:
        wandb = None

    return wandb

def load_datasets(args):
    try:
        train_data = read_csv(os.path.abspath(args.train_filepath))
        val_data = read_csv(os.path.abspath(args.val_filepath))
        test_data = read_csv(os.path.abspath(args.test_filepath))
    except FileNotFoundError:
        print('File not found')
        return None, None, None

    return train_data, val_data, test_data

def inspect_dataset(args, train_data):
    if args.inspect_dataset:
        print('Inspecting dataset')
        train_inspector = TextDataInspector(train_data)
        train_inspector.run()
        return True
    return False

def create_datasets(train_data, val_data, test_data, train_text_frequencies, glove_embeddings, args):
    train_dataset = NLPDataset(train_data, train_text_frequencies, glove_embeddings, args)  
    valid_dataset = NLPDataset(val_data,  train_text_frequencies, glove_embeddings, args)
    test_dataset = NLPDataset(test_data,  train_text_frequencies, glove_embeddings, args)
    return train_dataset, valid_dataset, test_dataset

def create_models(args, model_params, random_optimizer, random_lr):
    baseline_model_params = model_params.copy()
    baseline_model_params.update({
        'fc1_width': args.fc1_width,
        'fc2_width': args.fc2_width,
        'output_dim': args.num_classes
    })

    rnn_model_params = model_params.copy()
    rnn_model_params.update({
        'rnn_cell_type': args.rnn_cell_type,
        'hidden_size': args.hidden_size,
        'num_classes': args.num_classes,
        'num_layers': args.num_layers,
        'bidirectional': args.bidirectional
    })

    baseline_model, baseline_criterion, baseline_optimizer = create_model_and_optimizer(BaseLineModel, baseline_model_params, random_optimizer, random_lr)
    rnn_model, rnn_criterion, rnn_optimizer = create_model_and_optimizer(RNN, rnn_model_params, random_optimizer, random_lr)
    return baseline_model, baseline_criterion, baseline_optimizer, rnn_model, rnn_criterion, rnn_optimizer

def create_vocab_and_embeddings(args, train_text_frequencies, random_vocab_size, random_freeze_vector_rep):
    train_text_vocab = Vocab(train_text_frequencies, max_size=random_vocab_size, min_freq=args.min_freq)
    #print the length of the vocab
    print('-------------------Vocab length:', len(train_text_vocab))

    glove_embeddings = load_glove_embeddings(args.glove_file, train_text_vocab, embedding_dim=args.embedding_dim, use_pretrained_embeddings=args.use_pretrained_embeddings, freeze=random_freeze_vector_rep)
    args.glove_embeddings = glove_embeddings
    return glove_embeddings

def train_models(args, baseline_model, rnn_model, train_dataset, valid_dataset, test_dataset, baseline_optimizer, baseline_criterion, rnn_optimizer, rnn_criterion, wandb, verbose=True):
    if verbose:
        print('---------- TRAINING BASELINE MODEL-------------------')
    baseline_metrics = train_and_evaluate_model(baseline_model, train_dataset, valid_dataset, test_dataset, baseline_optimizer, baseline_criterion, args, wandb)
    if verbose:
        print('-------------TRAINING RNN MODEL--------------')
    rnn_metrics = train_and_evaluate_model(rnn_model, train_dataset, valid_dataset, test_dataset, rnn_optimizer, rnn_criterion, args, wandb)
    return baseline_metrics, rnn_metrics

def run_experiment(args, train_data, val_data, test_data, wandb):
    chosen_hyperparameters = []
    all_runs_dir = []
    train_text_frequencies = calculate_frequencies([row[0] for row in train_data])

    # Randomly sample hyperparameters n unique times without replacement
    random_vocab_sizes = random.sample(args.vocab_sizes, k=args.num_samples)
    random_lrs = random.sample(args.learning_rates, k=args.num_samples)
    random_dropouts = random.sample(args.dropouts, k=args.num_samples)
    random_optimizers = random.sample(args.optimizers, k=args.num_samples)
    random_grad_clip_values = random.sample(args.gradient_clipping_values, k=args.num_samples)
    random_freeze_vector_reps = random.sample(args.freezing_vector_rep_values, k=args.num_samples)
    random_activation_functions = random.sample(args.activation_functions, k=args.num_samples)

    for i in range(args.num_samples):
        args.clip = random_grad_clip_values[i]
        args.max_size = random_vocab_sizes[i]
        glove_embeddings = create_vocab_and_embeddings(args, train_text_frequencies, random_vocab_sizes[i], random_freeze_vector_reps[i])

        train_dataset, valid_dataset, test_dataset = create_datasets(train_data, val_data, test_data, train_text_frequencies, glove_embeddings, args)

        model_params = create_model_params(args, random_dropouts[i], random_activation_functions[i])

        baseline_model, baseline_criterion, baseline_optimizer, rnn_model, rnn_criterion, rnn_optimizer = create_models(args, model_params, random_optimizers[i], random_lrs[i])

        with initialize_run(args, wandb, random_vocab_sizes[i], random_lrs[i], random_dropouts[i], random_optimizers[i], random_grad_clip_values[i], random_freeze_vector_reps[i], random_activation_functions[i]) as run:

            if run:
                run_dir = run.dir
                #only append the last part of the run directory
                all_runs_dir.append(run_dir.split('/')[-1] if run_dir else run_dir)

            baseline_metrics, rnn_metrics = train_models(args, baseline_model, rnn_model, train_dataset, valid_dataset, test_dataset, baseline_optimizer, baseline_criterion, rnn_optimizer, rnn_criterion, wandb)

            print('-----------------------------------')
            print('Baseline metrics:', baseline_metrics)
            print('-----------------------------------')
            print('RNN metrics:', rnn_metrics)
            print('-----------------------------------')

        chosen_hyperparameters.append({
            'vocab_size': random_vocab_sizes[i],
            'lr': random_lrs[i],
            'dropout': random_dropouts[i],
            'optimizer': random_optimizers[i],
            'grad_clip_value': random_grad_clip_values[i],
            'freeze_vector_rep': random_freeze_vector_reps[i],
            'activation_function': random_activation_functions[i],
            'baseline_metrics': baseline_metrics,
            'rnn_metrics': rnn_metrics
        })
        
        if args.setup_wandb:
            # log the chosen_hyperparameters to wandb
            wandb.log({'chosen_hyperparameters': chosen_hyperparameters})
    print('-----------------------------------')
    print('All runs directory:', all_runs_dir)
    print('-----------------------------------')
    return chosen_hyperparameters


def find_best_params(chosen_hyperparameters):
    best_val_acc_baseline = None
    best_val_acc_rnn = None
    best_params_baseline = None
    best_params_rnn = None

    for params in chosen_hyperparameters:
        if best_val_acc_baseline is None or params['baseline_metrics']['val_acc'] > best_val_acc_baseline:
            best_val_acc_baseline = params['baseline_metrics']['val_acc']
            best_params_baseline = params

        if best_val_acc_rnn is None or params['rnn_metrics']['val_acc'] > best_val_acc_rnn:
            best_val_acc_rnn = params['rnn_metrics']['val_acc']
            best_params_rnn = params
    return best_params_baseline, best_params_rnn

def main(args):
    wandb = setup(args)
    train_data, val_data, test_data = load_datasets(args)

    if train_data is None or val_data is None or test_data is None:
        return

    if inspect_dataset(args, train_data):
        return

    chosen_hyperparameters = run_experiment(args, train_data, val_data, test_data, wandb)
    print_results(chosen_hyperparameters)
    best_params_baseline, best_params_rnn = find_best_params(chosen_hyperparameters)
    print('-----------------------------------')
    print('Best baseline model')
    print('-----------------------------------')
    print(best_params_baseline)
    print('-----------------------------------')
    print('Best RNN model')
    print('-----------------------------------')
    print(best_params_rnn)
    

if __name__ == '__main__':
    class Args:
        """
        Class representing the arguments for the deep learning model.
        
        Attributes:
            seed (int): The random seed for reproducibility.
            train_filepath (str): The file path for the training data.
            val_filepath (str): The file path for the validation data.
            test_filepath (str): The file path for the test data.
            glove_file (str): The file path for the GloVe word embeddings.
            embedding_dim (int): The dimension of the word embeddings.
            fc1_width (int): The width of the first fully connected layer.
            fc2_width (int): The width of the second fully connected layer.
            output_dim (int): The dimension of the output.
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            lr (float): The learning rate for the optimizer.
            clip (float): The gradient clipping value.
            min_freq (int): The minimum frequency of words to include in the vocabulary.
            max_size (int): The maximum size of the vocabulary.
            device (torch.device): The device to use for training.
            wandb_projectname (str): The name of the Weights & Biases project.
            wandb_entity (str): The Weights & Biases entity.
            log_interval (int): The interval for logging training progress.
            inspect_dataset (bool): Whether to inspect the dataset.
            setup_wandb (bool): Whether to setup wandb for logging.
            wandb_api_key (str): The API key for wandb.
            install_requirements (bool): Whether to install the required packages.
            num_samples (int): The number of times to repeat the randomized search without replacement for hyperparameters. 
            
        """
        seed = 7052020
        # seed = 8200
        train_filepath = './data/sst_train_raw.csv'
        val_filepath = './data/sst_valid_raw.csv'
        test_filepath = './data/sst_test_raw.csv'
        glove_file = './data/sst_glove_6b_300d.txt'
        vocab_sizes = [10000, 15000, 20000] 
        learning_rates = [0.0001, 0.001, 0.01]
        dropouts = [0.2, 0.5, 0.8]
        optimizers = ['SGD', 'Adagrad', 'RMSprop']
        gradient_clipping_values = [0.25, 0.5, 0.75]
        freezing_vector_rep_values = [True, False, True]
        activation_functions = ['LeakyReLU', 'SELU', 'Tanh']
        rnn_cell_type = 'GRU'
        embedding_dim = 300
        hidden_size = 150
        num_classes = 1
        num_layers = 2
        bidirectional = False
        num_samples = 3 # Number of times to repeat the randomized search without replacement for hyperparameters
        epochs = 10
        batch_size = 10
        lr = 1e-4
        clip = 0.25
        min_freq = 1
        max_size = -1
        #parameters for the FCN:
        fc1_width = 150
        fc2_width = 150
        use_pretrained_embeddings = True
        freeze_embeddings = True
        find_best_rnn_cell_type = False
        # device = torch.device("cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        setup_wandb = False
        wandb_projectname = 'lab3_task4'
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



# TASK:
    # TAKE THE BEST HYPERPARAMETER VALUE AND TRAIN THE BASELINE MODEL AND THE BEST MODEL WITH AND WITHOUT PRETRAINED REPRESENTATIONS







        # Set random hyperparameter values

        # current_hyperparameters = {
        #     'vocab_size': random_vocab_size,
        #     'lr': random_lr,
        #     'dropout_rate': random_dropout,
        #     'optimizer': random_optimizer,
        #     'gradient_clipping_value': random_grad_clip_value,
        #     'freeze_vector_rep': random_freeze_vector_rep,
        #     'activation_function': random_activation_function,
        # }

        # chosen_hyperparameters[f'{_}'] = current_hyperparameters




