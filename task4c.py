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
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from utils import install_requirements
from torch.nn.utils import rnn as rnn_utils
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# seed = 10000
seed = 7000000
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


class DummyContextManager:
    def __init__(self) -> None:
        self.dir = None

    def __enter__(self):
        return self 

    def __exit__(self, exc_type, exc_value, traceback):
        return False
    
class MeanPoolingLayer(nn.Module):
    def forward(self, x, x_lengths):
        """
        Performs mean pooling across the temporal dimension of the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, n, embedding_dim].
            x_lengths (list): List of lengths for each input sequence in the batch.

        Returns:
            torch.Tensor: Mean pooled tensor of shape [batch_size, embedding_dim].
        """
        # Sum pooling across the temporal dimension 
        sum_pool = torch.sum(x, dim=1)  # Assuming x has shape [batch_size, n, embedding_dim]
        # Calculate mean by dividing sum by original lengths
        mean_pool = sum_pool / x_lengths
        return mean_pool

class BaseLineModel(nn.Module):
    """
    BaseLineModel is a deep learning model for text classification.

    Args:
        embedding_dim (int): The dimensionality of the input word embeddings.
        fc1_width (int): The width of the first fully connected layer.
        fc2_width (int): The width of the second fully connected layer.
        output_dim (int): The number of output classes.

    Attributes:
        mean_pool (MeanPoolingLayer): The mean pooling layer.
        fc1 (nn.Linear): The first fully connected layer.
        relu1 (nn.ReLU): The ReLU activation function.
        fc2 (nn.Linear): The second fully connected layer.
        relu2 (nn.ReLU): The ReLU activation function.
        fc3 (nn.Linear): The final fully connected layer.

    Methods:
        forward(batch_texts, batch_text_lengths=None): Performs forward pass of the model.
        initialize_parameters(): Initializes the parameters of the model.
    """

    def __init__(self, embedding_dim: int, fc1_width: int, fc2_width: int, output_dim: int):
        super(BaseLineModel, self).__init__()
        self.mean_pool = MeanPoolingLayer() 
        self.fc1 = nn.Linear(embedding_dim, fc1_width)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(fc1_width, fc2_width)
        self.fc3 = nn.Linear(fc2_width, output_dim)
        self.relu = nn.ReLU()
        self.reset_parameters()
 
        
    def forward(self, batch_texts, batch_text_lengths):
        """
        Performs forward pass of the model.

        Args:
            batch_texts (torch.Tensor): The input batch of texts.
            batch_text_lengths (list): A list of the original lengths of
                                       each input sequences in the batch.

        Returns:
            torch.Tensor: The output tensor of the model.
        """

        #move the batch_texts and batch_text_lengths to the device
        batch_texts = batch_texts.to(args.device)
        batch_text_lengths = batch_text_lengths.to(args.device)
        # Process each sequence in the batch individually
        x = self.mean_pool(batch_texts, batch_text_lengths)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def reset_parameters(self):
        """
        Initializes the parameters of the model.
        """
        # Kaiming initialization for layers with ReLU activation
        for layer in self.modules():
            if isinstance(layer, nn.Linear) and layer != self.fc3:
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        self.fc3.reset_parameters()  # Reset final layer parameters to uniform distribution



class RNN(nn.Module):
    def __init__(self, rnn_cell_type, embedding_dim, hidden_size, num_classes=1, num_layers=2, bidirectional=False, dropout_rate=0.5, device=torch.device('cpu')):
        super(RNN, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(p=dropout_rate)
        self.rnn_cell_type = rnn_cell_type
        # self.layer_norm = nn.LayerNorm(hidden_size)

        if rnn_cell_type == 'RNN':
            self.rnn = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout_rate if num_layers > 1 else 0
            )
        elif rnn_cell_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout_rate if num_layers > 1 else 0
            )
        elif rnn_cell_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout_rate if num_layers > 1 else 0
            )

        self.fc1 = nn.Linear(hidden_size * (2 if bidirectional else 1), hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.init_weights()

    def forward(self, x, lengths):
        # x = x.permute(1, 0, 2)
        x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), lengths.size(0), self.hidden_size).to(self.device)

        if self.rnn_cell_type == 'LSTM':
            c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), lengths.size(0), self.hidden_size).to(self.device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)

        out, _ = rnn_utils.pad_packed_sequence(out, batch_first=True)
        # out = out[-1, :, :]
        out = out[torch.arange(out.size(0)), lengths - 1]
        # out = self.layer_norm(out)
        # out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu1(out)
        # out = self.dropout(out)
        out = self.fc2(out)
        return out

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.data.size()) >= 2:
                    nn.init.xavier_uniform_(param.data)
                else:
                    nn.init.normal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)



class Instance:
    """
    Represents an instance with text and label.
    Used for each example in the dataset.

    Attributes:
        text (str): The text of the instance.
        label (int or str): The label of the instance.
    """
    def __init__(self, text, label):
        self.text = text
        self.label = label


class TextDataInspector:
    """
    A class for inspecting text data.

    Args:
        data (pandas.DataFrame): The input data containing text and labels.

    Attributes:
        data (pandas.DataFrame): The input data containing text and labels.

    Methods:
        tokenize_text: Tokenizes the text data.
        plot_most_common_tokens: Plots the most common tokens.
        plot_least_common_tokens: Plots the least common tokens.
        plot_token_frequency_distribution: Plots the token frequency distribution.
        plot_sentence_length_distribution: Plots the distribution of sentence lengths.
        plot_label_count: Plots the count of each label.
        show_word_cloud: Generates and displays a word cloud.
        run: Executes all the plotting methods.

    """

    def __init__(self, data):
        self.data = data

    def tokenize_text(self):
        # tokens = self.data.iloc[:, 0].apply(word_tokenize)
        tokens = [row[0].split() for row in self.data]
        tokens = [item for sublist in tokens for item in sublist]
        tokens = [token for token in tokens if token not in string.punctuation]
        return tokens

    def plot_most_common_tokens(self):
        token_counts = Counter(self.tokenize_text())
        plt.figure(figsize=(10, 10))
        plt.barh([x[0] for x in token_counts.most_common(10)], [x[1] for x in token_counts.most_common(10)])
        plt.gca().invert_yaxis()
        plt.xlabel('Frequency')
        plt.title('10 Most Common Tokens')
        plt.show()

    def plot_least_common_tokens(self):
        token_counts = Counter(self.tokenize_text())
        plt.figure(figsize=(10, 10))
        plt.barh([x[0] for x in token_counts.most_common()[-10:]], [x[1] for x in token_counts.most_common()[-10:]])
        plt.gca().invert_yaxis()
        plt.xlabel('Frequency')
        plt.title('10 Least Common Tokens')
        plt.show()

    def plot_token_frequency_distribution(self):
        token_counts = Counter(self.tokenize_text())
        plt.figure(figsize=(10, 10))
        plt.hist([x[1] for x in token_counts.most_common()], bins=100)
        plt.xlabel('Frequency')
        plt.ylabel('Number of Tokens')
        plt.title('Token Frequency Distribution')
        plt.show()

    def plot_sentence_length_distribution(self):
        sns.histplot(self.data.iloc[:, 0].str.len())

    def plot_label_count(self):
        sns.countplot(x=self.data.iloc[:, 1])

    def show_word_cloud(self, title, sentiment='positive'):
        reviews = self.data[self.data.iloc[:, 1] == f' {sentiment}']
        reviews_text = " ".join(review for review in reviews.iloc[:, 0])
        cloud = WordCloud(stopwords=STOPWORDS, background_color="white").generate(reviews_text)
        
        plt.figure(figsize = (16, 10))
        plt.imshow(cloud, interpolation='bilinear')
        plt.title(title)
        plt.axis("off")
        plt.show()

    def run(self):
        self.plot_most_common_tokens()
        self.plot_least_common_tokens()
        self.plot_token_frequency_distribution()
        self.plot_sentence_length_distribution()
        self.plot_label_count()
        self.show_word_cloud('Positive Reviews', sentiment='positive')
        self.show_word_cloud('Negative Reviews', sentiment='negative')


class NLPDataset(Dataset):
    """
    A PyTorch dataset for Natural Language Processing tasks.

    Args:
        data (pandas.DataFrame): The input data containing text and labels.
        glove_embedding_matrix (torch.Tensor): The pre-trained GloVe embedding matrix.
        args (argparse.Namespace): Additional arguments.

    Attributes:
        instances (list): List of instances in the dataset.
        vocab (Vocab): Vocabulary object for encoding text.
        label_vocab (Vocab): Vocabulary object for encoding labels.
        tokenizer (function): Tokenizer function for tokenizing text.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the text indices and label for a given index.
        from_file(file_path): Creates an NLPDataset object from a CSV file.

    """
    def __init__(self, data=None, text_frequencies=None, glove_embedding_matrix=None, args=None):
        if data is not None:
            self.instances = []
            self.vocab = Vocab(text_frequencies, max_size=args.max_size, min_freq=args.min_freq)
            self.label_vocab = Vocab({}, is_target_field=True)
            
            for row in data:
                # The first column is the text and the second column is the label
                text = row[0]
                label = row[1]
                self.instances.append(Instance(text, label))
    
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]
        text_indices = self.vocab.encode(instance.text.split())
        #encode the label
        label = self.label_vocab.encode(instance.label)
        # Check if text_indices is already a tensor
        if not isinstance(text_indices, torch.Tensor):
            text_indices = torch.tensor(text_indices, dtype=torch.long)
        return text_indices, label

    def from_file(self, file_path):
        data = pd.read_csv(file_path)
        return NLPDataset(data)


class Vocab:
    """
    A class representing a vocabulary.

    Attributes:
    - frequencies (dict): A dictionary containing word frequencies.
    - max_size (int): The maximum size of the vocabulary.
    - min_freq (int): The minimum frequency threshold for including words in the vocabulary.
    - is_target_field (bool): Indicates whether the vocabulary is for the target field.

    Methods:
    - __init__(self, frequencies, max_size=-1, min_freq=0, is_target_field=False): Initializes the Vocab object.
    - encode(self, tokens): Encodes a list of tokens into a list of indices.
    - decode(self, indices): Decodes a list of indices into a list of tokens.
    - __len__(self): Returns the size of the vocabulary.
    - __getitem__(self, key): Returns the word corresponding to the given index or the index corresponding to the given word.
    """

    def __init__(self, frequencies, max_size=-1, min_freq=1, is_target_field=False):
        """
        Initializes the Vocab object.

        Parameters:
        - frequencies (dict): A dictionary containing word frequencies.
        - max_size (int): The maximum size of the vocabulary. Default is -1, which means no maximum size.
        - min_freq (int): The minimum frequency threshold for including words in the vocabulary. Default is 0.
        - is_target_field (bool): Indicates whether the vocabulary is for the target field. Default is False.
        """
        if is_target_field:
            self.word2idx = {" positive": 0, " negative": 1}
            self.idx2word = {0: " positive", 1: " negative"}
        else:
            self.word2idx = {"<PAD>": 0, "<UNK>": 1}
            self.idx2word = {0: "<PAD>", 1: "<UNK>"}

        if not is_target_field:
            sorted_words = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)

            if max_size == -1:
                max_size = len(sorted_words) + 2
                for word, freq in sorted_words:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
            else:
                for word, freq in sorted_words:
                    if freq >= min_freq and len(self.word2idx) <= max_size:
                        idx = len(self.word2idx)
                        self.word2idx[word] = idx
                        self.idx2word[idx] = word

    def encode(self, tokens):
        """
        Encodes a list of tokens into a list of indices.

        Parameters:
        - tokens (list): A list of tokens (strings).

        Returns:
        - indices (torch.Tensor): A tensor containing the indices corresponding to the input tokens.
        """
        if isinstance(tokens, str):
            # indices = [self.word2idx.get(tokens, self.word2idx.get("<UNK>", -1))]
            indices = [self.word2idx.get(tokens, self.word2idx.get("<UNK>"))]

        else:
            indices = [self.word2idx.get(token, self.word2idx.get("<UNK>")) for token in tokens]
            # indices = [self.word2idx.get(token, self.word2idx.get("<UNK>", -1)) for token in tokens]
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices):
        """
        Decodes a list of indices into a list of tokens.

        Parameters:
        - indices (list): A list of indices.

        Returns:
        - tokens (list): A list of tokens corresponding to the input indices.
        """
        if isinstance(indices, int):
            tokens = [self.idx2word.get(indices, "<UNK>")]
        else:
            tokens = [self.idx2word.get(index, "<UNK>") for index in indices]
        return tokens
    
    def __len__(self):
        """
        Returns the size of the vocabulary.

        Returns:
        - size (int): The size of the vocabulary.
        """
        return len(self.word2idx)
    
    def __getitem__(self, key):
        """
        Returns the word corresponding to the given index or the index corresponding to the given word.

        Parameters:
        - key (int or str): The index or word.

        Returns:
        - word or index: The word corresponding to the given index or the index corresponding to the given word.
        """
        if isinstance(key, int):
            return self.idx2word.get(key, "<UNK>")
        elif isinstance(key, str):
            return self.word2idx.get(key, -1)
        else:
            raise TypeError("Invalid argument type for Vocab __getitem__")
    
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
    # Initialize the wandb project
    # wandb.init(project=args.wandb_projectname, entity=args.wandb_entity, config=vars(args)) 
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
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_fn)
    for epoch in range(args.epochs):
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{args.epochs}', unit='batch') as pbar:
            train_loss, train_acc, _ = train_epoch(model, train_loader, optimizer, criterion, args.device, epoch + 1, args, pbar=pbar, wandb=wandb)
            val_loss, val_acc, _ = evaluate(model, val_data, criterion, args)

            print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')
            # Log to wandb
            if wandb is not None:
                wandb.log({'Epoch': epoch + 1, 'Train/Loss': train_loss, 'Train/Accuracy': train_acc, 'Val/Loss': val_loss, 'Val/Accuracy': val_acc})
    return val_acc, val_loss

def inspect_dataset(dataset):
    """
    This function takes a dataset as input and performs the following operations:
    1. Converts the dataset into a pandas Series.
    2. Prints the descriptive statistics of the dataset.
    3. Plots a histogram of the sequence lengths in the dataset.

    Parameters:
    - dataset: The input dataset.

    Returns:
    None
    """
    data = pd.Series(dataset)
    print(data.describe())
    plt.hist(dataset, bins=10)
    plt.title('Histogram of sequence lengths in the dataset')
    plt.xlabel('Sequence length')
    plt.ylabel('Frequency')
    plt.show()
        
def read_csv(filepath):
    with open(os.path.abspath(filepath), 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

def find_best_rnn_cell_type(args, wandb, train_dataset, valid_dataset, test_dataset):
    all_runs_dir = []
    best_metrics = {'accuracy': 0.0, 'loss': float('inf'), 'rnn_cell_type': None}
    for rnn_cell_type in args.rnn_cell_types:
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
    
    # Initialize the vocabular
    train_text_frequencies = calculate_frequencies([row[0] for row in train_data])
    train_text_vocab = Vocab(train_text_frequencies, max_size=args.max_size, min_freq=args.min_freq)
    # # Load GloVe embeddings. We only need the embeddings for the training text vocab
    glove_embeddings = load_glove_embeddings(args.glove_file, train_text_vocab, embedding_dim=args.embedding_dim)
    args.glove_embeddings = glove_embeddings #add the glove embeddings to the args object

    # # # Initialize datasets
    train_dataset = NLPDataset(train_data, train_text_frequencies, glove_embeddings, args)  
    valid_dataset = NLPDataset(val_data,  train_text_frequencies, glove_embeddings, args)  # Assuming data is your validation data
    test_dataset = NLPDataset(test_data,  train_text_frequencies, glove_embeddings, args)  # Assuming data is your test data
    # # You should also prepare valid_dataset and test_dataset similarly if available
    # model = BaseLineModel(embedding_dim=args.embedding_dim, fc1_width=args.fc1_width, fc2_width=args.fc2_width, output_dim=args.output_dim)
    # model.to(args.device) #send the model to the device
    # Iterate over hyperparameter combinations


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
    best_metrics = {'accuracy': 0.0, 'loss': float('inf')}
    all_runs_dir = []
    chosen_hyperparameters = {} #dictionary for storing the chosen hyperparameters for each run

    # ========================= USING RANDOM CHOICE ===========================
    # # # Randomized search
    for _ in range(args.num_repeat):
        random_hidden_size = random.choice(args.hidden_sizes)
        random_num_layers = random.choice(args.num_layers_values)
        random_dropout = random.choice(args.dropout_values)
        random_bidirectional = random.choice(args.bidirectional_values)
        chosen_hyperparameters[f'{_}'] = {'hidden_size': random_hidden_size, 'num_layers': random_num_layers, 'dropout': random_dropout, 'bidirectional': random_bidirectional}
        for rnn_cell_type in args.rnn_cell_types:
            all_runs_dir, best_metrics = run_experiment(args, wandb, rnn_cell_type, random_hidden_size, random_num_layers, random_dropout, random_bidirectional, train_dataset, valid_dataset, test_dataset, all_runs_dir, best_metrics)
            print('\n \n ===============================NEW RUN======================================')

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
    val_acc, val_loss = train_and_validate(model, train_dataset, valid_dataset, optimizer, criterion, args, wandb=wandb)

    test_loss, test_acc, _ = evaluate(model, test_dataset, criterion, args)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

    if wandb is not None:
        wandb.log({'Test/Loss': test_loss, 'Test/Accuracy': test_acc})

    return val_acc, val_loss

def initialize_run(args, wandb, rnn_cell_type, hidden_size, num_layers, dropout, bidirectional):
    config = {
        'seed': args.seed,
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

    if args.setup_wandb:
        return wandb.init(project=args.wandb_projectname, entity=args.wandb_entity, config=config)
    else:
        return DummyContextManager()

def run_experiment(args, wandb, rnn_cell_type, hidden_size, num_layers, dropout, bidirectional, train_dataset, valid_dataset, test_dataset, all_runs_dir=None, best_metrics=None):
    best_hyperparameters = None
    with initialize_run(args, wandb, rnn_cell_type, hidden_size, num_layers, dropout, bidirectional) as run:
        if run:
            run_dir = run.dir
            #only append the last part of the run directory
            all_runs_dir.append(run_dir.split('/')[-1] if run_dir else run_dir)

        model, optimizer, criterion = initialize_model(args, rnn_cell_type, hidden_size, num_layers, dropout, bidirectional)
        with run:
            val_acc, val_loss = train_and_evaluate(args, model, train_dataset, valid_dataset, test_dataset, optimizer, criterion, wandb=run if args.setup_wandb else None)
        
        if val_acc > best_metrics['accuracy']:
            best_metrics['accuracy'] = val_acc
            best_metrics['loss'] = val_loss
            best_metrics['rnn_cell_type'] = rnn_cell_type
            best_metrics['hidden_size'] = hidden_size
            best_metrics['num_layers'] = num_layers
            best_metrics['dropout'] = dropout
            best_metrics['bidirectional'] = bidirectional
    return all_runs_dir, best_metrics


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
            num_repeat (int): The number of times to repeat the randomized search without replacement for hyperparameters. 
            
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
        dropout_values = [0.5, 0.4, 0.7]
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












