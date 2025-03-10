
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from collections import Counter


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
        print(f"Creating Vocab instance with is_target_field={is_target_field}")
    
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
                    if freq >= min_freq and len(self.word2idx) < max_size:
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
            indices = [self.word2idx.get(token, self.word2idx.get("<UNK>")) for token in tokens.copy()]
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

