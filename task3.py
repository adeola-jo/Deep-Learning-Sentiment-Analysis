import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import install_requirements, inspect_dataset, TextDataInspector, read_csv,plot_training_progress, setup_wandb
from nlp import Vocab, NLPDataset, calculate_frequencies, load_glove_embeddings
from networks import RNN
from torch.nn.utils import rnn as rnn_utils
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


seed = 800
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    

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
        # Log to wandb
        # if wandb is not None and batch_num % args.log_interval == 0:
        #     wandb.log({'Train/epoch': epoch, 'Batch': batch_num, 'Train/Loss': total_loss / batch_count, 'Train/Accuracy': total_correct / batch_count * 100})
            
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
    glove_embeddings = load_glove_embeddings(args.glove_file, train_text_vocab, embedding_dim=args.embedding_dim, use_pretrained_embeddings=args.use_pretrained_embeddings, freeze=args.freeze_embeddings)
    args.glove_embeddings = glove_embeddings #add the glove embeddings to the args object

    # # # Initialize datasets
    train_dataset = NLPDataset(train_data, train_text_frequencies, glove_embeddings, args)  
    valid_dataset = NLPDataset(val_data,  train_text_frequencies, glove_embeddings, args)  # 

    model = RNN(args.rnn_cell_type, args.embedding_dim, args.hidden_size, args.num_classes, args.num_layers, bidirectional=args.bidirectional, dropout_rate=args.dropout_rate, device=args.device)
    # model = RNN(args.embedding_dim, args.hidden_size, args.num_classes, args.num_layers, args.device)

    model.to(args.device) #send the model to the device

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(model, train_dataset, valid_dataset, optimizer, criterion, args, wandb=wandb)




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
        """
        # seed = 8200
        train_filepath = './data/sst_train_raw.csv'
        val_filepath = './data/sst_valid_raw.csv'
        test_filepath = './data/sst_test_raw.csv'
        glove_file = './data/sst_glove_6b_300d.txt'
        # seed = 7052020
        embedding_dim = 300
        hidden_size = 150
        bidirectional = False
        dropout_rate = 0.5
        rnn_cell_type = 'GRU'
        num_classes = 1
        num_layers = 2
        epochs = 5
        batch_size = 10
        lr = 1e-4
        clip = 0.25
        min_freq = 1
        max_size = -1
        freeze_embeddings = True
        use_pretrained_embeddings = True
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











