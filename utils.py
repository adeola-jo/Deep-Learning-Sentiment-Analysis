import os
import ast
import sys
import subprocess
import importlib.util
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import ast
import subprocess
import sys
import importlib.util
import string
import io
import wandb
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from collections import Counter




class DummyContextManager:
    def __init__(self) -> None:
        self.dir = None

    def __enter__(self):
        return self 

    def __exit__(self, exc_type, exc_value, traceback):
        return False
    
class RequirementsInstaller:
    """
    A class that installs required packages based on the imports in a given Python file.

    Args:
        filepath (str): The path to the Python file.

    Methods:
        get_imports(): Returns a list of all the imports in the Python file.
        install_packages(): Installs the required packages using pip.
    """

    def __init__(self, filepath):
        self.filepath = filepath

    def get_imports(self):
        with open(self.filepath, 'r') as file:
            root = ast.parse(file.read())

        # Collect all module names from 'import' statements
        imports = [node.names[0].name.split('.')[0] for node in ast.walk(root) if isinstance(node, ast.Import)]
        # Collect all module names from 'from ... import' statements
        import_froms = [node.module.split('.')[0] for node in ast.walk(root) if isinstance(node, ast.ImportFrom) and node.module is not None]
        all_imports = set(imports + import_froms)

        # Filter out modules that are either in the standard library or already installed
        third_party_imports = [name for name in all_imports if not importlib.util.find_spec(name)]
        return third_party_imports

    def install_packages(self):
        packages = self.get_imports()

        for package in packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f'Successfully installed {package}')
            except subprocess.CalledProcessError:
                print(f'Failed to install {package}. Please check if the package name is correct.')

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

def install_requirements(filepath):
    """
    Installs required packages based on the imports in a given Python file.

    Args:
        filepath (str): The path to the Python file.
    """
    installer = RequirementsInstaller(filepath)
    installer.install_packages()




def plot_training_progress(data, show=False, save_dir=None, use_wandb=False, wandb=None, description=None):
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

    linewidth = 2
    legend_size = 10
    train_color = 'm'
    val_color = 'c'

    num_points = len(data['train_loss'])
    x_data = np.linspace(1, num_points, num_points)
    ax1.set_title('Binary Cross-entropy loss')
    ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
            linewidth=linewidth, linestyle='-', label='train')
    ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
            linewidth=linewidth, linestyle='-', label='validation')
    ax1.legend(loc='upper right', fontsize=legend_size)
    ax2.set_title('Average class accuracy')
    ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
            linewidth=linewidth, linestyle='-', label='train')
    ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
            linewidth=linewidth, linestyle='-', label='validation')
    ax2.legend(loc='upper left', fontsize=legend_size)
    ax3.set_title('Learning rate')
    ax3.plot(x_data, data['lr'], marker='o', color=train_color,
            linewidth=linewidth, linestyle='-', label='learning_rate')
    ax3.legend(loc='upper left', fontsize=legend_size)

    # add descriptions to the bottom of the plot
    if description is not None:
        # Convert the dictionary to a string
        description_str = str(description)

        # Ensure the description doesn't overflow so break it up into multiple lines
        description_str = description_str.split(' ')
        description_str = [description_str[i:i + 18] for i in range(0, len(description_str), 18)]
        description_str = [' '.join(line) for line in description_str]
        description_str = '\n'.join(description_str)

        # Add the description to the bottom of the plot
        fig.text(0.5, 0.01, description_str, horizontalalignment='center', fontsize=10)


    if save_dir is not None:
        save_path = os.path.join(save_dir)#, 'training_plot.png')
        print('Plotting in: ', save_path)
        plt.savefig(save_path)
    if show:
        plt.show()
    if use_wandb:
        #save the image in the buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        #log the image to wandb
        wandb.log({'Metric plots': wandb.Image(image)})
        buf.close()


if __name__ == "__main__":
    # Use the current file as the filepath
    install_requirements(os.path.abspath(__file__))