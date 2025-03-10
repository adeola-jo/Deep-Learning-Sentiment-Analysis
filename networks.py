import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F


torch.softmax 
torch.nn.functional.softmax


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

    def __init__(self, embedding_dim: int, fc1_width: int, fc2_width: int, output_dim: int, dropout_rate: float = 0.5, device: torch.device = torch.device('cpu'), activation_function: torch.nn = nn.ReLU()):

        super(BaseLineModel, self).__init__()
        self.device = device
        self.mean_pool = MeanPoolingLayer() 
        self.fc1 = nn.Linear(embedding_dim, fc1_width)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(fc1_width, fc2_width)
        self.fc3 = nn.Linear(fc2_width, output_dim)
        self.activation_function = activation_function
 
        
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

        # Convert the batch text lengths to a 1D tensor
        batch_text_lengths = batch_text_lengths.view(-1,1)

        #move the batch_texts and batch_text_lengths to the device
        batch_texts = batch_texts.to(self.device)
        batch_text_lengths = batch_text_lengths.to(self.device)
        # Process each sequence in the batch individually
        x = self.mean_pool(batch_texts, batch_text_lengths)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.activation_function(self.fc1(x))
        x = self.dropout1(x)
        x = self.activation_function(self.fc2(x))
        x = self.fc3(x)
        return x
    


class RNN(nn.Module):
    def __init__(self, rnn_cell_type, embedding_dim, hidden_size, num_classes=1, num_layers=2, bidirectional=False, dropout_rate=0.5, device=torch.device('cpu'),
    activation_function: torch.nn = nn.ReLU()):
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
        self.activation_function = activation_function
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        # x = x.permute(1, 0, 2)
        x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Initialize hidden state.
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), lengths.size(0), self.hidden_size).to(self.device)

        if self.rnn_cell_type == 'LSTM':
            c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), lengths.size(0), self.hidden_size).to(self.device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)

        # extract the last hidden state of the last element in each sequence
        out, _ = rnn_utils.pad_packed_sequence(out, batch_first=True)
        #NOTE: Extract up to the last valid element in each sequence
        out = out[torch.arange(out.size(0)), lengths - 1]
        # out = self.layer_norm(out)
        # out = self.dropout(out)
        out = self.fc1(out)
        out = self.activation_function(out)
        # out = self.dropout(out)
        out = self.fc2(out)
        return out

class BahdanauAttention(nn.Module):
    def __init__(self, input_size, output_size):
        super(BahdanauAttention, self).__init__()
        self.W_1 = nn.Linear(input_size, output_size)

    def forward(self, h):
        a = torch.tanh(self.W_1(h))
        alpha = F.softmax(a, dim=1)
        out_attn = torch.sum(alpha * h, dim=1)
        return out_attn


class RNNWithAttention(nn.Module):
    def __init__(self, rnn_cell_type, embedding_dim, hidden_size, num_classes=1, num_layers=2, bidirectional=False, dropout_rate=0.5, device=torch.device('cpu'),
                 activation_function: torch.nn = nn.ReLU(), use_attention=False):
        super(RNNWithAttention, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(p=dropout_rate)
        self.rnn_cell_type = rnn_cell_type
        self.use_attention = use_attention

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

        if use_attention:
            self.attention = BahdanauAttention(hidden_size * (2 if bidirectional else 1), hidden_size // 2)

        self.fc1 = nn.Linear(hidden_size * (2 if bidirectional else 1), hidden_size)
        self.activation_function = activation_function
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Initialize hidden state.
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), lengths.size(0), self.hidden_size).to(self.device)

        if self.rnn_cell_type == 'LSTM':
            c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), lengths.size(0), self.hidden_size).to(self.device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)

        if self.use_attention:
            context_vector = self.attention(out)
            out = torch.cat([out[:, -1, :], context_vector], dim=1)
        else:
            out, _ = rnn_utils.pad_packed_sequence(out, batch_first=True)
            out = out[torch.arange(out.size(0)), lengths - 1]

        # Apply the remaining layers of your model
        out = self.fc1(out)
        out = self.activation_function(out)
        out = self.fc2(out)

        return out



# class BahdanauAttention(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(BahdanauAttention, self).__init__()
#         self.W1 = nn.Linear(input_size, hidden_size, bias=False)
#         self.W2 = nn.Linear(input_size, hidden_size, bias=False)
#         self.V = nn.Linear(hidden_size, 1, bias=False)

#     def forward(self, x):
#         keys = self.W1(x)
#         values = self.W2(x)
#         attention_scores = torch.tanh(keys + values)
#         attention_weights = F.softmax(self.V(attention_scores), dim=1)
#         context_vector = torch.sum(attention_weights * values, dim=1)
#         return context_vector



# class BahdanauAttention(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(BahdanauAttention, self).__init__()
#         self.W_1 = nn.Linear(input_size, output_size)

#     def forward(self, h):
#         a = torch.tanh(self.W_1(h))
#         alpha = F.softmax(a, dim=1)
#         out_attn = torch.sum(alpha * h, dim=1)
#         return out_attn
    
# class RNNWithAttention(nn.Module):
#     def __init__(self, rnn_cell_type, embedding_dim, hidden_size, num_classes=1, num_layers=2, bidirectional=False, dropout_rate=0.5, device=torch.device('cpu'),
#                  activation_function: torch.nn = nn.ReLU(), use_attention=False):
#         super(RNNWithAttention, self).__init__()
#         self.device = device
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.bidirectional = bidirectional
#         self.dropout = nn.Dropout(p=dropout_rate)
#         self.rnn_cell_type = rnn_cell_type
#         self.use_attention = use_attention

#         if rnn_cell_type == 'RNN':
#             self.rnn = nn.RNN(
#                 input_size=embedding_dim,
#                 hidden_size=hidden_size,
#                 num_layers=num_layers,
#                 batch_first=True,
#                 bidirectional=bidirectional,
#                 dropout=dropout_rate if num_layers > 1 else 0
#             )
#         elif rnn_cell_type == 'GRU':
#             self.rnn = nn.GRU(
#                 input_size=embedding_dim,
#                 hidden_size=hidden_size,
#                 num_layers=num_layers,
#                 batch_first=True,
#                 bidirectional=bidirectional,
#                 dropout=dropout_rate if num_layers > 1 else 0
#             )
#         elif rnn_cell_type == 'LSTM':
#             self.rnn = nn.LSTM(
#                 input_size=embedding_dim,
#                 hidden_size=hidden_size,
#                 num_layers=num_layers,
#                 batch_first=True,
#                 bidirectional=bidirectional,
#                 dropout=dropout_rate if num_layers > 1 else 0
#             )

#         if use_attention:
#             self.attention = BahdanauAttention(hidden_size * (2 if bidirectional else 1), hidden_size)

#         self.fc1 = nn.Linear(hidden_size * (2 if bidirectional else 1), hidden_size)
#         self.activation_function = activation_function
#         self.fc2 = nn.Linear(hidden_size, num_classes)
#         self.init_weights()

#     def forward(self, x, lengths):
#         # x = x.permute(1, 0, 2)
#         x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

#         # Initialize hidden state.
#         h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), lengths.size(0), self.hidden_size).to(self.device)

#         if self.rnn_cell_type == 'LSTM':
#             c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), lengths.size(0), self.hidden_size).to(self.device)
#             out, _ = self.rnn(x, (h0, c0))
#         else:
#             out, _ = self.rnn(x, h0)

#         if self.use_attention:
#             context_vector = self.attention(out)
#             out = torch.cat([out[:, -1, :], context_vector], dim=1)
#         else:
#             # Extract the last hidden state of the last element in each sequence
#             out, _ = rnn_utils.pad_packed_sequence(out, batch_first=True)
#             out = out[torch.arange(out.size(0)), lengths - 1]

#         # Apply the remaining layers of your model
#         out = self.fc1(out)
#         out = self.activation_function(out)
#         out = self.fc2(out)

#         return out
