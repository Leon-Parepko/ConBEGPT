import torch.nn as nn

class Estimator(nn.Module):
    """
    A simple bag-of-words linear estimator for the toxicity of a sentence.
    """
    def __init__(self, hidden_dim, dropout, vocab_len):
        super(Estimator, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_len, hidden_dim, sparse=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the embedding and linear layers.
        :return: None
        """
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        """
        Forward pass of the estimator.
        :param text: input text torch tensor (batch_size, seq_len)
        :param offsets: offsets of the text (None for unbatched)
        :return: output of the estimator
        """
        x = self.embedding(text, offsets)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x