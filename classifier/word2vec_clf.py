import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from classifier import *
from torch.nn import BCEWithLogitsLoss
from classifier.metric import *
from torch import nn


###################################################################
# Word CNN from https://github.com/cezannec/CNN_Text_Classification

class WordCNN(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""

    def __init__(self,
                 #  pretrained_embedding=None,
                 #  freeze_embedding=False,
                 #  vocab_size=None,
                 #  embed_dim=300,
                 filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 num_classes=6,
                 dropout=0.5):
        """
        The constructor for CNN_NLP class.

        Args:
            # pretrained_embedding (torch.Tensor): Pretrained embeddings with
            #     shape (vocab_size, embed_dim)
            # freeze_embedding (bool): Set to False to fine-tune pretraiend
            #     vectors. Default: False
            # vocab_size (int): Need to be specified when not pretrained word
            #     embeddings are not used.
            # embed_dim (int): Dimension of word vectors. Need to be specified
            #     when pretrained word embeddings are not used. Default: 300
            filter_sizes (List[int]): List of filter sizes. Default: [3, 4, 5]
            num_filters (List[int]): List of number of filters, has the same
                length as `filter_sizes`. Default: [100, 100, 100]
            n_classes (int): Number of classes. Default: 2
            dropout (float): Dropout rate. Default: 0.5
        """

        super(WordCNN, self).__init__()

        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=300,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)
        self._random_initalize()

    def _random_initalize(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def forward(self, input_ids):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """

        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        # x_embed = self.embedding(input_ids).float()

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        # x_reshaped = x_embed.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(input_ids)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)

        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc(self.dropout(x_fc))

        return logits
###################################################################

class MultiLabelWordCNNClassifier(NNClassifier):
    def __init__(self,
                 training: TensorDataset,
                 validation: TensorDataset,
                 num_labels: int
                 ):
        super(MultiLabelWordCNNClassifier, self).__init__(WordCNN, training, validation, {'num_classes': num_labels})

        self.optim = self._adam
        self.criterion = BCEWithLogitsLoss()

    def _pred(self, x: Tensor):
        """
        argmax prediction - use the highest value as prediction
        :param x:
        :return:
        """
        self.network.eval()
        with torch.no_grad():
            pred = self.network(x.float())
            pred = (torch.sigmoid(pred) > 0.5).to(torch.int)
        self.network.train()
        return pred


class MultiClassWordCNNClassifier(NNClassifier):
    def __init__(self,
                 training: TensorDataset,
                 validation: TensorDataset,
                 num_classes: int
                 ):
        super(MultiClassWordCNNClassifier, self).__init__(WordCNN, training, validation, {'num_classes': num_classes})

        self.optim = self._adam
        self.criterion = CrossEntropyLoss()


class WordLogisticRegression(Module):
    def __init__(self, num_classes: int):
        super(WordLogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(300, num_classes)

    def forward(self, x: Tensor):
        sentence_embedding = torch.sum(x, dim=2)
        output = self.linear(sentence_embedding)
        return output


class MultiLabelWordLogisticRegressionClassifier(NNClassifier):
    def __init__(self,
                 training: TensorDataset,
                 validation: TensorDataset,
                 num_labels: int
                 ):
        super(MultiLabelWordLogisticRegressionClassifier, self).__init__(
            WordLogisticRegression, training, validation, {'num_classes': num_labels})

        self.optim = self._adam
        self.criterion = BCEWithLogitsLoss()

    def _pred(self, x: Tensor):
        """
        argmax prediction - use the highest value as prediction
        :param x:
        :return:
        """
        self.network.eval()
        with torch.no_grad():
            pred = self.network(x.float())
            pred = (torch.sigmoid(pred) > 0.5).to(torch.int)
        self.network.train()
        return pred


class MultiClassWordLogisticRegressionClassifier(NNClassifier):
    def __init__(self,
                 training: TensorDataset,
                 validation: TensorDataset,
                 num_classes: int
                 ):
        super(MultiClassWordLogisticRegressionClassifier, self).__init__(
            WordLogisticRegression, training, validation, {'num_classes': num_classes})

        self.optim = self._adam
        self.criterion = CrossEntropyLoss()
