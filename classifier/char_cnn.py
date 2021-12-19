from classifier import *
from torch.nn import BCEWithLogitsLoss
from classifier.metric import *
from torch import nn


class CharCNN(Module):
    def __init__(self, num_chars: int, alphabet_size: int, num_output: int):
        super(CharCNN, self).__init__()
        self.alphabet_size = alphabet_size
        self.num_chars = num_chars
        self.num_labels = num_output

        channel_size = 256
        kernel_sizes = [7, 7, 3, 3, 3, 3]

        self.conv = nn.Sequential(
            nn.Conv1d(alphabet_size, channel_size, kernel_sizes[0]),
            nn.MaxPool1d(3, 3),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel_size, channel_size, kernel_sizes[1]),
            nn.MaxPool1d(3, 3),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel_size, channel_size, kernel_sizes[2]),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel_size, channel_size, kernel_sizes[3]),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel_size, channel_size, kernel_sizes[4]),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel_size, channel_size, kernel_sizes[5]),
            nn.MaxPool1d(3, 3),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(channel_size * 26, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_output),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class MultiLabelCharCNNClassifier(NNClassifier):
    def __init__(self,
                 training: TensorDataset,
                 validation: TensorDataset,
                 num_chars: int,
                 alphabet_size: int,
                 num_labels: int,
                 ):
        super(MultiLabelCharCNNClassifier, self).__init__(CharCNN, training, validation,
                                                          {'num_chars': num_chars,
                                                           'alphabet_size': alphabet_size,
                                                           'num_output': num_labels})

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


class MultiClassCharCNNClassifier(NNClassifier):
    def __init__(self,
                 training: TensorDataset,
                 validation: TensorDataset,
                 num_chars: int,
                 alphabet_size: int,
                 num_classes: int,
                 ):
        super(MultiClassCharCNNClassifier, self).__init__(CharCNN, training, validation,
                                                          {'num_chars': num_chars,
                                                           'alphabet_size': alphabet_size,
                                                           'num_output': num_classes})

        self.optim = self._adam
        self.criterion = CrossEntropyLoss()
