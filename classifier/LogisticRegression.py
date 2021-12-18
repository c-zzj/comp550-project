from classifier import *
from torch.nn import BCEWithLogitsLoss
from classifier.metric import *

class LogisticRegression(Module):
    def __init__(self, in_size):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(in_size, 1)

    def forward(self, x):
        output = self.linear(x)
        return output


class LogisticRegressionClassifier(NNClassifier):
    def __init__(self,
                 training: TensorDataset,
                 validation: TensorDataset,
                 in_size,
                 ):
        super(LogisticRegressionClassifier, self).__init__(LogisticRegression, training, validation,
                                                           {'in_size': in_size})

        self.optim = SGD(self.network.parameters(), lr=1e-3, momentum=0.99)
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
            pred[:,0] = (torch.sigmoid(pred[:,0]) > 0.5).to(torch.int)
            pred = (torch.sigmoid(pred) > 0.5).to(torch.int)
        self.network.train()

        return pred


class ChainLogisticRegressionClassifier(Classifier):
    def __init__(self,
                 training: TensorDataset,
                 validation: TensorDataset,
                 in_size: int,
                 num_labels: int,
                 ):
        super(ChainLogisticRegressionClassifier, self).__init__(training, validation)

        self.in_size = in_size
        self.num_labels = num_labels
        self.logits = []
        for i in range(self.num_labels):
            train_y = torch.reshape(self.training.tensors[1][:, i], (-1,1))
            train = TensorDataset(self.training.tensors[0], train_y)
            val_y = torch.reshape(self.validation.tensors[1][:,i], (-1,1))
            val = TensorDataset(self.validation.tensors[0], val_y)
            self.logits.append(LogisticRegressionClassifier(train, val, self.in_size))

    def train(self, *args, **kwargs) -> None:
        for i in range(self.num_labels):
            print(f"Training {i}-th logit model")
            self.logits[i].train(*args, **kwargs)
        print(f"Training finished")
        print(f"TRAINING PERFORMANCE:")
        print(f"Accuracy: {self.train_performance(Accuracy(), proportion=1)}")
        print(f"F1-micro: {self.train_performance(F1Score('micro'), proportion=1)}")
        print(f"F1-macro: {self.train_performance(F1Score('macro'), proportion=1)}")
        print(f"Validation PERFORMANCE:")
        print(f"Accuracy: {self.val_performance(Accuracy())}")
        print(f"F1-micro: {self.val_performance(F1Score('micro'))}")
        print(f"F1-macro: {self.val_performance(F1Score('macro'))}")

    def _pred(self, x):
        result = Tensor([]).to(self.device)
        for i in range(self.num_labels):
            result = torch.cat((result, self.logits[i].predict(x)), dim=1)
        return result



