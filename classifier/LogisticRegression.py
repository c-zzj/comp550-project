from classifier import *


class LogisticRegression(Module):
    def __init__(self, in_size, out_size):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(in_size, out_size)

    def forward(self, x):
        output = self.linear(x)
        return output


class LogisticRegressionClassifier(NNClassifier):
    def __init__(self,
                 training: Dataset,
                 validation: Dataset,
                 in_size,
                 out_size,
                 ):
        super(LogisticRegressionClassifier, self).__init__(LogisticRegression, training, validation,
                                                           {'in_size': in_size, 'out_size': out_size})

    def _pred(self, x: Tensor):
        """
        argmax prediction - use the highest value as prediction
        :param x:
        :return:
        """
        self.network.eval()
        with torch.no_grad():
            pred = self.network(x.float())
        self.network.train()

        indices = torch.argmax(pred, dim=1)
        return indices