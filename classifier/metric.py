from classifier import *
from sklearn.metrics import f1_score


class Accuracy:
    def __call__(self, clf: Classifier, data_loader: DataLoader, ) -> float:
        total = 0
        correct = 0
        for i, data in enumerate(data_loader, 0):
            x = data[0].to(clf.device)
            pred = clf.predict(x)
            true = data[1].to(clf.device)
            total += data_loader.batch_size
            correct += (pred == true).sum().item()
        return correct / total

    def __str__(self):
        return 'Accuracy'


class F1Score:
    def __init__(self, average: str = 'macro'):
        self.average = average

    def __call__(self, clf: Classifier, data_loader: DataLoader, ) -> float:
        true = []
        pred = []
        for i, data in enumerate(data_loader, 0):
            x = data[0].to(clf.device)
            pred += clf.predict(x).tolist()
            true += data[1].to(clf.device).tolist()
        return f1_score(true, pred, average=self.average)

    def __str__(self):
        return f'F1-{self.average}'
