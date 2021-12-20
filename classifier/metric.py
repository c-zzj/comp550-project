from classifier import *
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


class Accuracy:
    def __call__(self, clf: Classifier, data_loader: DataLoader, ) -> float:
        true = []
        pred = []
        for i, data in enumerate(data_loader, 0):
            x = data[0].to(clf.device)
            pred += clf.predict(x).tolist()
            true += data[1].to(clf.device).tolist()
        return accuracy_score(true, pred)

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
