from classifier import *
from classifier.plugin import *
from classifier.metric import *
import random


def train_model(classifier: Union[Callable[..., Classifier], Classifier],
                model_path: Path,
                clf_params: Optional[Dict[str, Any]] = None,
                epochs: int = 100,
                continue_from: int = 0,
                batch_size: int = 100,
                plugins: Optional[List[TrainingPlugin]] = None) -> Classifier:

    if not isinstance(classifier, Classifier):
        classifier = classifier(**clf_params)

    if plugins is None:
        plugins = [
            SaveModel(model_path),
            CalcTrainValPerformance(F1Score('micro')),
            PrintTrainValPerformance(F1Score('micro')),
            LogTrainValPerformance(F1Score('micro')),

            CalcTrainValPerformance(F1Score()),
            PrintTrainValPerformance(F1Score()),
            LogTrainValPerformance(F1Score()),

            CalcTrainValPerformance(Accuracy()),
            PrintTrainValPerformance(Accuracy()),
            LogTrainValPerformance(Accuracy()),

            SaveTrainingMessage(model_path),
            ElapsedTime(),
              ]
    classifier.train(epochs,
                     batch_size=batch_size,
                     plugins=plugins,
                     start_epoch=continue_from + 1
                     )
    return classifier

def random_seed_global(seed: Optional[int]):
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)