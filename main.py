from classifier.LogisticRegression import *
from classifier.metric import *
from classifier.plugin import *

RAW_DATASET_PATH = Path('hate_speech_mlma/en_dataset_with_stop_words.csv')

data = read_data(RAW_DATASET_PATH)

TRAINED_MODELS_PATH = Path("trained-models")


def train_model(classifier: Union[Callable[..., Classifier], Classifier],
                fname: str,
                clf_params: Optional[Dict[str, Any]] = None,
                epochs: int = 100,
                continue_from: int = 0,
                batch_size: int = 100):

    model_path = Path(TRAINED_MODELS_PATH / fname)

    if not isinstance(classifier, Classifier):
        classifier = classifier(**clf_params)

    classifier.train(epochs,
              batch_size=batch_size,
              plugins=[
                  calc_train_val_performance(F1Score()),
                  SaveGoodModels(model_path, F1Score()),
                  PrintTrainValPerformance(F1Score()),
                  LogTrainValPerformance(F1Score()),
                  SaveTrainingMessage(model_path),
                  PlotTrainValPerformance(model_path, 'Model', F1Score(), show=False,
                                          save=True),
                  SaveTrainValPerformance(model_path, F1Score()),
                  ElapsedTime(),
              ],
              start_epoch=continue_from + 1
              )


if __name__ == '__main__':
    raw = read_data(RAW_DATASET_PATH)
    p1 = [
        df_to_text_label,
        clean_text,
        transform_label,
    ]
    transformed_data = transform_raw_data(raw, p1)

    split = split_dataset(transformed_data)
    train, val, test = split[0], split[1], split[2]
    p2_train = [
        GetTfidfVectorizer(train=True, max_features=8000),
        ndarray_to_dataset
    ]
    p2 = [
        GetTfidfVectorizer(max_features=8000),
        ndarray_to_dataset
    ]
    train = preprocess(train, p2_train)
    val = preprocess(val, p2)
    logit = ChainLogisticRegressionClassifier(training=train, validation=val, in_size=8000, num_labels=6)

    train_model(logit, fname='logit', epochs=20)
