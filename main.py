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
                batch_size: int = 100,
                plugins: Optional[List[TrainingPlugin]] = None):

    model_path = Path(TRAINED_MODELS_PATH / fname)

    if not isinstance(classifier, Classifier):
        classifier = classifier(**clf_params)

    if plugins is None:
        plugins = [
                  CalcTrainValPerformance(F1Score()),
                  SaveGoodModels(model_path, F1Score()),
                  PrintTrainValPerformance(F1Score()),
                  LogTrainValPerformance(F1Score()),
                  SaveTrainingMessage(model_path),
                  PlotTrainValPerformance(model_path, 'Model', F1Score(), show=False,
                                          save=True),
                  SaveTrainValPerformance(model_path, F1Score()),
                  ElapsedTime(),
              ]
    classifier.train(epochs,
              batch_size=batch_size,
              plugins=plugins,
              start_epoch=continue_from + 1
              )


def run_baseline():
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
    plugins = [
        PrintTrainValPerformance(F1Score('micro')),
        PrintTrainValPerformance(F1Score('macro')),
        PrintTrainValPerformance(Accuracy()),
        ElapsedTime(),
              ]
    train_model(logit, fname='logit', epochs=10, plugins=plugins)


def run_char_cnn():
    raw = read_data(RAW_DATASET_PATH)
    p1 = [
        df_to_text_label,
        GetCharListConverter(),
        transform_label,
    ]
    transformed_data = transform_raw_data(raw, p1)
    split = split_dataset(transformed_data)
    train, val, test = split[0], split[1], split[2]
    p2 = [
        ndarray_to_dataset
    ]
    train, val = preprocess(train, p2), preprocess(val, p2)
    logit = ChainLogisticRegressionClassifier(training=train, validation=val, in_size=8000, num_labels=6)
    train_model(logit, fname='logit', epochs=10)


if __name__ == '__main__':
    run_baseline()
