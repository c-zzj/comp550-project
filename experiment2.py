from classifier.logit_model import *
from classifier.char_cnn import *
from classifier.metric import *
from classifier.plugin import *
from util import *

RAW_DATASET_PATH = Path('implicit-hate-corpus/implicit_hate_v1_stg1_posts.tsv')

TRAINED_MODELS_PATH = Path("trained-models-d2")


def df_to_text_label_(data: pd.DataFrame) -> List[np.ndarray]:
    data = data.to_numpy()
    return [data[:,0], data[:,1]]


def ndarray_to_dataset(data: List[np.ndarray]) -> TensorDataset:
    x = torch.from_numpy(data[0]).float()
    y = torch.from_numpy(data[1]).long()
    return TensorDataset(x, y)


def run_baseline_2():
    raw = pd.read_csv(RAW_DATASET_PATH, sep='\t')
    p1 = [
        df_to_text_label_,
        clean_text,
        transform_label_multiclass,
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
    logit = MultiClassLogisticRegressionClassifier(train, val, 8000, 3)
    plugins = [
        PrintTrainValPerformance(F1Score('micro')),
        PrintTrainValPerformance(F1Score('macro')),
        PrintTrainValPerformance(Accuracy()),
        ElapsedTime(),
              ]

    model_path = Path(TRAINED_MODELS_PATH / 'logit')
    train_model(logit, model_path=model_path, epochs=5, plugins=plugins)


def run_char_cnn_2():
    raw = pd.read_csv(RAW_DATASET_PATH, sep='\t')
    p1 = [
        df_to_text_label_,
        GetWordProcessor(2),
        transform_label_multiclass,
    ]
    transformed_data = transform_raw_data(raw, p1)
    split = split_dataset(transformed_data)
    train, val, test = split[0], split[1], split[2]
    p2 = [
        GetAugmenter(),
        GetCharListConverter(num_chars=800, ),
        ndarray_to_dataset
    ]
    train, val = preprocess(train, p2), preprocess(val, p2)

    charcnn = MultiClassCharCNNClassifier(training=train, validation=val, num_chars=800, alphabet_size=68, num_classes=3)

    model_path = Path(TRAINED_MODELS_PATH / 'char-cnn')
    plugins = [
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
    train_model(charcnn, model_path=model_path, epochs=2, plugins=plugins)