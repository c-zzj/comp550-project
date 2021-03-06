from classifier.logit_model import *
from classifier.char_cnn import *
from classifier.word2vec_clf import *
from util import *
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

RAW_DATASET_PATH = Path('HASOC.tsv')

TRAINED_MODELS_PATH = Path("trained-models-d2")


def df_to_text_label(data: pd.DataFrame) -> List[np.ndarray]:
    data = data.to_numpy()
    return [data[:, 1], data[:, 2]]


def ndarray_to_dataset(data: List[np.ndarray]) -> TensorDataset:
    x = torch.from_numpy(data[0]).float()
    y = torch.from_numpy(data[1]).float().reshape((-1,1))
    return TensorDataset(x, y)


def run_baseline_2(epochs: int = 2):
    raw = pd.read_csv(RAW_DATASET_PATH, sep='\t')
    p1 = [
        df_to_text_label,
        clean_text,
        transform_label_multiclass,
    ]
    transformed_data = transform_raw_data(raw, p1)
    split = split_dataset(transformed_data)
    train, val, test = split[0], split[1], split[2]

    ros = RandomOverSampler(random_state=0)
    train[0] = np.reshape(train[0], (-1, 1))
    train[0], train[1] = ros.fit_resample(train[0], train[1])
    train[0] = np.reshape(train[0], (-1,))

    p2_train = [
        GetTfidfVectorizer(train=True, max_features=8000),
        ndarray_to_dataset
    ]
    p2 = [
        GetTfidfVectorizer(max_features=8000),
        ndarray_to_dataset
    ]
    train, val, test = preprocess(train, p2_train), preprocess(val, p2), preprocess(test, p2)
    logit = LogisticRegressionClassifier(training=train, validation=val, in_size=8000)
    model_path = Path(TRAINED_MODELS_PATH / 'logit')
    clf = train_model(logit, model_path=model_path, epochs=epochs)
    micro = clf.evaluate(test, F1Score('micro'))
    macro = clf.evaluate(test, F1Score())
    acc = clf.evaluate(test, Accuracy())
    print("TEST RESULTS:")
    print(f"F1-micro: {micro}")
    print(f"F1-macro: {macro}")
    print(f"Accuracy: {acc}")


def run_char_cnn_2(preprocessing: List[int] = [],
                   augmentation: Dict[str, Any] = None,
                   epochs: int = 2,
                   get_test: bool = False
                   ):
    raw = pd.read_csv(RAW_DATASET_PATH, sep='\t')
    p1 = [df_to_text_label,
          transform_label_multiclass,
          to_ascii]
    transformed_data = transform_raw_data(raw, p1)

    split = split_dataset(transformed_data)
    train, val, test = split[0], split[1], split[2]

    ros = RandomOverSampler(random_state=0)
    train[0] = np.reshape(train[0], (-1, 1))
    train[0], train[1] = ros.fit_resample(train[0], train[1])
    train[0] = np.reshape(train[0], (-1,))

    p2_train = []
    p2 = []
    if augmentation is not None:
        p2_train.insert(0, GetAugmenter(**augmentation))
    for i in preprocessing:
        p2_train.append(GetWordProcessor(i))
        p2.append(GetWordProcessor(i))

    p2_train += [display_char_length_stats, GetCharListConverter(num_chars=800), ndarray_to_dataset]
    p2 += [GetCharListConverter(num_chars=800), ndarray_to_dataset]
    train, val, test = preprocess(train, p2_train), preprocess(val, p2), preprocess(test, p2)

    clf = MultiLabelCharCNNClassifier(training=train, validation=val, num_chars=800, alphabet_size=68, num_labels=1)
    model_path = Path(TRAINED_MODELS_PATH / 'char-cnn')
    if get_test:
        clf.load_network(model_path, epochs)
    else:
        clf = train_model(clf, model_path=model_path, epochs=epochs)

    micro = clf.evaluate(test, F1Score('micro'))
    macro = clf.evaluate(test, F1Score())
    acc = clf.evaluate(test, Accuracy())
    print("TEST RESULTS:")
    print(f"F1-micro: {micro}")
    print(f"F1-macro: {macro}")
    print(f"Accuracy: {acc}")


def run_word_cnn_2(preprocessing: List[int] = [],
                   augmentation: Dict[str, Any] = None,
                   epochs: int = 2,
                   get_test: bool = False):
    raw = pd.read_csv(RAW_DATASET_PATH, sep='\t')
    p1 = [
        df_to_text_label,
        clean_text,
        transform_label_multiclass,
    ]
    transformed_data = transform_raw_data(raw, p1)
    split = split_dataset(transformed_data)
    train, val, test = split[0], split[1], split[2]

    ros = RandomOverSampler(random_state=0)
    train[0] = np.reshape(train[0], (-1, 1))
    train[0], train[1] = ros.fit_resample(train[0], train[1])
    train[0] = np.reshape(train[0], (-1,))

    p2_train = []
    p2 = []
    if augmentation is not None:
        p2_train.insert(0, GetAugmenter(**augmentation))
    for i in preprocessing:
        p2_train.append(GetWordProcessor(i))
        p2.append(GetWordProcessor(i))

    p2_train += [GetWord2VecConverter(length=50), ndarray_to_dataset]
    p2 += [GetWord2VecConverter(length=50), ndarray_to_dataset]
    train, val, test = preprocess(train, p2_train), preprocess(val, p2), preprocess(test, p2)

    clf = MultiLabelWordCNNClassifier(training=train, validation=val, num_labels=1)
    model_path = Path(TRAINED_MODELS_PATH / 'word-cnn')
    if get_test:
        clf.load_network(model_path, epochs)
    else:
        clf = train_model(clf, model_path=model_path, epochs=epochs)

    micro = clf.evaluate(test, F1Score('micro'))
    macro = clf.evaluate(test, F1Score())
    acc = clf.evaluate(test, Accuracy())
    print("TEST RESULTS:")
    print(f"F1-micro: {micro}")
    print(f"F1-macro: {macro}")
    print(f"Accuracy: {acc}")


