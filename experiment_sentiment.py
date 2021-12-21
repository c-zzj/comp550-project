from classifier.logit_model import *
from classifier.char_cnn import *
from classifier.word2vec_clf import *
from classifier.metric import *
from classifier.plugin import *
from util import *

RAW_DATASET_PATH = Path('sentiment140.csv')

TRAINED_MODELS_PATH = Path("trained-models-sentiment")


def df_to_text_label(data: pd.DataFrame) -> List[np.ndarray]:
    #data = data.to_numpy()
    return [data[:, 0], data[:, 1]]


def ndarray_to_dataset(data: List[np.ndarray]) -> TensorDataset:
    x = torch.from_numpy(data[0]).float()
    y = torch.from_numpy(data[1]).long()
    return TensorDataset(x, y)


def GetDatasetSizeLimiter(max_entries: int):
    def limit_dataset_size(data: List[np.ndarray]) -> List[np.ndarray]:
        x, y = data[0], data[1]
        length = x.shape[0]
        if length <= max_entries or max_entries == -1:
            return data
        perm = np.random.permutation(length)
        x, y = x[perm[:max_entries]], y[perm[:max_entries]]
        return [x, y]
    return limit_dataset_size


def transform_label_multiclass(data: List[np.ndarray]) -> List[np.ndarray]:
    text = data[0]

    labeler = LabelEncoder()
    labels = [s for s in data[1]]
    labels = labeler.fit_transform(labels)
    print(np.unique(labels, return_counts=True))
    return [text, labels]


def run_baseline_sentiment(epochs: int = 2, get_test: bool = False, max_data_size: int = -1):
    #raw = pd.read_csv(RAW_DATASET_PATH, encoding='latin')
    neg_raw = np.loadtxt("trained-models-sentiment/rt-polarity.neg", dtype=str, delimiter="\n", encoding="latin-1")
    pos_raw = np.loadtxt("trained-models-sentiment/rt-polarity.pos", dtype=str, delimiter="\n", encoding="latin-1")
    neg = np.concatenate((neg_raw[:,None], np.zeros((neg_raw.shape[0], 1))),axis=1)
    pos = np.concatenate((pos_raw[:, None], np.ones((pos_raw.shape[0], 1))), axis=1)
    raw = np.concatenate((neg, pos), axis=0)

    p1 = [
        df_to_text_label,
        GetDatasetSizeLimiter(max_data_size),
        clean_text,
        transform_label_multiclass,
    ]
    transformed_data = transform_raw_data(raw, p1)
    split = split_dataset(transformed_data)
    train, val, test = split[0], split[1], split[2]

    # ros = RandomOverSampler(random_state=0)
    # train[0] = np.reshape(train[0], (-1, 1))
    # train[0], train[1] = ros.fit_resample(train[0], train[1])
    # train[0] = np.reshape(train[0], (-1,))

    p2_train = [
        GetTfidfVectorizer(train=True, max_features=8000),
        ndarray_to_dataset
    ]
    p2 = [
        GetTfidfVectorizer(max_features=8000),
        ndarray_to_dataset
    ]
    train, val, test = preprocess(train, p2_train), preprocess(val, p2), preprocess(test, p2)
    clf = MultiClassLogisticRegressionClassifier(training=train, validation=val, in_size=8000, num_classes=2)
    model_path = Path(TRAINED_MODELS_PATH / 'logit')
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


def run_char_cnn_sentiment(preprocessing: List[int] = [],
                           augmentation: Dict[str, Any] = None,
                           epochs: int = 2,
                           get_test: bool = False,
                           max_data_size: int = -1):
    # raw = pd.read_csv(RAW_DATASET_PATH, encoding='latin')
    neg_raw = np.loadtxt("trained-models-sentiment/rt-polarity.neg", dtype=str, delimiter="\n", encoding="latin-1")
    pos_raw = np.loadtxt("trained-models-sentiment/rt-polarity.pos", dtype=str, delimiter="\n", encoding="latin-1")
    neg = np.concatenate((neg_raw[:, None], np.zeros((neg_raw.shape[0], 1))), axis=1)
    pos = np.concatenate((pos_raw[:, None], np.ones((pos_raw.shape[0], 1))), axis=1)
    raw = np.concatenate((neg, pos), axis=0)
    p1 = [df_to_text_label,
          GetDatasetSizeLimiter(max_data_size),
          transform_label_multiclass,
          to_ascii]
    transformed_data = transform_raw_data(raw, p1)

    split = split_dataset(transformed_data)
    train, val, test = split[0], split[1], split[2]

    # ros = RandomOverSampler(random_state=0)
    # train[0] = np.reshape(train[0], (-1, 1))
    # train[0], train[1] = ros.fit_resample(train[0], train[1])
    # train[0] = np.reshape(train[0], (-1,))

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

    clf = MultiClassCharCNNClassifier(training=train, validation=val, num_chars=200, alphabet_size=68, num_classes=2)
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


def run_word_cnn_sentiment(preprocessing: List[int] = [],
                           augmentation: Dict[str, Any] = None,
                           epochs: int = 2,
                           get_test: bool = False,
                           max_data_size: int = -1):
    # raw = pd.read_csv(RAW_DATASET_PATH, encoding='latin')
    neg_raw = np.loadtxt("trained-models-sentiment/rt-polarity.neg", dtype=str, delimiter="\n", encoding="latin-1")
    pos_raw = np.loadtxt("trained-models-sentiment/rt-polarity.pos", dtype=str, delimiter="\n", encoding="latin-1")
    neg = np.concatenate((neg_raw[:, None], np.zeros((neg_raw.shape[0], 1))), axis=1)
    pos = np.concatenate((pos_raw[:, None], np.ones((pos_raw.shape[0], 1))), axis=1)
    raw = np.concatenate((neg, pos), axis=0)
    p1 = [df_to_text_label,
          GetDatasetSizeLimiter(max_data_size),
          clean_text,
          transform_label_multiclass,
          ]
    transformed_data = transform_raw_data(raw, p1)
    split = split_dataset(transformed_data)
    train, val, test = split[0], split[1], split[2]

    # ros = RandomOverSampler(random_state=0)
    # train[0] = np.reshape(train[0], (-1, 1))
    # train[0], train[1] = ros.fit_resample(train[0], train[1])
    # train[0] = np.reshape(train[0], (-1,))

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

    clf = MultiClassWordCNNClassifier(training=train, validation=val, num_classes=2)
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


def run_word_lr_sentiment(preprocessing: List[int] = [],
                           augmentation: Dict[str, Any] = None,
                           epochs: int = 2,
                           get_test: bool = False,
                           max_data_size: int = -1):
    # raw = pd.read_csv(RAW_DATASET_PATH, encoding='latin')
    neg_raw = np.loadtxt("trained-models-sentiment/rt-polarity.neg", dtype=str, delimiter="\n", encoding="latin-1")
    pos_raw = np.loadtxt("trained-models-sentiment/rt-polarity.pos", dtype=str, delimiter="\n", encoding="latin-1")
    neg = np.concatenate((neg_raw[:, None], np.zeros((neg_raw.shape[0], 1))), axis=1)
    pos = np.concatenate((pos_raw[:, None], np.ones((pos_raw.shape[0], 1))), axis=1)
    raw = np.concatenate((neg, pos), axis=0)
    p1 = [df_to_text_label,
          GetDatasetSizeLimiter(max_data_size),
          clean_text,
          transform_label_multiclass,
          ]
    transformed_data = transform_raw_data(raw, p1)
    split = split_dataset(transformed_data)
    train, val, test = split[0], split[1], split[2]

    # ros = RandomOverSampler(random_state=0)
    # train[0] = np.reshape(train[0], (-1, 1))
    # train[0], train[1] = ros.fit_resample(train[0], train[1])
    # train[0] = np.reshape(train[0], (-1,))

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

    clf = MultiClassWordLogisticRegressionClassifier(training=train, validation=val, num_classes=2)
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
