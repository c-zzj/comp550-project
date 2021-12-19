from typing import List, Any, Union, Optional, Callable, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset
import torch
from torch import Tensor
import re
import transformers
from transformers import AutoModel, BertTokenizerFast
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer


def read_data(raw_data: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_data)
    return df


def preprocess(data: List[Any], pipeline: List[Callable]) -> TensorDataset:
    """
    Apply list of preprocesses to data
    :param data:
    :param pipeline:
    :return: The preprocessed data. __getitem__(self, idx) should return a tuple (features, labels)
    """
    for step in pipeline:
        data = step(data)
    return data


def ndarray_to_dataset(data: List[np.ndarray]) -> TensorDataset:
    x = torch.from_numpy(data[0]).float()
    y = torch.from_numpy(data[1]).float()
    return TensorDataset(x, y)


def transform_raw_data(data: Any, pipeline: List[Callable]) -> List[np.ndarray]:
    """
    Transformation of raw data to be done before splitting train, val, test sets.
    :param data:
    :param pipeline:
    :return:
    """
    for step in pipeline:
        data = step(data)
    return data


def split_dataset(data: List[np.ndarray], ratios: Tuple[int] = (8, 1, 1), random_seed: Optional[int] = None)\
        -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    :param data:
    :param ratios: (training, validation, test) ratio of training, validation, and test sets
    :return: a tuple of tuples length 3, (train, val, test)
    """
    np.random.seed(random_seed)
    perms = [np.random.permutation(part.shape[0]) for part in data]
    data = [data[i][perms[i]] for i in range(len(data))]
    np.random.seed(None)

    train = []
    val = []
    test = []
    for i in range(len(data)):
        length = data[i].shape[0]
        train_ratio, val_ratio = ratios[0] / sum(ratios), ratios[1] / sum(ratios)
        val_start = int(length * train_ratio)
        test_start = int(length * (train_ratio + val_ratio))
        train.append(data[i][:val_start])
        val.append(data[i][val_start:test_start])
        test.append(data[i][test_start:])

    return train, val, test


def df_to_text_label(data: pd.DataFrame) -> List[np.ndarray]:
    data = data.to_numpy()
    return [data[:,1], data[:,2]]


def transform_label(data: List[np.ndarray]) -> List[np.ndarray]:
    text = data[0]

    labeler = MultiLabelBinarizer()
    labels = [s.split('_') for s in data[1]]
    labels = labeler.fit_transform(labels)
    return [text, labels]


def clean_text(data: List[np.ndarray]) -> List[np.ndarray]:
    def transform(s: str) -> str:
        s = s.replace("@user", "")
        s = s.replace("@URL", "")
        s = s.lower()
        s = re.sub('[^a-z]', ' ', s)
        return s

    text = np.array([transform(s) for s in data[0]])
    labels = data[1]
    return [text, labels]


def GetCharListConverter(num_chars: int = 800,
                         alphabet: Optional[str] = None,
                         include_upper: bool = False) -> Callable:
    letters_lower = 'abcdefghijklmnopqrstuvwxyz'
    letter_upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    numbers = '0123456789'
    symbols = ',;.!?:\'\"/\\|_@#$%^&*~`+-=<>()[]{}'
    if alphabet is None:
        alphabet = letters_lower + numbers + symbols
        if include_upper:
            alphabet += letter_upper
    alphabet = dict(zip(alphabet, range(len(alphabet))))

    def convert_to_charlist(data: List[np.ndarray]) -> List[np.ndarray]:
        def transform(s: str) -> np.ndarray:
            s = s.encode('ascii', 'ignore')
            s = s.decode()
            s = list(s)
            vec = np.zeros((len(alphabet), num_chars))
            for i in range(min(num_chars, len(s))):
                if s[i] in alphabet:
                    vec[alphabet[s[i]]][i] = 1
            return vec

        text = np.array([transform(s) for s in data[0]])
        labels = data[1]
        return [text, labels]

    return convert_to_charlist



_TFIDFVEC = TfidfVectorizer()


def GetTfidfVectorizer(train: bool = False, max_features: int = 6000) -> Callable:
    _TFIDFVEC.max_features = max_features

    def vectorize_tfidf(data: List[np.ndarray]) -> List[np.ndarray]:
        if train:
            _TFIDFVEC.fit(data[0])
        text = _TFIDFVEC.transform(data[0]).toarray()
        labels = data[1]
        return [text, labels]

    return vectorize_tfidf


def to_embedding_bert(text_set: np.ndarray) -> transformers.tokenization_utils_base.BatchEncoding:
    # Load the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # tokenize and encode sequences in the training set
    tokens = tokenizer.batch_encode_plus(
        text_set.tolist(),
        max_length=25,
        padding=True,
        truncation=True,
        return_token_type_ids=False
    )
    return tokens
