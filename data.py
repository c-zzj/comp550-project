from typing import List, Any, Union, Optional, Callable, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset
from torch import Tensor
import re
import transformers
from transformers import AutoModel, BertTokenizerFast
import random


def read_data(raw_data: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_data)
    return df


def preprocess(data: Any, pipeline: List[Callable]) -> TensorDataset:
    """
    Apply list of preprocesses to data
    :param data:
    :param pipeline:
    :return: The preprocessed data. __getitem__(self, idx) should return a tuple (features, labels)
    """
    for step in pipeline:
        data = step(data)
    return data


def transform_raw_data(data: Any, pipeline: List[Callable]) -> np.ndarray:
    for step in pipeline:
        data = step(data)
    return data


def split_dataset(data: np.ndarray, ratios: Tuple[int] = (8, 1, 1), random_seed: Optional[int] = None) -> Tuple:
    """
    :param data:
    :param ratios: (training, validation, test) ratio of training, validation, and test sets
    :return: a 3-tuple, (train, val, test)
    """
    data = np.array(data)
    np.random.seed(random_seed)
    np.random.shuffle(data)
    np.random.seed(None)
    length = data.shape[0]

    train_ratio, val_ratio = ratios[0] / sum(ratios), ratios[1] / sum(ratios)
    val_start = int(length * train_ratio)
    test_start = int(length * (train_ratio + val_ratio))
    train, val, test = data[:val_start], data[val_start:test_start], data[test_start:]
    return train, val, test


def prepare_dataset(df):
    """

    :param df: a dataframe
    :return: a 2-tuple, (text, labels)
    """
    labels = [transform_label(s) for s in df['sentiment']]
    text = [clean(s) for s in df['tweet']]
    return text, labels


def transform_label(s: str) -> str:
    """
    :param s: a string
    :return: a numpy array of labels
    """
    # d = {'abusive': 0, 'hateful': 1, 'offensive': 2, 'disrespectful': 3, 'fearful': 4}
    label = np.zeros(5)
    if 'abusive' in s:
        label[0] = 1
    if 'fearful' in s:
        label[4] = 1

    # randomly assign a label
    if 'hateful' and 'offensive' and 'disrespectful' in s:
        i = random.randint(0, 2)
        indices = [1, 2, 3]
        label[indices[i]] = 1
    elif 'hateful' and 'offensive' in s:
        i = random.randint(0, 1)
        indices = [1, 2]
        label[indices[i]] = 1
    elif 'hateful' and 'disrespectful' in s:
        i = random.randint(0, 1)
        indices = [1, 3]
        label[indices[i]] = 1
    elif 'offensive' and 'disrespectful' in s:
        i = random.randint(0, 1)
        indices = [2, 3]
        label[indices[i]] = 1
    return label



def clean(s: str) -> str:
    """
    :param s: a string
    :return:
    """
    s = s.replace("@user", "")
    s = s.replace("@URL", "")
    s = s.lower()
    s = re.sub('[^a-z]', ' ', s)
    return s


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










