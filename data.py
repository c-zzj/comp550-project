from typing import List, Any, Union, Optional, Callable, Tuple
from pathlib import Path
import pandas as pd
import numpy as np


def read_data(raw_data: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_data)
    return df


def preprocess(data: Any, pipeline: List[Callable]) -> Any:
    """
    Apply list of preprocesses to data
    :param data:
    :param pipeline:
    :return:
    """
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
