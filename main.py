import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import transformers
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

from classifier import *
from classifier.LogisticRegression import *
from classifier.metric import *
from classifier.plugin import *


RAW_DATASET_PATH = Path('hate_speech_mlma/en_dataset_with_stop_words.csv')

data = read_data(RAW_DATASET_PATH)

TRAINED_MODELS_PATH = Path("trained-models")


def train_model(model: Callable[..., Module], fname: str, model_params: Dict[str, Any] = {},
                epochs: int = 100,
                continue_from: int = 0,
                batch_size: int = 100):
    model_path = Path(TRAINED_MODELS_PATH / fname)

    clf = LogisticRegressionClassifier()

    clf.train(epochs,
              batch_size=batch_size,
              plugins=[
                  save_good_models(model_path),
                  calc_train_val_performance(F1Score()),
                  print_train_val_performance(F1Score()),
                  log_train_val_performance(F1Score()),
                  save_training_message(model_path),
                  plot_train_val_performance(model_path, 'Model', F1Score(), show=False,
                                             save=True),
                  elapsed_time(),
                  save_train_val_performance(model_path, F1Score()),
              ],
              start_epoch=continue_from + 1
              )

