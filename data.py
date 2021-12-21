from typing import List, Any, Union, Optional, Callable, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset
import torch
from torch import Tensor
import re
import transformers
import nltk
import random
from random import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
from gensim.models import KeyedVectors
import gensim.downloader as api

nltk.download('wordnet')
from nltk.corpus import wordnet


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


def split_dataset(data: List[np.ndarray], ratios: Tuple[int] = (8, 1, 1)) \
        -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    :param data:
    :param ratios: (training, validation, test) ratio of training, validation, and test sets
    :return: a tuple of tuples length 3, (train, val, test)
    """
    perm = np.random.permutation(data[0].shape[0])
    data = [data[i][perm] for i in range(len(data))]

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


def transform_label_multilabel(data: List[np.ndarray]) -> List[np.ndarray]:

    text = data[0]

    labeler = MultiLabelBinarizer()
    labels = [s.split('_') for s in data[1]]
    labels = labeler.fit_transform(labels)
    return [text, labels]


def transform_label_multiclass(data: List[np.ndarray]) -> List[np.ndarray]:
    text = data[0]

    labeler = LabelEncoder()
    labels = [s for s in data[1]]
    labels = labeler.fit_transform(labels)
    return [text, labels]


def GetWordProcessor(type=0) -> Callable[[List[np.ndarray]], List[np.ndarray]]:
    '''
    0 - remove stopwords, 1 - stemming, 2 - lemmatization
    :param type:
    :return:
    '''
    if type == 0:
        stop = stopwords.words('english')

        def transform(s: str) -> str:
            s = s.split(' ')
            s = [w for w in s if w not in stop]
            return ' '.join(s)
    elif type == 1:
        lemmatizer = WordNetLemmatizer()

        def transform(s: str) -> str:
            s = s.split(' ')
            s = [lemmatizer.lemmatize(w) for w in s]
            return ' '.join(s)
    else:
        stemmer = SnowballStemmer('english')

        def transform(s: str) -> str:
            s = s.split(' ')
            s = [stemmer.stem(w) for w in s]
            return ' '.join(s)

    def processor(data: List[np.ndarray]) -> List[np.ndarray]:
        text = np.array([transform(s) for s in data[0]])
        labels = data[1]
        return [text, labels]

    return processor


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


def display_char_length_stats(data: List[np.ndarray]) -> List[np.ndarray]:
    text = data[0]
    lengths = []
    for s in text:
        lengths.append(len(s))

    print(f'Largest length of characters: {max(lengths)}')
    print(f'Mean length of characters: {np.mean(lengths)}')
    print(f'STD length of characters: {np.std(lengths,)}')
    return data


def to_ascii(data: List[np.ndarray]) -> List[np.ndarray]:
    def transform(s: str) -> str:
        s = s.encode('ascii', 'ignore')
        return s.decode()
    text = np.array([transform(s) for s in data[0]])
    labels = data[1]
    return [text, labels]


def GetCharListConverter(num_chars: int = 800,
                         alphabet: Optional[str] = None,
                         to_lower: bool = True) -> Callable:
    letters_lower = 'abcdefghijklmnopqrstuvwxyz'
    letter_upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    numbers = '0123456789'
    symbols = ',;.!?:\'\"/\\|_@#$%^&*~`+-=<>()[]{}'
    if alphabet is None:
        alphabet = letters_lower + numbers + symbols
        if not to_lower:
            alphabet += letter_upper
    alphabet = dict(zip(alphabet, range(len(alphabet))))

    def convert_to_charlist(data: List[np.ndarray]) -> List[np.ndarray]:
        def transform(s: str) -> np.ndarray:
            if to_lower:
                s = s.lower()
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

# region augmentation helpers

def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stopwords.words('english')]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:  # only replace up to n words
            break

    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


def random_deletion(words, p):
    # obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    # randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    # if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    return new_words


def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words


def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words) - 1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words) - 1)
    new_words.insert(random_idx, random_synonym)


def eda(data, percentage_sr, percentage_ri, percentage_rs, percentage_rd, num_aug):
    """
    
    :param data: input data
    :param percentage_sr: percentage of words to be replaced by their synonym
    :param percentage_ri: proportion of random new words to be added in each sentence
    :param percentage_rs: percentage of words to be randomly swapped
    :param percentage_rd: percentage of words to be randomly deleted
    :param num_aug: number of new sentences generated for each sentence
    :return: 
    """
    text = data[0]
    label = data[1]
    new_text = []
    new_label = []
    for i in range(len(text)):
        sentence = text[i]
        words = sentence.split(' ')
        words = [word for word in words if word != '']
        num_words = len(words)

        augmented_sentences = []

        # sr
        if (percentage_sr > 0):
            n_sr = max(1, int(percentage_sr * num_words))
            for _ in range(num_aug):
                a_words = synonym_replacement(words, n_sr)
                augmented_sentences.append(' '.join(a_words))

        # ri
        if (percentage_ri > 0):
            n_ri = max(1, int(percentage_ri * num_words))
            for _ in range(num_aug):
                a_words = random_insertion(words, n_ri)
                augmented_sentences.append(' '.join(a_words))

        # rs
        if (percentage_rs > 0):
            n_rs = max(1, int(percentage_rs * num_words))
            for _ in range(num_aug):
                a_words = random_swap(words, n_rs)
                augmented_sentences.append(' '.join(a_words))

        # rd
        if (percentage_rd > 0):
            for _ in range(num_aug):
                a_words = random_deletion(words, percentage_rd)
                augmented_sentences.append(' '.join(a_words))

        shuffle(augmented_sentences)

        # trim so that we have the desired number of augmented sentences
        if num_aug >= 1:
            augmented_sentences = augmented_sentences[:num_aug]
        else:
            keep_prob = num_aug / len(augmented_sentences)
            augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

        # add to new dataset
        new_text.append(text[i])
        new_label.append(label[i])
        new_text += augmented_sentences
        new_label += [label[j] for j in range(len(augmented_sentences))]

    data = [np.array(new_text), np.array(new_label)]
    perm = np.random.permutation(data[0].shape[0])
    data = [data[i][perm] for i in range(len(data))]
    return data

# endregion

def GetAugmenter(
                 percentage_sr: float = 0,
                 percentage_ri: float = 0,
                 percentage_rs: float = 0,
                 percentage_rd: float = 0,
                 num_aug: int = 1,
                 target_path: Optional[Union[str, Path]] = None,):
    """
    :param percentage_sr: percentage of words to be replaced by their synonym
    :param percentage_ri: proportion of random new words to be added in each sentence
    :param percentage_rs: percentage of words to be randomly swapped
    :param percentage_rd: percentage of words to be randomly deleted
    :param num_aug: number of new sentences generated for each sentence for each method
    :param target_path: target path to save augmented data
    :return:
    """
    def augment_data(data: List[np.ndarray]) -> List[np.ndarray]:
        if target_path is not None and Path(target_path).exists():
            loaded = np.load(target_path)
            augmented = [loaded['text'], loaded['label']]
        else:
            augmented = eda(data, percentage_sr, percentage_ri, percentage_rs, percentage_rd, num_aug)
            if target_path is not None:
                if not target_path.parent.exists():
                    target_path.parent.mkdir(parents=True)
                np.savez_compressed(target_path, text=augmented[0], label=augmented[1])
        return augmented
    return augment_data


def GetWord2VecConverter(length: int = 50):
    # load pretrained embedding
    if not os.path.isfile('word2vec.d2v'):
        model = api.load("word2vec-google-news-300")
        model.save('word2vec.d2v')
    model = KeyedVectors.load('word2vec.d2v')

    def convert_to_word2vec(data: List[np.ndarray]) -> List[np.ndarray]:
        def transform(s: str) -> np.ndarray:
            words = s.split()
            vec = [model[w] for w in words if w in model]
            if len(vec) == 0:
                return np.zeros((300, length))
            if len(vec) < length:
                padding = np.zeros((300, (length - len(vec))))
                vec = np.array(vec).transpose()
                vec = np.concatenate((vec, padding), axis=1)
            else:
                vec = np.array(vec[:length]).transpose()
            return vec

        text = np.array([transform(s) for s in data[0]])
        label = data[1]
        return [text, label]

    return convert_to_word2vec
