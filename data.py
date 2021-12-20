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
from transformers import AutoModel, BertTokenizerFast
import random
from random import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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


def split_dataset(data: List[np.ndarray], ratios: Tuple[int] = (8, 1, 1), random_seed: Optional[int] = None) \
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
    return [data[:, 1], data[:, 2]]


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


def get_only_chars(line):
    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ")
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +', ' ', clean_line)  # delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line


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


def eda(data, alpha_sr, alpha_ri, alpha_rs, p_rd, num_aug):
    df = data[1].values.tolist()
    label = data[0].values.tolist()
    for sentence in df:
        sentence = get_only_chars(sentence)
        words = sentence.split(' ')
        words = [word for word in words if word is not '']
        num_words = len(words)

        augmented_sentences = []
        num_new_per_technique = int(num_aug / 4) + 1

        # sr
        if (alpha_sr > 0):
            n_sr = max(1, int(alpha_sr * num_words))
            for _ in range(num_new_per_technique):
                a_words = synonym_replacement(words, n_sr)
                augmented_sentences.append(' '.join(a_words))

        # ri
        if (alpha_ri > 0):
            n_ri = max(1, int(alpha_ri * num_words))
            for _ in range(num_new_per_technique):
                a_words = random_insertion(words, n_ri)
                augmented_sentences.append(' '.join(a_words))

        # rs
        if (alpha_rs > 0):
            n_rs = max(1, int(alpha_rs * num_words))
            for _ in range(num_new_per_technique):
                a_words = random_swap(words, n_rs)
                augmented_sentences.append(' '.join(a_words))

        # rd
        if (p_rd > 0):
            for _ in range(num_new_per_technique):
                a_words = random_deletion(words, p_rd)
                augmented_sentences.append(' '.join(a_words))

        augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
        shuffle(augmented_sentences)

        # trim so that we have the desired number of augmented sentences
        if num_aug >= 1:
            augmented_sentences = augmented_sentences[:num_aug]
        else:
            keep_prob = num_aug / len(augmented_sentences)
            augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

        # append the original sentence
        for lab in label:
            for sent in augmented_sentences:
                to_append = [lab, sent]
                df_length = len(df)
                df.loc[df_length] = to_append

    return df



def GetWord2VecConverter(length: int = 50):
    # load pretrained embedding
    if not os.path.isfile('word2vec.d2v'):
        model = api.load("word2vec-google-news-300")
        model.save('word2vec.d2v')
    embed_lookup = KeyedVectors.load('word2vec.d2v')

    def convert_to_word2vec(data: List[np.ndarray]) -> List[np.ndarray]:
        def transform(s: str) -> np.ndarray:
            words = s.split()
            vec = [embed_lookup[w] for w in words if w in embed_lookup]
            if len(vec) == 0:
                return np.zeros((length - len(vec), 300))
            if len(vec) < length:
                padding = np.zeros((length - len(vec), 300))
                vec = np.array(vec)
                vec = np.concatenate((padding, vec), axis=0)
            else:
                vec = vec[:length]
            return vec
        text = np.array([transform(s, 50, embed_lookup) for s in data[0]])
        label = data[1]
        return [text, label]

    return convert_to_word2vec

