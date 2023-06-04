import numpy as np

import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
from typing import Tuple, List

import requests
import random
import os

from keras import Sequential
from keras.callbacks import History

import tensorflow as tf
from tensorflow.data import Dataset


def request_url(url: str) -> BeautifulSoup:
    request = requests.get(url)
    soup = BeautifulSoup(request.content, 'html.parser')
    return soup


def get_url_data(url: str) -> list:
    soup = request_url(url)
    scrapped_text = []
    
    h1 = soup.h1.text.strip()
    p = soup.find_all('p')
    
    scrapped_text.append(h1)
    scrapped_text.extend([p_i.text.strip() for p_i in p])
    
    return scrapped_text


def get_data(url: str) -> list:
    soup = request_url(url)
    text = []
    
    text.extend([
        soup.h1.text.strip() + '.',
        soup.h2.text.strip() + '.',
        soup.article.p.text.strip()
    ])
    
    url_chapters = [link.get('href') for link in soup.find_all('a', class_='link')]
    
    for url in url_chapters:
        scrapped_text = get_url_data(url)
        text.extend(scrapped_text)
        
    text = ' '.join(text).lower()
    
    return text


def train_test_split(text: list, test_size: float) -> Tuple[List[str], List[str]]:
    random.shuffle(text)
    threshold = int((1 - test_size) * len(text))
    
    train = text[:threshold]
    test = text[threshold:]
    
    return train, test


def split_into_train_valid_test(text: str,
                                path_train: str,
                                path_valid: str,
                                path_test: str = None,
                                test_size: float = 0.25) -> None:
    
    if os.path.isfile(path_train) == False and os.path.isfile(path_valid) == False:
        
        text_split = [s.strip() for s in text.split('.')]
        
        train_text, valid_text = train_test_split(text_split, test_size)
        test_text = None
        
        if path_test != None:
            valid_text, test_text = train_test_split(valid_text, test_size)
            test_text = ' '.join(test_text)
        
        train_valid_test = [(path_train, train_text), (path_valid, valid_text), (path_test, test_text)]
        
        for (path, text) in train_valid_test:
            if path != None:
                with open(path, 'w', encoding='utf-8') as file:
                    file.write(' '.join(text))
        
        print('Splitted into:', len(train_text), 'train,', len(valid_text), 'valid and', 
              len(test_text) if test_text != None else 0, 'test')
    
    else:
        print('Files already exist')

    
def text_vector_sample(text: str or list, 
                       vector: np.array, 
                       end: int = 100) -> None:
    
    print('Исходный текст:\n', text[:end], '\n')
    print('Векторное представление:\n', vector[:end])
    
    
def print_single_batch(dataset: Dataset,
                       vocabulary: np.array,
                       word: bool = False) -> None:
    
    for vector_single, target_single in dataset.take(1):
        print('Векторное представление:')
        print(vector_single.numpy())
        print(target_single.numpy())
        
        if word == True:
            vector_single_text = ' '.join(list(vocabulary[word.numpy()] for word in vector_single))
            target_single_text = ' '.join(list(vocabulary[word.numpy()] for word in target_single))
        else:
            vector_single_text = ''.join(vocabulary[vector_single])
            target_single_text = ''.join(vocabulary[target_single])

        print('\nПеревод в текст:')
        print(repr(vector_single_text))
        print(repr(target_single_text))
        
        
def print_single_dim(data) -> None:
    for vector_single, target_single in data.take(1):
        print('Размерность входящей последовательности:', vector_single.numpy().shape)
        print('Размерность целевой последовательности:', target_single.numpy().shape)
        
        
def prepare_dataset(seq: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    input_vector = seq[:-1]
    target_vector = seq[1:]
    return input_vector, target_vector


def test_single_model(model, data, vocabulary: dict) -> None:
    for vector_single, target_single in data.take(1):
        vector_single_pred = model(vector_single)
        ids = tf.random.categorical(vector_single_pred[0], num_samples=1)
        id_pred = ids[0][-1].numpy()

        print('Размерность целевой последовательности:', target_single.numpy().shape)
        print('Размерность предсказанной последовательности:', vector_single_pred.shape)
        print('Размерность тензора с 1 индексом классов', ids.shape)
        print(f'Индекс класса: {id_pred} ({vocabulary[id_pred]})')
        
        
def plot_performance(history: History, title: str) -> None:
    font_s = 12
    plt.figure(figsize=(6,5))
    
    loss = history.history['loss']
    
    plt.plot(loss, '+-r')
    
    plt.title(f'{title}\n', size=font_s+4)
    
    plt.xlabel('Epoch', size=font_s,)
    plt.ylabel('Loss', size=font_s)
    
    plt.xticks(range(len(loss)))
    plt.grid()
    plt.show()
    
    
def predict_next(sample: str,
                 model: Sequential,
                 tokenizer: dict,
                 vocabulary: dict,
                 n_char_word: int,
                 temperature: float,
                 batch_size: int,
                 word: bool = False) -> str:
    
    sample = sample.split() if word else sample
        
    sample_vector = [tokenizer[s] for s in sample]
    predicted = sample_vector
    
    sample_tensor = tf.expand_dims(sample_vector, 0)
    sample_tensor = tf.repeat(sample_tensor, batch_size, axis=0)

    for i in range(n_char_word):
        pred = model(sample_tensor)
        
        pred = pred[0].numpy() / temperature
        pred = tf.random.categorical(pred, num_samples=1)[-1, 0].numpy()
        
        predicted.append(pred)
        
        sample_tensor = predicted[-99:]
        sample_tensor = tf.expand_dims([pred], 0)
        
        sample_tensor = tf.repeat(sample_tensor, batch_size, axis=0)
        
    pred_seq = [vocabulary[i] for i in predicted]
    generated = ' '.join(pred_seq) if word else ''.join(pred_seq)
    
    return generated