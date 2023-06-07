import numpy as np
from typing import List, Dict, Mapping
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.data import Dataset


def print_total(text: str) -> None:
    print('Всего слов:', len(text.split()))
    
    if len(text.split('.')) != 1:
        print('Всего предложений:', len(text.split('.')))
        

'''LSTM bidirectional | RNN char
'''
def print_tokenizer_vocabulary(vocabulary: Dict[str, int], index: int = 10) -> List[tuple]:    
    vocab_pairs = list(zip(
        list(vocabulary.keys())[:index], 
        list(vocabulary.values())[:index]
    ))
    return vocab_pairs


'''LSTM word | LSTM char
'''
def print_init_vector(text: str, vector: np.array, end: int = 100) -> None:
    print('Исходный текст:\n', text[:end], '\n')
    print('Векторное представление:\n', vector[:end])
    
    
'''RNN char
'''
def print_single_batch(dataset: Dataset, vocabulary: Dict[str, int]) -> None:
    random_id = random.choice(range(len(dataset)))
                        
    for features, target in dataset.take(random_id):
        print('Векторное представление:')
        print('Признаки:\n', features.numpy())
        print('Целевой признак:\n', target.numpy())

        print('\nПеревод в текст:')
        print('Признаки:\n', ''.join(vocabulary[features]))
        print('Целевой признак:\n', ''.join(vocabulary[target]))
                              

'''LSTM bidirectional | GPT architecture | LSTM word | LSTM char
'''
def print_single_element(features: np.ndarray, 
                         target: np.ndarray, 
                         vocabulary: dict or Mapping, 
                         take_random: bool = True,
                         char: bool = False) -> None:
    
    space = '' if char else ' '
    translation_dict = lambda x: space.join(list(vocabulary[word] for word in x if word != 0))
    translation_func = lambda x: ' '.join(list(vocabulary(word) for word in x))
    
    if take_random:
        random_id = random.choice(range(len(features)))
        features = features[random_id]
        target = target[random_id]
    
    try:
        features_words = translation_dict(features)
    except:
        features_words = translation_func(features)
    
    try:
        target_count = sum(target > 0)
    except:
        target_count = len(target)
    
    if target_count == 1:
        target_id = np.argmax(target)
        target_words = vocabulary[target_id]
    else:
        try:
            target_words = translation_dict(target)
        except:
            target_words = translation_func(target)

    print(f'Признаки {features.shape}:\n{features}\n')
    print(f'Перевод в текст:\n{features_words}\n')

    print(f'Целевой признак {target.shape}:\n{target_id if target_count == 1 else target}\n')
    print(f'Перевод в текст:\n{target_words}\n')
        
        
'''GPT architecture
'''
def print_single_dimension(features: np.ndarray, target: np.ndarray) -> None:
    print('Размерность признаков:', features.shape)
    print('Размерность целевого признака:', target.shape)
    
    
'''GPT architecture
'''
def print_max_min_len(text: list) -> None:
    print('Максимальная длина строки:', len(max(text)))
    print('Минимальная длина строки:', len(min(text)))


'''LSTM word
'''
def print_single_test(model, features: np.ndarray, target: np.ndarray, vocabulary: dict) -> None:
    pred = model(features)
    ids = tf.random.categorical(pred[0], num_samples=1)
    id_pred = ids[0][-1].numpy()

    print('Размерность признаков:', features.numpy().shape)
    print('Размерность предсказаний:', pred.shape)
    print('Размерность тензора с 1 индексом классов', ids.shape)
    print(f'Предсказанный класс: {id_pred} ({vocabulary[id_pred]})')