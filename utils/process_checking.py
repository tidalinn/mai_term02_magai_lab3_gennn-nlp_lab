import numpy as np
from typing import List
import random
from tensorflow import keras


def print_total(text: str) -> None:
    print('Всего слов:', len(text.split()))
    
    if len(text.split('.')) != 1:
        print('Всего предложений:', len(text.split('.')))
        
        
def print_tokenizer_vocabulary(tokenizer: keras.preprocessing.text.Tokenizer,
                               index: int = 10) -> List[tuple]:
    vocab_pairs = list(zip(
        list(tokenizer.word_index.keys())[:index], 
        list(tokenizer.word_index.values())[:index]
    ))
    return vocab_pairs

    
def print_init_vector(text: str, vector: np.array, end: int = 100) -> None:
    print('Исходный текст:\n', text[:end], '\n')
    print('Векторное представление:\n', vector[:end])
    
    
'''
def print_single_batch(dataset: Dataset,
                       vocabulary: Dict[str, int]) -> None:
    
    random_id = random.choice(range(len(dataset)))
                        
    for features, target in dataset.take(random_id):
        print('Векторное представление:')
        print('Признаки:\n', features.numpy())
        print('Целевой признак:\n', target.numpy())

        print('\nПеревод в текст:')
        print('Признаки:\n', ''.join(vocabulary[features]))
        print('Целевой признак:\n', ''.join(vocabulary[target]))
'''
                              
                              
def print_single_element(features: np.ndarray, target: np.ndarray, vocabulary: dict) -> None:
    random_id = random.choice(range(len(features)))
    
    features = features[random_id]
    target = target[random_id]
    
    features_words = ' '.join(list(vocabulary[word] for word in features if word != 0))
    target_id = np.argmax(target)
    
    print(f'Признаки {features.shape}:\n{features}\n')
    print(f'Перевод в текст:\n{features_words}\n')
    
    print(f'Целевой признак {target.shape}:\n{target_id}\n')
    print(f'Перевод в текст:\n{vocabulary[target_id]}\n')
        
        
def print_single_element_dimension(data) -> None:
    random_id = random.choice(range(len(dataset)))
    
    for features, target in data.take(random_id):
        print('Размерность признаков:', features.numpy().shape)
        print('Размерность целевого признака:', target.numpy().shape)


'''
def print_single_test(model: Sequential, data, vocabulary: dict) -> None:
    random_id = random.choice(range(len(dataset)))
    
    for features, target in data.take(random_id):
        pred = model(features)
        ids = tf.random.categorical(vector_single_pred[0], num_samples=1)
        id_pred = ids[0][-1].numpy()

        print('Размерность признаков:', features.numpy().shape)
        print('Размерность предсказаний:', pred.shape)
        print('Размерность тензора с 1 индексом классов', ids.shape)
        print(f'Предсказанный класс: {id_pred} ({vocabulary[id_pred]})')
'''