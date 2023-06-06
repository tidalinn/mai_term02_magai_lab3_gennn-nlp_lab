'''Data preprocessing module
'''

from typing import Tuple, List
import random
import os
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.data import TextLineDataset
from keras.preprocessing import text
from keras_nlp.tokenizers import word_piece_tokenizer
from keras_nlp.layers import start_end_packer


'''LSTM bidirectional
'''
def split_into_sentences(text: str, regex: str = '[^а-яА-ЯёЁ0-9 ,-]') -> List[str]:
    sentences = [re.sub(regex, '', s).strip() for s in text.split('.')]
    sentences = list(filter(None, sentences))
    return sentences


def train_test_split(text: list, test_size: float) -> Tuple[List[str], List[str]]:
    random.shuffle(text)
    threshold = int((1 - test_size) * len(text))
    
    train = text[:threshold]
    test = text[threshold:]
    
    return train, test


'''GPT architecture
'''
def train_valid_test_split_save(text: str,
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
        

def get_features_target(seq: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    features = seq[:-1]
    target = seq[1:]
    return features, target


'''LSTM bidirectional
'''
def get_features(text: List[str], tokenizer: text.Tokenizer) -> List[int]:
    features = []

    for line in text:
        token_list = tokenizer.texts_to_sequences([line])[0]

        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            features.append(n_gram_sequence)
            
    return features