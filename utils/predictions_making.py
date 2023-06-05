'''Predictions making module
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
from keras.preprocessing import text


def predict_next_character(sample: str,
                           model: tf.keras.Sequential,
                           tokenizer: dict,
                           vocabulary: dict,
                           n_chars: int,
                           temperature: float,
                           batch_size: int) -> str:
            
    sample_vector = [tokenizer[char] for char in sample]
    predicted = sample_vector
    
    sample_tensor = tf.expand_dims(sample_vector, 0)
    sample_tensor = tf.repeat(sample_tensor, batch_size, axis=0)

    for i in range(n_chars):
        pred = model(sample_tensor)
        
        pred = pred[0].numpy() / temperature
        pred = tf.random.categorical(pred, num_samples=1)[-1, 0].numpy()
        
        predicted.append(pred)
        
        sample_tensor = predicted[-99:]
        sample_tensor = tf.expand_dims([pred], 0)
        
        sample_tensor = tf.repeat(sample_tensor, batch_size, axis=0)
        
    pred_seq = [vocabulary[i] for i in predicted]
    generated = ''.join(pred_seq)
    
    return generated
    
    
def predict_next_word(sample: str,
                      model: tf.keras.Sequential,
                      tokenizer: text.Tokenizer,
                      n_words: int,
                      max_len: int) -> str:
    
    generated = sample
    
    for _ in range(n_words):
        token_list = tokenizer.texts_to_sequences([generated])[0]
        token_list = sequence.pad_sequences([token_list], maxlen=max_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted = np.argmax(predicted)
        
        next_word = ''
        
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                next_word = word
                break
                
        generated += ' ' + next_word
    
    return generated