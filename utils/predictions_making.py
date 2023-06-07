'''Predictions making module
'''

from typing import Dict
import tensorflow as tf
from tensorflow import keras


def predict_next(sample: str,
                 model: keras.Sequential,
                 tokenizer: Dict[str, int],
                 vocabulary: Dict[int, str],
                 n_next: int,
                 temperature: float,
                 batch_size: int,
                 word: False) -> str:
    
    if word:
        sample_vector = [tokenizer[word] for word in sample.split()]
    else:
        sample_vector = [tokenizer[char] for char in sample]
        
    predicted = sample_vector
    
    sample_tensor = tf.expand_dims(sample_vector, 0)
    sample_tensor = tf.repeat(sample_tensor, batch_size, axis=0)

    for i in range(n_next):
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