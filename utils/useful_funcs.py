from bs4 import BeautifulSoup
from typing import Tuple, List
import requests
import random
import os


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
        
        
def print_max_min_string(text: str) -> None:
    print('Максимальная длина строки:', len(max(text.split('.'), key=len).split()))
    print('Минимальная длина строки:', len(min(text.split('.'), key=len).split()))