'''Web scrapping module
'''

from typing import List
from bs4 import BeautifulSoup
import requests
import os


def request_url(url: str) -> BeautifulSoup:
    request = requests.get(url)
    soup = BeautifulSoup(request.content, 'html.parser')
    return soup


def get_url_data(url: str) -> List[str]:
    soup = request_url(url)
    scrapped_text = []
    
    h1 = soup.h1.text.strip()
    p = soup.find_all('p')
    
    scrapped_text.append(h1)
    scrapped_text.extend([p_i.text.strip() for p_i in p])
    
    return scrapped_text


def get_data(url: str) -> str:
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


def load_data(url: str, file_name: str, path_dir: str = 'data/') -> str:
    if os.path.isdir(path_dir) == False:
        os.mkdir(path_dir)
        print(f'Created {path_dir} directory')
        
    path_file = f'{path_dir}{file_name}'
    
    try:
        with open(path_file, 'r', encoding='utf-8') as file:
            text = file.read()

        print('Uploaded from', path_file)

    except:
        text = get_data(url)

        with open(path_file, 'w', encoding='utf-8') as file:
            file.write(text)

        print('Saved to', path_file)
        
    return text