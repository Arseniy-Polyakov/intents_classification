import os
import re
import getpass
import requests
from natasha import (
    Segmenter,
    MorphVocab,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    
    PER,
    NamesExtractor,

    Doc
)

segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

names_extractor = NamesExtractor(morph_vocab)

AUTHORIZATION_TOKEN = getpass.getpass()

def text_parsing(i: int) -> str:
    """
    Функция для парсинга интентов для дальнейшего извлечения именованных сущностей для обезличивания
    """
    datasets = os.listdir("datasets\intents_preprocessed")

    full_path = "datasets\intents_preprocessed\\" + datasets[i]  
    with open(full_path, "rt", encoding="utf-8") as file:
        text_splitted = file.read().split(": ")
        text_final = " ".join([re.sub(r"[^\w\s-]", "", text_splitted[i]) for i in range(len(text_splitted))])
    return text_final

def access_token() -> str:
    """
    Функция для получения access token для доступа к хэндлерам моделей семейста GigaChat
    """
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    payload={
        'scope': 'GIGACHAT_API_PERS'
    }
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': '5ec13f5f-65fa-488a-8610-bad1d46b7f82',
        'Authorization': f'Basic {AUTHORIZATION_TOKEN}'
    }
    response = requests.post(url, headers=headers, data=payload, verify=False)
    access_token = response.json()["access_token"]
    return access_token

def ner_gigachat(text: str, ACCESS_TOKEN: str) -> dict:
    """
    Функция для метода POST, излечение именованных сущностей от модели GigaChat
    """
    prompt = f"""Ты аналитик данных.
            Извлеки из данного текста {text} названия компаний, например: 
            Альфа страхование, Алокабанк, Сбер, Яндекс, Ультрамар и т.д.
            Если в тексте нет названий компаний, то напиши в ответе 'Нет сущностей'"""

    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json", 
        "Authorization": f"Bearer {ACCESS_TOKEN}"
    }
    body = {
        "model": "GigaChat-2",
        "messages": [{
                    "role": "user",
                    "content": prompt
                }],
        "n": 1,
        "stream": False,
        "max_tokens": 512,
        "repetition_penalty": 1,
        "update_interval": 0
    }
    response = requests.post(url, headers=headers, json=body, verify=False)
    if response.status_code != 200:
        return "Ошибка в подключении REST API"
    else:
        response_json = response.json()
        message = response_json["choices"][0]["message"]["content"]
        return message

def ner_natasha():
    """
    Функция для извлечения именованных сущностей с помощью библиотеки natasha
    """
    for i in range(70, 77):
        doc = Doc(text_parsing(i))
        doc.segment(segmenter)
        doc.tag_ner(ner_tagger)
        print(i+1)
        print([span.text for span in doc.spans])
        print()