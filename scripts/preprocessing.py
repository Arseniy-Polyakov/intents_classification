import re
import nltk
import pymorphy3
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = stopwords.words("russian")

def text_preprocessing(text: str) -> str:
  """
  Функция для предобработки текста: 
  1. Приведение к нижнему регистру
  2. Удаление пробельных символов с крайних сторон
  3. Токенизация
  4. Удаление стоп-слов
  5. Лемматизация
  6. Удаление некириллических и непробельных символов
  """
  lemmatizer = pymorphy3.MorphAnalyzer()
  text_lower = text.lower().strip().split()
  text_without_stopwords = " ".join([lemmatizer.parse(word)[0].normal_form for word in text_lower if word not in stop_words])
  text_cleaned = re.sub(r"[^а-яё\-\s]", "", text_without_stopwords)
  return text_cleaned