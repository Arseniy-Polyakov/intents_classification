import json_repair
from tqdm import tqdm
from preprocessing import text_preprocessing

def parsing_file(file_name: str) -> str:
  """
  Функция для парсинга JSON файлов c интентами: 
  1. Чтение JSON файла
  2. Извлечение интентов и меток
  3. Предобработка интентов и меток
  4. Составление списка словарей формата: {<название интента>: <текст интента>}
  """
  whole_intents = []
  with open(file_name, "rt", encoding="utf-8") as file:
    text = file.read()
    text_json = json_repair.loads(text)
    for i in tqdm(range(len(text_json))):
      try:
        for sentence in text_json[i]["phrases"]:
          item = {text_json[i]["path"]: text_preprocessing(sentence["text"]).strip()}
          if len(" ".join(list(item.values()))) != 0:
            whole_intents.append(item)
      except:
        i+=1
  return whole_intents

start = 1
final = 86
for i in tqdm(range(start, final)):
  result = parsing_file(f"datasets\intents_raw\intents_{i}.json")
  with open(f"datasets\intents_preprocessed\intents_preprocessed_{i}.json", "wt", encoding="utf-8") as file:
    file.writelines(str(result))