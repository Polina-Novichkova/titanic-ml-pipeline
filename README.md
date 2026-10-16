# Titanic Machine Learning Project

Решение задачи Titanic на Kaggle с использованием ансамблирования моделей CatBoost, LightGBM, XGBoost и Random Forest.

## Структура проекта

* `data/` — директория для исходных данных (`train.csv`, `test.csv`)
* `submissions/` — директория для итоговых файлов с предсказаниями
* `notebooks/` — Jupyter Notebooks с первичным анализом (EDA) и экспериментами (baseline, ensembles и commits)
* `config.py` — гиперпараметры моделей, SEED и пути к файлам
* `preprocessing.py` — модуль загрузки, очистки данных и генерации признаков (Feature Engineering)
* `model.py` — модуль сборки ансамбля моделей
* `main.py` — главный скрипт для запуска полного цикла обучения и формирования submission
* `requirements.txt` — список зависимостей проекта

## Инструкция по запуску

1. Клонируйте репозиторий и перейдите в папку проекта:
   ```bash
   git clone https://github.com/Polina-Novichkova/titanic-ml-pipeline
   cd titanic-ml-pipeline
   ```
   
2. Установите необходимые зависимости:
pip install -r requirements.txt

3. Подготовьте данные:
Поместите файлы train.csv и test.csv из соревнования Titanic на Kaggle в папку data/.

4.Запустите обучение и генерацию предсказаний:
python main.py

5.Итоговый файл с предсказаниями автоматически сохранится в папке submissions/.



