# Titanic Machine Learning Project

Решение задачи Titanic на Kaggle с использованием ансамблирования моделей CatBoost, LightGBM, XGBoost и Random Forest.

## Структура проекта

* `data/` — директория для исходных данных (`train.csv`, `test.csv`)
* `submissions/` — директория для итоговых файлов с предсказаниями
* `notebooks/` — Jupyter Notebooks с первичным анализом (EDA) и экспериментами (baseline и ensembles)
* `config.py` — гиперпараметры моделей, RANDOM_STATE и пути к файлам
* `preprocessing.py` — модуль загрузки, очистки данных и генерации признаков (Feature Engineering)
* `model.py` — создание выбранной модели или ансамбля
* `main.py` — главный скрипт для запуска полного цикла обучения и формирования submission
* `requirements.txt` — список зависимостей проекта

## Используемые модели

В проекте реализованы следующие алгоритмы:
- CatBoost
- LightGBM
- XGBoost
- Random Forest
- VotingClassifier

Для выбора модели необходимо изменить параметр MODEL_TYPE в файле config.py:

Примеры значений:
- catboost — CatBoost.
- lightgbm — LightGBM.
- xgboost — XGBoost.
- random_forest — Random Forest.
- voting — ансамбль VotingClassifier.
- Для ансамбля также доступна настройка типа голосования через параметр VOTING_TYPE (hard или soft)

## Предварительная обработка данных

В модуле preprocessing.py выполняются:
- обработка пропущенных значений;
- извлечение обращений пассажиров из имени (Title);
- создание признаков на основе семейных данных;
- кодирование категориальных признаков;
- подготовка признаков для обучения моделей.

## Результаты кросс-валидации

Для оценки качества моделей используется StratifiedKFold с 5 разбиениями, перемешиванием данных и фиксированным `RANDOM_STATE = 42`

В качестве основной модели в текущем эксперименте используется VotingClassifier с типом голосования `hard`

Для сравнения используется baseline на основе `DummyClassifier` со стратегией `most_frequent`. Эта модель всегда предсказывает наиболее частый класс обучающей выборки

## Сравнение моделей

Baseline (DummyClassifier): 0.6162
VotingClassifier (hard): 0.8361

Основная модель превысила результат baseline на 0.2200

## Результаты по разбиениям

| Разбиение | VotingClassifier | Baseline |
|-----------|------------------|----------|
| Fold 1    | 0.8436           | 0.6145   |
| Fold 2    | 0.8483           | 0.6180   |
| Fold 3    | 0.8371           | 0.6180   |
| Fold 4    | 0.8315           | 0.6180   |
| Fold 5    | 0.8202           | 0.6124   |
| Среднее   | 0.8361           | 0.6162   |

Таким образом, VotingClassifier показывает более высокое качество по сравнению с простейшей baseline-моделью на всех пяти разбиениях

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

4. Запустите обучение и генерацию предсказаний:
python main.py

5. Итоговый файл с предсказаниями автоматически сохранится в папке submissions/