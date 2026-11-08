import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data(filename):
    """Загружает CSV-файл из доступной директории проекта"""
    paths = [
        os.path.join(BASE_DIR, '..', 'data', filename),
        os.path.join(BASE_DIR, 'data', filename),
        os.path.join(BASE_DIR, filename),
        filename,
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(f'Файл {filename} не найден!')

def prepare_features(train=None, test=None):
    """Подготавливает признаки для обучения и предсказания модели"""
    if train is None:
        train = load_data('train.csv')
    if test is None:
        test = load_data('test.csv')
    train = train.copy()
    test = test.copy()
    
    y_train = train['Survived'].astype(int)
    train = train.drop(columns=['Survived'])
    test = test.drop(columns=['Survived'], errors='ignore')
    test_passenger_ids = test['PassengerId']
    
    titles = {
        'Mlle': 'Miss',
        'Ms': 'Miss',
        'Mme': 'Mrs',
        'Lady': 'Mrs',
        'Countess': 'Mrs',
        'Dona': 'Mrs',
        'Jonkheer': 'Other',
        'Don': 'Other',
        'Sir': 'Mr',
        'Capt': 'Other',
        'Col': 'Other',
        'Major': 'Other',
        'Dr': 'Other',
        'Rev': 'Other'
    }
    
    train['Title'] = train['Name'].str.extract(r'([A-Za-z]+)\.', expand=False)
    test['Title'] = test['Name'].str.extract(r'([A-Za-z]+)\.', expand=False)

    train['Title'] = train['Title'].replace(titles)
    test['Title'] = test['Title'].replace(titles)
    
    dt = pd.concat([train, test], ignore_index=True)

    # Пропуски и базовые фичи
    age_med = train.groupby('Title')['Age'].median()
    age_median = train['Age'].median()
    fare_med = train['Fare'].median()
    emb_mode = train['Embarked'].mode()[0]
    
    dt['Age'] = dt['Age'].fillna(dt['Title'].map(age_med)).fillna(age_median)
    dt['Fare'] = dt['Fare'].fillna(fare_med)
    dt['Embarked'] = dt['Embarked'].fillna(emb_mode)
    
    dt['Family_Size'] = dt['SibSp'] + dt['Parch'] + 1
    dt['Alone'] = (dt['Family_Size'] == 1).astype(int)

    ticket_counts = train['Ticket'].value_counts()
    dt['Group_Size'] = dt['Ticket'].map(ticket_counts).fillna(1)
    dt['Fare_Per_Person'] = dt['Fare'] / dt['Group_Size']
    dt['Fare_log'] = np.log1p(dt['Fare'])

    dt['Deck'] = dt['Cabin'].str[0].fillna('Unknown')
    dt['Deck'] = dt['Deck'].replace(['T'], 'Unknown')

    dt['Sex'] = dt['Sex'].map({'female': 0, 'male': 1}).astype(int)
    dt = pd.get_dummies(dt, columns=['Embarked', 'Title', 'Deck', 'Pclass'], drop_first=True, dtype=int)

    drop_cols = [
        'PassengerId',
        'Name',
        'Ticket',
        'Cabin',
        'Fare',
        'SibSp',
        'Parch'
    ]
    dt = dt.drop(columns=[col for col in drop_cols if col in dt.columns])

    train_clean = dt.iloc[:len(train)].copy()
    test_clean = dt.iloc[len(train):].copy()
    
    X_train = train_clean
    X_test = test_clean[X_train.columns]

    return X_train, y_train, X_test, test_passenger_ids
