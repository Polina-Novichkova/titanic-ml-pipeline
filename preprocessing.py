import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data(filename):
    paths = [
        os.path.join(BASE_DIR, '..', 'data', filename),
        os.path.join(BASE_DIR, 'data', filename),
        os.path.join(BASE_DIR, filename),
        filename
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(f"Файл {filename} не найден ни по одному из путей!")

train = load_data('train.csv')
test = load_data('test.csv')

dt = pd.concat([train, test], ignore_index = True)

titles = {
    'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
    'Dr': 'Other', 'Rev': 'Other', 'Col': 'Other', 
    'Major': 'Other', 'Capt': 'Other', 'Countess': 'Other',
    'Lady': 'Other', 'Jonkheer': 'Other', 'Don': 'Other', 
    'Sir': 'Other', 'Dona': 'Other'
}

dt['Title'] = dt['Name'].str.extract(r' ([A-Za-z]+)\.', expand = False)
dt['Title'] = dt['Title'].replace(titles)

dt['Embarked'] = dt['Embarked'].fillna('S')
dt['Age'] = dt['Age'].fillna(dt['Age'].median())
dt['Fare'] = dt['Fare'].fillna(dt['Fare'].median())

dt['Family_Size'] = dt['SibSp'] + dt['Parch'] + 1
dt['Alone'] = (dt['Family_Size'] == 1).astype(int)
dt['Fare_log'] = np.log1p(dt['Fare'])
    
dt['Sex'] = dt['Sex'].map({'female': 0, 'male': 1})
dt = pd.get_dummies(dt, columns = ['Embarked', 'Title'], drop_first = True, dtype = int)

drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
dt = dt.drop(columns = [col for col in drop_cols if col in dt.columns])

train_clean = dt[dt['Survived'].notnull()].copy()
test_clean = dt[dt['Survived'].isnull()].drop(columns = ['Survived']).copy()