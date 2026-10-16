import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data(filename):
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


train = load_data('train.csv')
test = load_data('test.csv')
dt = pd.concat([train, test], ignore_index=True)

dt['Title'] = dt['Name'].str.extract(r'([A-Za-z]+)\.', expand=False)
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
    'Rev': 'Other',
}
dt['Title'] = dt['Title'].replace(titles)

#Пропуски и базовые фичи
dt['Age'] = dt.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
dt['Fare'] = dt['Fare'].fillna(dt['Fare'].median())
dt['Family_Size'] = dt['SibSp'] + dt['Parch'] + 1
dt['Alone'] = (dt['Family_Size'] == 1).astype(int)

ticket_counts = dt['Ticket'].value_counts()
dt['Group_Size'] = dt['Ticket'].map(ticket_counts)
dt['Fare_Per_Person'] = dt['Fare'] / dt['Group_Size']
dt['Fare_log'] = np.log1p(dt['Fare'])

dt['Deck'] = dt['Cabin'].str[0].fillna('Unknown')
dt['Deck'] = dt['Deck'].replace(['T'], 'Unknown')

#Групповая выживаемость
dt['Surname'] = dt['Name'].apply(lambda x: x.split(',')[0].strip())
dt['Family_ID'] = dt['Surname'] + '_' + dt['Fare'].astype(str)
dt['Is_Woman_Child'] = (
    (dt['Sex'] == 'female') | (dt['Title'] == 'Master')
).astype(int)
dt['Group_Survival'] = 0.5

#По билетам
for ticket, group in dt.groupby('Ticket'):
  if len(group) > 1:
    train_wc = group[
        (group['Survived'].notnull()) & (group['Is_Woman_Child'] == 1)
    ]
    if len(train_wc) > 0:
      dt.loc[dt['Ticket'] == ticket, 'Group_Survival'] = train_wc[
          'Survived'
      ].mean()

#По семьям
for fam_id, group in dt.groupby('Family_ID'):
  if len(group) > 1:
    train_wc = group[
        (group['Survived'].notnull()) & (group['Is_Woman_Child'] == 1)
    ]
    if len(train_wc) > 0:
      mask = (dt['Family_ID'] == fam_id) & (dt['Group_Survival'] == 0.5)
      dt.loc[mask, 'Group_Survival'] = train_wc['Survived'].mean()

# 4. Кодирование
dt['Sex'] = dt['Sex'].map({'female': 0, 'male': 1}).astype(int)
dt['Embarked'] = dt['Embarked'].fillna('S')
dt = pd.get_dummies(
    dt, columns=['Embarked', 'Title', 'Deck', 'Pclass'], drop_first=True, dtype=int
)

drop_cols = [
    'PassengerId',
    'Name',
    'Ticket',
    'Cabin',
    'Fare',
    'SibSp',
    'Parch',
    'Surname',
    'Family_ID',
    'Is_Woman_Child',
]
dt = dt.drop(columns=[col for col in drop_cols if col in dt.columns])

train_clean = dt[dt['Survived'].notnull()].copy()
test_clean = dt[dt['Survived'].isnull()].drop(columns=['Survived']).copy()
train_clean['Survived'] = train_clean['Survived'].astype(int)
