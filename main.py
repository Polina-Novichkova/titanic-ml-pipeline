from catboost import CatBoostClassifier
import pandas as pd
from preprocessing import train_clean, test_clean

orig_test = pd.read_csv('data/test.csv')

X_train = train_clean.drop(columns = ['Survived', 'PassengerId'], errors = 'ignore')
y_train = train_clean['Survived']
X_test = test_clean[X_train.columns]

model = CatBoostClassifier(iterations = 200, learning_rate = 0.05, depth = 4, verbose = 0, random_state = 42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': orig_test['PassengerId'],
    'Survived': predictions
})

submission['Survived'] = submission['Survived'].astype(int)

submission.to_csv('submission_baseline.csv', index = False)
