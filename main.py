import pandas as pd
import os
from preprocessing import train_clean, test_clean
from lightgbm import LGBMClassifier

X_train = train_clean.drop(columns=['Survived', 'PassengerId'], errors='ignore')
y_train = train_clean['Survived']
X_test = test_clean[X_train.columns]
orig_test = pd.read_csv('data/test.csv')

lgbm_model = LGBMClassifier(learning_rate=0.1, max_depth=3, n_estimators = 100, random_state=42, verbose=-1)

lgbm_model.fit(X_train, y_train)
preds = lgbm_model.predict(X_test)

preds = preds.astype(int)
passenger_id = orig_test['PassengerId'].astype(int)

os.makedirs('submissions', exist_ok = True)
submission = pd.DataFrame({
    'PassengerId': passenger_id,
    'Survived': preds
})

submission.to_csv('submissions/lgbm_model_submission.csv', index = False, sep = ',', encoding = 'utf-8')