import os
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from preprocessing import load_data, test_clean, train_clean
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

X_train = train_clean.drop(columns=['Survived'], errors='ignore')
y_train = train_clean['Survived']
X_test = test_clean[X_train.columns]

cb = CatBoostClassifier(
    depth=5, iterations=150, learning_rate=0.03, random_seed=42, verbose=0
)
lgb = LGBMClassifier(
    max_depth=5,
    n_estimators=100,
    learning_rate=0.03,
    random_state=42,
    verbose=-1,
)
xgb = XGBClassifier(
    max_depth=3,
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
)

rf = RandomForestClassifier(
    n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
)

voting_hard_model = VotingClassifier(
    estimators=[('cat', cb), ('lgb', lgb), ('xgb', xgb), ('rf', rf)],
    voting='hard',
)
voting_hard_model.fit(X_train, y_train)

orig_test = load_data('test.csv')
submission = pd.DataFrame(
    {'PassengerId': orig_test['PassengerId'], 'Survived': voting_hard_model.predict(X_test)}
)

os.makedirs('submissions', exist_ok=True)
submission.to_csv('submissions/voting_hard_submission.csv', index=False)