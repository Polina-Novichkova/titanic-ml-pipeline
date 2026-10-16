import os
import pandas as pd
from preprocessing import prepare_features
from model import build_ensemble_model
from config import SUBMISSION_DIR

def main():
    print("1. Подготовка данных и фичей...")
    X_train, y_train, X_test, passenger_ids = prepare_features()

    print("2. Обучение ансамбля моделей...")
    model = build_ensemble_model()
    model.fit(X_train, y_train)

    print("3. Предсказание результатов на тестовой выборке...")
    predictions = model.predict(X_test)

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(SUBMISSION_DIR, 'voting_hard_submission.csv')

    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })
    submission.to_csv(submission_path, index=False)

if __name__ == '__main__':
    main()
