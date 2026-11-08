import os
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from preprocessing import prepare_features, load_data
from model import build_ensemble_model
from config import SUBMISSION_DIR, RANDOM_STATE, MODEL_TYPE

def main():
    """Запускает кросс-валидацию, обучение и создание submission"""
    train = load_data('train.csv')
    y = train['Survived'].astype(int)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    baseline_scores = []
    
    print('Кросс-валидация:')
    for fold, (train_idx, val_idx) in enumerate(cv.split(train, y), start = 1):
        train_fold = train.iloc[train_idx].copy()
        val_fold = train.iloc[val_idx].copy() 
   
        X_train, y_train, X_val, _ = prepare_features(train_fold, val_fold)
    
        model = build_ensemble_model()
        model.fit(X_train, y_train)

        # Оценка основной модели
        predictions = model.predict(X_val)
        score = accuracy_score(y.iloc[val_idx], predictions)
        scores.append(score)

        # Оценка baseline-модели
        baseline = DummyClassifier(strategy='most_frequent')
        baseline.fit(X_train, y_train)

        baseline_predictions = baseline.predict(X_val)
        baseline_score = accuracy_score(y.iloc[val_idx], baseline_predictions)
        baseline_scores.append(baseline_score)

        print(f'Fold {fold}:')
        print(f'  Основная модель: Accuracy = {score:.4f}')
        print(f'  Baseline:        Accuracy = {baseline_score:.4f}')
    
    mean_score = sum(scores) / len(scores)
    mean_baseline = sum(baseline_scores) / len(baseline_scores)

    print(f'\nСредняя Accuracy основной модели: {mean_score:.4f}')
    print(f'Средняя Accuracy baseline:        {mean_baseline:.4f}')
    print(f'Превышение baseline:               {mean_score - mean_baseline:.4f}')
    
    X_train, y_train, X_test, passenger_ids = prepare_features()

    model = build_ensemble_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })

    submission_path = os.path.join(
        SUBMISSION_DIR,
        f'{MODEL_TYPE}_submission.csv'
    )

    submission.to_csv(submission_path, index=False)

    print(f'Submission сохранён: {submission_path}')


if __name__ == '__main__':
    main()