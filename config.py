import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SUBMISSION_DIR = os.path.join(BASE_DIR, 'submissions')

TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')

RANDOM_STATE = 42

CATBOOST_PARAMS = {
    'depth': 5,
    'iterations': 150,
    'learning_rate': 0.03,
    'random_seed': RANDOM_STATE,
    'verbose': 0
}

LGBM_PARAMS = {
    'max_depth': 5,
    'n_estimators': 100,
    'learning_rate': 0.03,
    'random_state': RANDOM_STATE,
    'verbose': -1
}

XGB_PARAMS = {
    'max_depth': 3,
    'n_estimators': 100,
    'learning_rate': 0.1,
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss'
}

RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}
