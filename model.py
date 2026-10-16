from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from config import CATBOOST_PARAMS, LGBM_PARAMS, XGB_PARAMS, RF_PARAMS

def build_ensemble_model():
    cb = CatBoostClassifier(**CATBOOST_PARAMS)
    lgb = LGBMClassifier(**LGBM_PARAMS)
    xgb = XGBClassifier(**XGB_PARAMS)
    rf = RandomForestClassifier(**RF_PARAMS)

    ensemble = VotingClassifier(
        estimators=[
            ('cat', cb),
            ('lgb', lgb),
            ('xgb', xgb),
            ('rf', rf)
        ],
        voting='hard'
    )
    return ensemble