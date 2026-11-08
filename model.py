from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from config import CATBOOST_PARAMS, LGBM_PARAMS, XGB_PARAMS, RF_PARAMS, MODEL_TYPE, VOTING_TYPE

def build_ensemble_model():
    """Создаёт выбранную модель на основе настроек config.py"""

    if MODEL_TYPE == 'catboost':
        return CatBoostClassifier(**CATBOOST_PARAMS)

    elif MODEL_TYPE == 'lightgbm':
        return LGBMClassifier(**LGBM_PARAMS)

    elif MODEL_TYPE == 'xgboost':
        return XGBClassifier(**XGB_PARAMS)

    elif MODEL_TYPE == 'random_forest':
        return RandomForestClassifier(**RF_PARAMS)

    elif MODEL_TYPE == 'voting':
        cb = CatBoostClassifier(**CATBOOST_PARAMS)
        lgb = LGBMClassifier(**LGBM_PARAMS)
        xgb = XGBClassifier(**XGB_PARAMS)
        rf = RandomForestClassifier(**RF_PARAMS)

        return VotingClassifier(
            estimators=[
                ('cat', cb),
                ('lgb', lgb),
                ('xgb', xgb),
                ('rf', rf)
            ],
            voting=VOTING_TYPE
        )

    raise ValueError(f'Неизвестный тип модели: {MODEL_TYPE}')
