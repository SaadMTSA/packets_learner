from xgboost.sklearn import XGBClassifier
from .base import BaseModel
from skopt import BayesSearchCV

class XGBoostModel(BaseModel):
    def __init__(self, random_state, fast=False):
        if fast:
            self.model = XGBClassifier(random_state=random_state)
        else:
            self.model = BayesSearchCV(
                XGBClassifier(random_state=random_state),
                {
                    'learning_rate': (0.01, 1.0, 'log-uniform'),
                    'min_child_weight': (0, 10),
                    'max_depth': (0, 50),
                    'max_delta_step': (0, 20),
                    'subsample': (0.01, 1.0, 'uniform'),
                    'colsample_bytree': (0.01, 1.0, 'uniform'),
                    'colsample_bylevel': (0.01, 1.0, 'uniform'),
                    'reg_lambda': (1e-4, 1000, 'log-uniform'),
                    'reg_alpha': (1e-4, 1.0, 'log-uniform'),
                    'gamma': (1e-4, 0.5, 'log-uniform'),
                    'min_child_weight': (0, 5),
                    'n_estimators': (50, 200),
                    'scale_pos_weight': (1e-4, 500, 'log-uniform')
                },
                n_iter=100,
                cv=3,
                scoring='average_precision',
                n_jobs=-1,
                random_state=random_state
            )