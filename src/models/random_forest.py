from sklearn.ensemble import RandomForestClassifier
from .base import BaseModel
from skopt import BayesSearchCV

import logging as LOGGER

LOGGER.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=LOGGER.INFO)

class RandomForestModel(BaseModel):
    def __init__(self, random_state, fast=False):
        if fast:
            self.model = RandomForestClassifier(random_state=random_state)
        else:
            self.model = BayesSearchCV(
                RandomForestClassifier(random_state=random_state),
                {
                    'n_estimators': [10,50,100,200,500],  
                    'min_samples_split': (100,1000),
                    'max_depth': (5, 15),
                    'min_samples_leaf': (100,1000),
                },
                n_iter=100,
                cv=3,
                scoring='f1',
                n_jobs=-1,
                random_state=random_state
            )
