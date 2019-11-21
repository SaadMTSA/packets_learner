from sklearn.linear_model import LogisticRegression
from .base import BaseModel
from skopt import BayesSearchCV

class LogisticRegressionModel(BaseModel):
    def __init__(self, random_state, fast=False):
        if fast:
            self.model = LogisticRegression(random_state=random_state, solver='lbfgs')
        else:
            self.model = BayesSearchCV(
                LogisticRegression(random_state=random_state),
                {
                    'C' : (1e-3, 1e+5, 'log-uniform'),
                    'solver' :['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],

                },
                n_iter=10,
                cv=10,
                scoring='f1',
                n_jobs=-1,
                random_state=random_state
            )