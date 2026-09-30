import pandas as pd
import numpy as np
import config


from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from catboost import CatBoostClassifier


models = {'logistic_regression': {
    'model': LogisticRegression(),
    'params':{
        'model__C':[0.01, 0.5, 1, 10]
    }
    },
    'knn_class':{
        'model': KNeighborsClassifier(),
        'params': {
            'model__n_neighbors':[3, 5, 7, 10],
            'model__weights': ['uniform', 'distance']
        }
    },
    'lasso':{
        'model': Lasso(),
        'params':{
            'model__alpha':[0, 1, 5, 10]
        }
    },
    'ridge':{
            'model': Ridge(),
            'params':{
                'model__alpha':[0, 1, 5, 10]
            }
        },

    'cat_b':{
        'model': CatBoostClassifier(random_seed=config.random_seed,
                                    early_stopping_rounds = 20,
                                    verbose=50),
        'params':{
            'model__iterations':[1000],
            'model__learning_rate':[1e-2] ,
            'model__depth':list(range(1,11)),
            'model__min_data_in_leaf':[3, 5, 10],
            'model__l2_leaf_reg': [5]
        }
        },

    'lightgbm':{
            'model': CatBoostClassifier(random_seed=config.random_seed, verbose=100),
            'params':{
                'model__iterations':[100, 50, 25], 
                'model__max_depth':[3, 5, 10],
                'model__l2_leaf_reg': [1,3,5,10]
            }
        },
    'dtree':{
        'model': DecisionTreeClassifier(random_state=config.random_seed),
        'params':{
            'model__max_depth':[3, 5, 7, 10],
            'model__min_samples_split':[2, 4, 6, 8, 10]
        }
    }
}