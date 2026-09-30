from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
import models
import config 
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os
import preprop as pr
import pandas as pd



class trainer():
    def __init__(self, config):
        self.config = config

        self.best_model = None
        self.best_score = float('-inf')
        self.best_params = None

        self.search_dispatcher = {
            'grid': self.grid_search,
            'random':self.random_search
        }
        self.models_results = dict()
        

    def random_search(self, model, params, split):
        '''
        функция для случайного поиска гиперпараметров
        '''

        search = RandomizedSearchCV(model,
                                    param_distributions=params,
                                    n_iter=self.config.search['n_iter'],
                                    cv=split,
                                    scoring = self.config.primary_metric,
                                    random_state= config.random_seed)
        return search
    
    def grid_search(self,model, params, split):
        '''
        функция для поиска гиперпараметров полным перебором

        '''
        search = GridSearchCV(model,
                              param_grid=params,
                              cv=split,
                              scoring = self.config.primary_metric
                              )
        
        return search
        
    def fit(self, X,y):
       
        for model in self.config.models:
            curr_model = models.models[model]['model']
            params = models.models[model]['params']
           
            search = self.search_dispatcher[self.config.search['method']](curr_model, params, pr.cv_splitter(X))
           
            search.fit(X,y)

            cv_res = search.cv_results_
            #best_10_indice = [i for i, x in enumerate(cv_res['rank_test_score']) if x = 10]

            self.models_results[model] = {
                'score': search.best_score_,
                'params': search.best_params_,
                'metric': self.config.primary_metric,
                'std_score':cv_res['std_test_score'][search.best_index_]
                }
            
            if self.best_score < search.best_score_:
                self.best_model = search.best_estimator_
                self.best_score = search.best_score_
                self.best_params = search.best_params_
                self.best_std_score = cv_res['std_test_score'][search.best_index_]

    def fit_cv(self, X, y=None, tr=None):

        if  y is None:
            data = X.copy()
            strata = tr.strata(data)
            y = data[tr.config.target]
            X = data.drop(columns=tr.config.target)
        else:
            data = pd.concat([X, y], axis=1)
            strata = tr.strata(data)

        X = tr.feature_engineering(X)

        

        for model in self.config.models:
            curr_model = models.models[model]['model']
            params = models.models[model]['params']

            ppl = Pipeline([
                ('fill_emp', pr.fill_emp_transformer(config.preprocessor_config)),
                ('preprocessor', ColumnTransformer(config.preprocessor_config.transformer)),
                ('model', curr_model)
            ])             

            search = self.search_dispatcher[self.config.search['method']](ppl, params, tr.cv_splitter(X, strata, config.random_seed))
            
            search.fit(X,y)

            cv_res = search.cv_results_
            #best_10_indice = [i for i, x in enumerate(cv_res['rank_test_score']) if x = 10]

            self.models_results[model] = {
                'score': search.best_score_,
                'params': search.best_params_,
                'metric': self.config.primary_metric,
                'std_score':cv_res['std_test_score'][search.best_index_]
                }
            
            if self.best_score < search.best_score_:
                self.best_model = search.best_estimator_
                self.best_score = search.best_score_
                self.best_params = search.best_params_
                self.best_std_score = cv_res['std_test_score'][search.best_index_]

        if self.best_model is None:
            raise ValueError('Обучение не успешно')

        self.best_model.fit(X, y)

        return self

    def fit_opt(self, X, y=None, tr=None):

        if  y is None:
            data = X.copy()
            strata = tr.strata(data)
            y = data[tr.config.target]
            X = data.drop(columns=tr.config.target)
        else:
            data = pd.concat([X, y], axis=1)
            strata = tr.strata(data)

        X = tr.feature_engineering(X)

        for model in self.config.models:
            curr_model = models.models[model]['model']
            params = models.models[model]['params']

            ppl = Pipeline([
                ('fill_emp', pr.fill_emp_transformer(config.preprocessor_config)),
                ('feature_engineer', pr.feature_engineering(config.feature_engineering)),
                ('preprocessor', ColumnTransformer(config.preprocessor_config.transformer)),
                ('model', curr_model)
            ])             

            def objective(trial):
                params = { 'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                           'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                            'subsample': trial.suggest_float('subsample', 0.6, 1.0), 
                            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), 
                            'n_estimators': 3000, 'random_state': 42 }

                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = []


            
            search.fit(X,y)

            cv_res = search.cv_results_
            #best_10_indice = [i for i, x in enumerate(cv_res['rank_test_score']) if x = 10]

            self.models_results[model] = {
                'score': search.best_score_,
                'params': search.best_params_,
                'metric': self.config.primary_metric,
                'std_score':cv_res['std_test_score'][search.best_index_]
                }
            
            if self.best_score < search.best_score_:
                self.best_model = search.best_estimator_
                self.best_score = search.best_score_
                self.best_params = search.best_params_
                self.best_std_score = cv_res['std_test_score'][search.best_index_]

        if self.best_model is None:
            raise ValueError('Обучение не успешно')

        self.best_model.fit(X, y)

        return self


    def save(self):
        if self.best_model is None:
            raise FileNotFoundError('Модели не существует')
        
        joblib.dump(self.best_model, config.saved_best_model)

        

    def load(self):
        if os.path.exists(config.saved_best_model):
            self.best_model = joblib.load(config.saved_best_model)
        else:
            raise ValueError("Файл не существует")
    

metric_dispatcher = {'f1':f1_score,
                     'accuracy':accuracy_score,
                     'recall':recall_score,
                     'precision':precision_score
                     }



def evaluate(y, pred):
    metrics = dict()

    for m in config.trainer_config.metrics:
        metrics[m] = metric_dispatcher[m](y, pred)

    return metrics

def predict(best_model, X):
    
    if best_model is None:
        raise ValueError('Модель отсутствует или необучена')

    if config.trainer_config.task == 'classification':
        if config.trainer_config.threshold == None:
            predict = best_model.predict(X)
        else:
            predict = best_model.predict_proba(X)[:,1]
            predict = [1 if p >= config.trainer_config.threshold else 0 for p in predict]
    else:
        raise ValueError('Выбранный тип задачи еще не предусмотрен, либо ошибка в названии типа задачи')

    return predict