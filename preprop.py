# предобработка

import joblib
import pandas as pd
import os

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer


class prepr():
    '''класс получает сырые данные и гарантированно работает в 2 режимах:
    - обучение: выдает готовые данные для обучения;
    - тестирование: используя обученный скейлер, подготовливает тестовые данные для предсказания
    '''
    def __init__(self, config):
        self.config = config# определяет конфигуратор в классе препроцессора
        self.history = dict() #начальное состояние истории
        self.fitted_preproccessor = None #начальное состояние препроцессора

        self.run_dispatcher = {'train':self.prepare_train,
                               'inference':self.prepare_inference}

        self.stratify_feature = None

        self.del_features = []
        
#очистка
    def dub_remove(self, data:pd.DataFrame):
        '''
        Удаление дублей в записи.
        '''
        if self.config.duplicates['enabled']: 

            print(f'Удаление дубликатов. Размер исходной таблицы:{data.shape}')
            
            self.history['remove_duplicates'] = dict()
            self.history['remove_duplicates']['rows_before'] = len(data)
            data = data.drop_duplicates(subset=self.config.duplicates['subset'], ignore_index=True).copy()
            self.history['remove_duplicates']['rows_after'] = len(data)
            self.history['remove_duplicates']['removed_duplicates'] = self.history['remove_duplicates']['rows_before'] - \
            self.history['remove_duplicates']['rows_after']        
            self.history['remove_duplicates']['distinct_cols'] = self.config.duplicates['subset']
           
        return data

    def anomaly_handler(self, data):

        i = 1
        self.history['anomaly']= dict()

        if self.config.data_filters['enabled']:
            for condition in self.config.data_filters['rules']:
                col, cond , border = condition
                rows_before = len(data)
                
                if cond =='more':  
                    data = data[data[col]>border].copy()              
                    
                elif cond =='less':                
                    data = data[data[col]<border].copy()
                    
                elif cond == 'eq':                
                    data = data[data[col]==border].copy()
                
                elif cond == 'neq':                
                    data = data[data[col]!=border].copy()
                else:
                    raise ValueError(f"Неверное условие {cond}")
                rows_after = len(data)
                self.history['anomaly'] [f'rule_{i}'] = {'rows_before': rows_before,
                                                        'rows_after': rows_after,
                                                        'removed_rows': rows_before - rows_after,
                                                        'column': col,
                                                        'condition': cond,
                                                        'border': border
                                                        }
                i += 1
        return data

    def feature_engineering(self, data):
        i = 1
        self.history['feat_eng'] = dict()

        if self.config.feat_eng['enabled']:

            for condition in self.config.feat_eng['rules']:
                cols, cond, border = condition
                shape_before = data.shape
                #(cond, cols, self.config.target, self.config.mode)

                if self.config.target in cols:
                    print('Попытка создания дополнительного признака на основе целевого пропущена')
                    continue

                elif cond =='union':
                    data[f'feature_{i}'] = data[cols].apply(lambda x: "_".join(x.astype(str)), axis = 1)

                elif cond =='sum':
                    data[f'feature_{i}'] = data[cols].apply(lambda x: x.sum(), axis = 1)

                elif cond == 'gt':
                    data[f'feature_{i}'] = (data[cols] > border).astype('int32')

                elif cond == 'eq':
                    data[f'feature_{i}'] = (data[cols] == border).astype('int32')
                    
                
                else:
                    raise ValueError('There is no rules')

                data = data.copy().drop(columns=cols)
                print(f'droped cols {cols}')
                shape_after = data.shape

                self.history['feat_eng'][f'rule_{i}'] = {
                    'shape_before': shape_before,
                    'shape_after': shape_after,
                    'operation': cond,
                    'deleted_cols': cols,
                    'new_cols': {f'feature_{i}': data[f'feature_{i}'].dtype}
                }
                    
        return data

    
    def strata(self, data):
        self.stratify_feature =  data[self.config.stratify['rule']].apply(lambda x: "_".join(x.astype(str)), axis = 1)
        return self.stratify_feature
    

    def splitter(self, X, y):
        '''
        Разделение выборки на обучающую и тестовую 
        '''
        self.history['split'] = dict()

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            stratify=self.stratify_features,
                                                            test_size=self.config.split['test_size'],
                                                            random_state=self.config.split['random_state'],
                                                            shuffle=self.config.split['shuffle'])
        self.history['split'] = {'test_size': self.config.split['test_size'],
                                 'random_state': self.config.split['random_state'],
                                 'shuffle': self.config.split['shuffle'],
                                 'row_before':len(y),
                                 'train_rows':len(y_train),
                                 'test_rows':len(y_test)}
        if self.stratify_features is None:
            self.history['split']['stratified'] = False
        else:
            self.history['split']['stratified'] = True
        return X_train, X_test, y_train, y_test

    def fit(self, X):
        '''
        обучение предобработчика
        '''
        self.fitted_preproccessor = ColumnTransformer(self.config.transformer)
        self.fitted_preproccessor.fit(X)
        

    def transform(self, X):
        self.history['transformed']={'complition':False, 'shape_before':X.shape}
        if self.fitted_preproccessor is None:
             raise ValueError('Преобразователь не обучен. Данные не преобразованы')
        X_transformed = self.fitted_preproccessor.transform(X)
        self.history['transformed']={'completion':True,'shape_after':X_transformed.shape}
        return X_transformed

    def run(self, data):
        return self.run_dispatcher[self.config.mode](data)

    def prepare_train(self, data):
        
        data = self.dub_remove(data)
        data = self.anomaly_handler(data)
        data = self.fill_emp(data)
        data = self.strata(data)
        data = self.feature_engineering(data)

        X_train, X_test, y_train, y_test = self.splitter(data.drop(self.config.target,axis=1),
                                                         data[self.config.target])
        self.fit(X_train)
        X_train_transformed = self.transform(X_train)
        X_test_transformed = self.transform(X_test)
        return X_train_transformed, X_test_transformed, y_train, y_test

    def prepr_cv(self, data):
            
            data = self.dub_remove(data)
            data = self.anomaly_handler(data)
            
            return data

    def cv_splitter(self, X, strata, random_state):
        folds = StratifiedKFold(self.config.cv_config['n_folds'],
                                random_state=random_state,
                                shuffle=True)
        return folds.split(X, strata)
        
    
    
    def postpr_cv(self, X):
                
        X = self.fill_emp(X)
        X = self.feature_engineering(X)

        self.fit(X)
        X_transformed = self.transform(X)
        return X_transformed

    def prepare_inference(self, data):
        data = self.fill_emp(data)
        data = self.feature_engineering(data)
        data_transformed = self.transform(data)
        return data_transformed

    def save(self):
        if self.fitted_preproccessor is None:
            raise FileNotFoundError('Предобработчик не обучен. Сохранять нечего.')
        saved = {
            'fitted_preproccessor': self.fitted_preproccessor,
            'history': self.history,
            'config': self.config
         }
        joblib.dump(saved, self.config.saved_prepr_data)

    def load(self):
            if os.path.exists(self.config.saved_prepr_data):
                loaded = joblib.load(self.config.saved_prepr_data)
                self.fitted_preproccessor = loaded['fitted_preproccessor']
                self.history = loaded['history']
                self.config = loaded['config']
            else:
                raise ValueError("Файл не существует")


class fill_emp_transformer():

    def __init__(self, config):
        self.config = config
        self.history = dict()
        
    def _get_group_value(self, data, grp, group_values):
        temp = data[grp].copy()
        temp['_original_index'] = data.index
        temp = temp.merge(group_values.rename('_fill_value').reset_index(),
                          how='left',
                          on=grp,
                          sort=False)
        temp = temp.set_index('_original_index')
        fill_value = temp['_fill_value'].reindex(data.index)
        return fill_value

    def fit(self, data, y = None):
        
        for i, rule in enumerate(self.config.fill_emp, start=1):  

            col = rule['column']
            grp = rule['grp_b']
            method = rule['method']
            
            history_rule = {
                'column':col,
                'grp_b': grp,
                'method': method
            }

            if method =='constant':
                fill_value = rule['value']
                history_rule['value'] = fill_value

            elif grp is None:

                if method == 'mode':
                    fill_value = data[col].mode().iloc[0]
                else:
                    fill_value = getattr(data[col], method)()

                history_rule['value'] = fill_value

            else:
                if method =='mode':
                    group_values = (data.groupby(grp)[col].agg(lambda x: x.mode().iloc[0]))
                    _global_agg = data[col].agg(lambda x: x.mode().iloc[0])
                else:
                    group_values = (data.groupby(grp)[col].agg(method))
                    _global_agg = data[col].agg(method)

                fill_value = self._get_group_value(data, grp, group_values)

                history_rule['value'] = group_values
                history_rule['_global_value'] = _global_agg

            self.history[f'rule_{i}'] = history_rule

        return self


    def transform(self, data):

        data = data.copy()

        for i in range(1, len(self.history)+1):

            rule = self.history[f'rule_{i}']

            grp = rule['grp_b']
            value = rule['value']
            col = rule['column']

            if grp is None:
                fill_value = value

            else:
                fill_value = self._get_group_value(data, grp, value)

            data[col] = data[col].fillna(fill_value)

            if grp is not None and data[col].isna().any():
                data[col] = data[col].fillna(self.history[f'rule_{i}']['_global_value'])

        return data


class feature_engineering():

    def __init__(self, config):
        self.config = config
        self.history = {}
        self.dispatcher = {
            'union':self.union,
            'sum':self.sum
        }
        

    def fit(self, data, y=None):
        return self

    def transform(self, data):
        
        i = 1
        data = data.copy()

        if self.config['enabled']:

            for condition in self.config['rules']:
                cols, cond, border = condition
                shape_before = data.shape

                if self.config.target in cols:
                    print('Попытка создания дополнительного признака на основе целевого пропущена')
                    continue

                elif cond =='union':
                    data[f'feature_{i}'] = data[cols].apply(lambda x: "_".join(x.astype(str)), axis = 1)

                elif cond =='sum':
                    data[f'feature_{i}'] = data[cols].apply(lambda x: x.sum(), axis = 1)

                elif cond == 'gt':
                    data[f'feature_{i}'] = (data[cols] > border).astype('int32')

                elif cond == 'eq':
                    data[f'feature_{i}'] = (data[cols] == border).astype('int32')
                    
                
                else:
                    raise ValueError('There is no rules')

                data = data.copy().drop(columns=cols)
                print(f'droped cols {cols}')
                shape_after = data.shape

                self.history[f'rule_{i}'] = {
                    'shape_before': shape_before,
                    'shape_after': shape_after,
                    'operation': cond,
                    'deleted_cols': cols,
                    'new_cols': {f'feature_{i}': data[f'feature_{i}'].dtype}
                }
                    
        return data