#Предообработка данных
#ЗАполнение
import config
from data.data import load_data
from preprop import prepr
#import sklearn
#import datetime.datetime

import train as t

import os

def run():
    if config.model_mode == 'train':
        
    #Загрузка данных

        train = load_data(config.source_train)
        
    #print(f'Завершена загрузка файлов')

    ### предобработка
        tr = prepr(config=config.preprocessor_config)

        X_train, X_test, y_train, y_test = tr.run(train)

        model = t.trainer(config.trainer_config)

        model.fit(X_train, y_train)

        pred = t.predict(model.best_model, X_test)

        metrics = t.evaluate(pred, y_test)

        print(model.best_model)
        print(metrics)     

        if config.prepr_to_file:
            tr.save()
            model.save()
    elif config.model_mode == 'cv':

        train = load_data(config.source_train)
        tr = prepr(config=config.preprocessor_config)
        train = tr.prepr_cv(train)
        tr.strata(train)
        print(train.columns)

        X = train.drop(columns=tr.config.target)
        y = train[tr.config.target]

        model = t.trainer(config.trainer_config)
        
        model.fit_cv(X, y, tr)

        print(model.models_results)

        #pred = t.predict(model.best_model, X_test)

        #metrics = t.evaluate(pred, y_test)

        print(model.best_model)
        #print(model.models_results)

        if config.prepr_to_file:
            #tr.save()
            model.save()
        

    elif config.model_mode == 'inference':

        test = load_data(config.source_test)
        
        tr = prepr(config.preprocessor_config)

        test = tr.feature_engineering(test)
        
        #if os.path.exists(config.preprocessor_config.saved_prepr_data):
        #     tr.load()
        # else:
        #     raise FileNotFoundError('Файл не существует. Либо предобработчик не сохранен, либо неверный адрес')

        #test_transformed = tr.run(test)

        
        model = t.trainer(config.trainer_config)
        if os.path.exists(config.saved_best_model):
             model.load()
        else:
            raise FileNotFoundError('Файл не существует. Либо модель не сохранена, либо неверный адрес')

        prediction = t.predict(model.best_model, test)

        submit = test[['PassengerId']].copy()
        submit['Survived'] = prediction
        submit.to_csv(config.submit_file, index=False)
        
        return prediction

prediction = run()