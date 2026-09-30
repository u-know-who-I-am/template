from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder

#model_mode = "inference"
model_mode = 'cv'
#model_mode = "train"

path_train = r'data\train.csv'
path_test = r'data\test.csv'
path_to_db = 'ds-plus-final.db'

saved_best_model = 'best_model.pkl'

submit_file = 'submit.csv'

prepr_to_file = True

num_scaler = StandardScaler()
cat_scaler = OneHotEncoder(handle_unknown='ignore')
ord_scaler = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

num_cols = ['Age', 'Fare']
cat_cols = ['Sex', 'Embarked']
ord_cols = ['Pclass', 'SibSp', 'Parch']

drop_cols = []
random_seed = 42

cleaning_rules = {'drop_duplicates':True}

db_config = {
    'user': 'praktikum_student', # имя пользователя,
    'pwd': 'Sdf4$2;d-d30pp', # пароль,
    'host': 'rc1b-wcoijxj3yxfsf3fs.mdb.yandexcloud.net',
    'port': 6432, # порт подключения,
    'db': 'data-science-vehicle-db' # название базы данных,
} 
source_train = \
    { # первый источник
        'type':'csv',
        'path':path_train
    }
source_test = \
     { # второй источник
         'type':'csv',
         'path':path_test
     }


class preprocessor_config():

    saved_prepr_data = 'saved/prepr.pkl'

    global model_mode
    
    mode = model_mode
    target = 'Survived'
    num_cols = ['Age', 'Fare']
    cat_cols = ['Sex', 'Embarked']
    ord_cols = ['Pclass', 
                'SibSp', 
                'Parch'
                ]

    cv_config = {'n_folds': 5}
    
    duplicates = {
        'enabled':True,
        'subset': None
    }

    fill_emp = [{'column':'Age',
                 'grp_b':['Pclass', 'Parch'],
                 'method':'mean'},
                {'column':'Cabin',
                 'grp_b' : None,
                 'method': 'constant',
                 'value': 'unknown'},
                {'column':'Fare',
                 'grp_b':['Pclass'],
                 'method':'mode'},
                {'column':'Embarked',
                 'grp_b': None,
                 'method':'mode'}
                ]

    data_filters = {'enabled': True,
                    'rules' : [('Fare', 'less', 500), #удаление выбросов
                               ('Age', 'less', 65)]}

    feat_eng = {'enabled': False,
                'rules' : [(['Sex', 'Pclass'], 'union', None)#,# та как target есть  в списке, то будет создан признак для стратификации.
                           #(['Age', 'Fare'], 'sum')
                           #(['Parch'], 'gt', 0),
                           #(['Pclass'], 'eq', 3)
                           ]}

    stratify = {'enabled': True,
                'rule' : ['Sex', 'Survived']#,# та как target есть  в списке, то будет создан признак для стратификации.
                }

    split = {'test_size':0.2,
             'random_state':42,
             'shuffle': True}

    transformer = [('num_col', num_scaler, num_cols),
                   ('cat_col', cat_scaler, cat_cols),
                   ('ord_col', ord_scaler, ord_cols)
                   ]

    
    
class trainer_config():

    task = 'classification'

    models = ['logistic_regression',
              'dtree',
              'knn_class'#, 
              #'cat_b',
              #'lightgbm'
              ]

    metrics = ['accuracy', 'f1', 'recall', 'precision']

    primary_metric = 'accuracy'

    threshold = 0.5

    search ={'method' : 'grid',
             'cv' : 5,
             'n_iter' : 10}

    