#Feature engineering
import pandas as pd
class FeatureEngi():
    def __init__(self):
        self.add_cols = []

    def add_relatives(self, df):
        df = df.copy()
        df['relatives'] = df.groupby('Ticket')['Ticket'].transform('count') - 1
        self.add_cols.append('relatives')
        return df

    def add_cabins_spl(self, df):
        df = df.copy()
        df['Cabin_sec', 'Cabin_num'] = df['Cabin'].str.split(n=1, regex=True, pat='(?<=^.)', expand=True)
        self.add_cols.append('Cabin_num', 'Cabin_sec' )



# Feature engineering


print(f'{datetime.now} Создание нового признака `relative`')
train['relatives'] = train.groupby('Ticket')['Ticket'].transform('count') - 1
print(f'{datetime.now} Создание новых признаки `Cabin_sec` и `Cabin_num`')
train[['Cabin_sec', 'Cabin_num']] = train['Cabin'].str.split(n=1, regex=True, pat='(?<=^.)', expand=True)
