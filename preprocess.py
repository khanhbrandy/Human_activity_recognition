import pandas as pd
import numpy as np

class Preprocess():
    def __init__(self):
        pass 

    def get_label(self, feature_path):
        columns = []
        file = open(feature_path, 'r')
        for row in file:
            columns.append(row.replace('  ', ' ').strip().split(' ')[1])
        file.close()
        return columns 

    def get_data(self, X_path, y_path, columns, label=False):
        X = pd.read_csv(X_path, header=None)
        y = pd.read_csv(y_path, header=None, names=['CLASS'])
        train_list = []
        for i in range(len(X)):
            train_list.append(X.loc[i][0].replace('  ', ' ').strip().split(' '))
        X_df = pd.DataFrame(train_list)
        if label:
            X_df.columns = columns
        else:
            pass
        df = pd.concat([X_df, y], axis = 1)
        return df
        