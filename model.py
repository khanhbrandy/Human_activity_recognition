import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


class Metrics():
    def __init__(self):
        pass
    def accuracy(self, y_test, y_pred):
        acc = metrics.accuracy_score(y_test, y_pred)
        return acc

class Model(Metrics):
    def __init__(self):
        self.seed = 1003
        self.clf = RandomForestClassifier(max_depth=5, random_state=self.seed)

    def split_data(self, data):
        X, y = data.iloc[:,:-1], data.iloc[:,-1]
        seed = self.seed
        test_size = 0.2
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=seed)
        return X, y, X_train, X_test, y_train, y_test
        
    def cross_validate(self, model, X, y):
        seed = self.seed
        kfold = model_selection.StratifiedKFold(n_splits=4,shuffle=True, random_state=seed)
        results = model_selection.cross_val_score(model, X, y, scoring='accuracy',cv=kfold)
        print('Validated accuracy: {:.2f} (+/- {})'.format(results.mean()*100, results.std()*100))
        pass 

