import preprocess
import model
import time

if __name__=='__main__':
    preprocessor = preprocess.Preprocess()
    modeler = model.Model()

    feature_path = 'UCI HAR Dataset/features.txt'
    columns = preprocessor.get_label(feature_path)
    dev = preprocessor.get_data(X_path = 'UCI HAR Dataset/train/X_train.txt', 
                    y_path='UCI HAR Dataset/train/y_train.txt', 
                    columns = columns, 
                    label=False)
    valid = preprocessor.get_data(X_path = 'UCI HAR Dataset/test/X_test.txt', 
                    y_path='UCI HAR Dataset/test/y_test.txt', 
                    columns = columns, 
                    label=False)
    print('Data loaded! ')
    X, y, X_train, X_test, y_train, y_test = modeler.split_data(dev)
    X_val, y_val = valid.iloc[:,:-1], valid.iloc[:,-1]
    start = time.time()
    print('Start fitting classifier...')
    modeler.clf.fit(X_train, y_train)
    print('Done fitting classifier. Time taken = {:.1f}(s)'.format(time.time()-start))
    acc_test = modeler.accuracy(y_test, modeler.clf.predict(X_test))
    print('Accuracy on Test set: {:.2f}'.format(acc_test*100))
    modeler.cross_validate(modeler.clf, X_val, y_val)