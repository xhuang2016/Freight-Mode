import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


data_train = pd.read_csv('OHE_train_set_2017.csv', header=0)
data_val = pd.read_csv('OHE_val_set_2017.csv', header=0)
data_test = pd.read_csv('OHE_test_set_2017.csv', header=0)


X_train = data_train.loc[:, data_train.columns != 'MODE']
y_train = data_train.loc[:, data_train.columns == 'MODE']


X_val = data_val.loc[:, data_val.columns != 'MODE']
y_val = data_val.loc[:, data_val.columns == 'MODE']


X_test = data_test.loc[:, data_test.columns != 'MODE']
y_test = data_test.loc[:, data_test.columns == 'MODE']


clf = AdaBoostClassifier(n_estimators=10, learning_rate=0.5, base_estimator = DecisionTreeClassifier(), random_state=42)
clf = clf.fit(X_train, y_train.values.ravel())

y_pred_train = clf.predict(X_train)
y_pred_val = clf.predict(X_val)
y_pred_test = clf.predict(X_test)


print('training set', confusion_matrix(y_train, y_pred_train))
print('val set', confusion_matrix(y_val, y_pred_val))
print('testing set', confusion_matrix(y_test, y_pred_test))


print('training set', classification_report(y_train, y_pred_train, digits=4))
print('val set', classification_report(y_val, y_pred_val, digits=4))
print('testing set', classification_report(y_test, y_pred_test, digits=4))
