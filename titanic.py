import pandas as pd
from sklearn.preprocessing import Imputer

train_dataset = pd.read_csv('train.csv')
X = train_dataset.iloc[:,[2,5,6,7,9]].values
y = train_dataset.iloc[:, 1].values

imputer = Imputer(missing_values = 'NaN' , strategy = 'mean' , axis = 0)
imputer = imputer.fit(X[:,:])
X[:,:] = imputer.transform(X)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

test_dataset = pd.read_csv('test.csv')
X_test_t = test_dataset.iloc[:,[1,4,5,6,8]].values
imputer = Imputer(missing_values = 'NaN' , strategy = 'mean' , axis = 0)
imputer = imputer.fit(X_test_t[:,:])
X_test_t[:,:] = imputer.transform(X_test_t)

y_test_pred = classifier.predict(X_test_t)
print(y_test_pred)
