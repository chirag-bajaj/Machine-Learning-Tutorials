import numpy as np
from sklearn import datasets,metrics
from sklearn.naive_bayes import GaussianNB

X,y = datasets.load_iris(return_X_y=True)

X_train = X[range(0,150,2),:]
y_train = y[range(0,150,2)]

X_test = X[range(1,150,2),:]
y_test = y[range(1,150,2)]

clf= GaussianNB()

clf.fit(X_train,y_train)
prediction = clf.predict(X_test)

print(prediction)
print(metrics.accuracy_score(y_test,prediction,normalize=True))
print(metrics.confusion_matrix(y_test,prediction))


