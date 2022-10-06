import sklearn
from sklearn import svm
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

# Support Vector Machines, work for high dimensional data and does not have a linear correspondence

# Use data sets come with sklearn
cancer = datasets.load_breast_cancer()

# print("Features: ", cancer.feature_names)
# print("Labels: ", cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
# print(x_train[:5], y_train[:5])

classes = ["malignant", "benign"]

# To work with KNN, but not appropriate since it takes time in several dimensions
# clf = KNeighborsClassifier(n_neighbors=11)  # To work with KNN, but not appropriate since it takes time in several
clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train, y_train)

prediction = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, prediction)

print(acc)
