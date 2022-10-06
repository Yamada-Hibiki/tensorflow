import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# Print the first 5 elements
# print(data.head())

# label, which attribute we want to look for
predict = "G3"

# Features
X = np.array(data.drop([predict], 1))
# Labels
Y = np.array(data[predict])

# x_test, y_test, test the accuracy of the data, we take 10% of the total data to test
# The rest of 90% create the model(train), to test with the rest of 10%
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)


# best = 0
# for _ in range(30):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
#     linear = linear_model.LinearRegression()  # To decide the model
#
#     linear.fit(x_train, y_train)  # Find the best fit line
#     acc = linear.score(x_test, y_test)
#     print("Accuracy: ", acc)  # With 5 attributes we could guess the G3 with this accuracy

    # if acc > best:  # We want to find the best accuracy
    #     best = acc
    #     with open("studentmodel.pickle", "wb") as f:
    #         pickle.dump(linear, f)  # Save the pickle file in the directory

# To load the model
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# To see the actual Linear Regression, Coefficient of 5 variables
print("Intercept: \n", linear.intercept_)
print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

# the prediction of G3, [G1, G2, study time, failures, absences], actual G3
for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])

plot = "G1"
style.use("ggplot")
pyplot.scatter(data[plot], data["G3"])  # decide x axe and y axe
pyplot.xlabel(plot)
pyplot.ylabel("Final Grade")
pyplot.show()
