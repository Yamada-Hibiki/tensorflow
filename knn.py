import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
# print(data.head())

# Transform non-numeric value into numeric values
# Transform it into appropriate integer, return it into numpy arra
le = preprocessing.LabelEncoder()
# fir_transform function takes in a list and returns new values
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

# Convert it into a big list
X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# To determine the number of neighbors
model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train, y_train)
acc = model.score(x_train, y_train)
print(acc)

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for i in range(len(predicted)):
    print("Predicted: ", names[predicted[i]], "Data: ", x_test[i], "Actual: ", names[y_test[i]])
    n = model.kneighbors([x_test[i]], 9, True)  # Need a big list so that it will be in two dimensions
    print("N: ", n)  # Show us distance between each item and index

