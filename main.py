import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()


def printSpecies(x):
    print(iris.target_names[x])


print(iris.feature_names)
print(iris.target_names)
print(iris.data[0])
printSpecies(iris.target[0])

# Training Classifier

# Setting aside some data for Testing the Model

test_ids = [0, 50, 100]  # each of the 3 species starts at this index
# so 0 is setosa 50 is versicolor and so on
# Getting training data without the extra stuff
train_target = np.delete(iris.target, test_ids)
train_data = np.delete(iris.data, test_ids, axis=0)


# Testing Data

test_target = iris.target[test_ids]
test_data = iris.data[test_ids]

# Now using tree to train

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

print("Expected:" + str(test_target))
print("Predicted:" + str(clf.predict(test_data)))
