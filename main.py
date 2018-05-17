# Gaussian Naive Bayes
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

# import numpy
import numpy as np


def prepare_sms_dataset(fname):
    spam_notspam = ['spam', 'ham']
    messages = []
    with open(fname) as sf:
        for line in sf:
            messages.append(line.strip().split())

    dataset = []
    # TODO: implement
    return dataset


# load the iris datasets
dataset = datasets.load_iris()

# fit a Naive Bayes model to the data
model = GaussianNB()
model.fit(dataset.data, dataset.target)
print(model)

# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


learnset = prepare_person_dataset("data/person_data.txt")
testset = prepare_person_dataset("data/person_data_testset.txt")
print(learnset)


weight, height = zip(*learnset)
weight = np.array(weight)
height = np.array(height)
model.fit(weight, height)
print(model)


weight, height = zip(*testset)
wight = np.array(weight)
height = np.array(height)
predicted = model.predict(weight)
print(predicted)
print(height)


# summarize the fit of the model
print(metrics.classification_report(height, predicted))
print(metrics.confusion_matrix(height, predicted))
