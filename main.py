import math
import random
import csv


# Feature extraction module
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from  sklearn.model_selection import train_test_split


def load_csv(filename):
    """
    Args:
        filename: file name
    Returns:
        list of lines of the file, yet not preprocessed
    """
    lines = csv.reader(open(filename, 'r', encoding='ISO-8859-1'))
    dataset = list(lines)
    return dataset


def vectorize_corpus(dataset):
    """
    Args:
        dataset: dataset not yet preprocessed
    Returns:
        Vectorized matrix of message corpus
        Corpus of text messages
    """
    corpus = []
    for row in dataset:
        corpus.append(row[1])

    vectorizer = CountVectorizer()
    vectorized = vectorizer.fit_transform(corpus).toarray()

    matrix = []
    for i in range(len(dataset)):
        matrix.append([dataset[i][0], vectorized[i]])

    return matrix


def split_dataset(dataset, split_ratio, test_split_ratio):
    """
    Args:
        dataset: input dataset
        split_ratio: split ratio of whole dataset into training and test set
        test_split_ratio: split ratio of test dataset into validation and evaluation set
    Returns:
        returns list of three divided sets in following order: training set, validation set, evaluation set
    """
    total_size = int(len(dataset))
    train_size = int(total_size * split_ratio)

    train_set = []
    test_set = list(dataset)
    while len(train_set) < train_size:
        index = random.randrange(len(test_set))
        train_set.append(test_set.pop(index))

    validation_size = (total_size - train_size) * test_split_ratio
    validation_set = []
    evaluation_set = list(test_set)
    while len(validation_set) < validation_size:
        index = random.randrange(len(evaluation_set))
        validation_set.append(evaluation_set.pop(index))

    return [train_set, validation_set, evaluation_set]


def separate_by_class(dataset):
    """
    Args:
        dataset: input dataset
    Returns:
        class separated dataset in form of python dictionary
    """
    separated = {}
    for i in range(len(dataset)):
        sms_class = 1 if dataset[i][0] == 'spam' else 0
        if sms_class not in separated:
            separated[sms_class] = []
        separated[sms_class].append(dataset[i][1])
    return separated


def mean(numbers):
    return sum(numbers)/float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)


def summarize(dataset):
    # print(dataset)
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    # print('summaries')
    # print(summaries)
    return summaries


def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = {}
    for class_value, instances in separated.items():
        summaries[class_value] = summarize(instances)
    return summaries


def calculate_probability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculate_class_probabilities(summaries, inputVector):
    probabilities = {}
    for class_value, class_summaries in summaries.iteritems():
        probabilities[class_value] = 1
        for i in range(len(class_summaries)):
            mean_, stdev_ = class_summaries[i]
            x = inputVector[i]
            probabilities[class_value] *= calculate_probability(x, mean_, stdev_)
    return probabilities


def predict(summaries, input_vector):
    probabilities = calculate_class_probabilities(summaries, input_vector)
    best_label, best_prob = None, -1

    for class_value, probability in probabilities.iteritems():
        if best_label is None or probability>best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


def get_predictions(summaries, test_set):
    predictions = []
    for i in range(len(test_set)):
        result = predict(summaries, test_set[i])
        predictions.append(result)
    return predictions


def get_accuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][0] == predictions[x]:
            correct += 1
    return (correct/float(len(test_set))) * 100.0


def get_precision(test_set, predictions):
    positive_prediction = 0
    true_positive = 0

    for x in range(len(test_set)):
        if predictions[x] == 1:
            positive_prediction += 1
            if test_set[x][0] == predictions[x]:
                true_positive += 1
    return (true_positive/float(positive_prediction)) * 100.0


def get_recall(test_set, predictions):
    true_positive = 0
    positive = 0
    for x in range(len(test_set)):
        if test_set[x] == 1:
            positive += 1
            if predictions[x] == test_set[x][0]:
                true_positive += 1
    return (true_positive/float(positive)) * 100.0


def main():
    filename = 'data/spam.csv'
    split_ratio = 0.67
    test_split_ratio = 0.5
    dataset = load_csv(filename)
    vectorized_dataset = vectorize_corpus(dataset)
    # training_set, test_set = split_dataset(dataset, split_ratio)
    training_set, validation_set, evaluation_set = split_dataset(vectorized_dataset, split_ratio, test_split_ratio)

    # prepare model
    summaries = summarize_by_class(training_set)
    print('test')
    print(summaries)


    # print predictions
    predictions = get_predictions(summaries, evaluation_set)
    # print(predictions)

    # test accuracy of the model
    accuracy = get_accuracy(evaluation_set, predictions)
    print('Accuracy: {0}').format(accuracy)


main()
