#A feature function for extracting the last letter of a word:

def gender_features(word):
    return {'last_letter': word[-1]}
#BoW model 
Documents:
1. "The cat sat on the mat."
2. "The dog lay on the rug."

Vocabulary: ["The", "cat", "sat", "on", "mat", "dog", "lay", "rug"]

Feature Vectors:
1. [2, 1, 1, 1, 1, 0, 0, 0]
2. [1, 0, 0, 1, 0, 1, 1, 1]


#how to extract features for classifying names by gender:
#code description in Readme file

import nltk

# Feature function: last letter of the name
def gender_features(name):
    return {'last_letter': name[-1]}

# Dataset of names
from nltk.corpus import names
import random

# Prepare labeled data
labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                 [(name, 'female') for name in names.words('female.txt')])
random.shuffle(labeled_names)

# Apply features
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]

# Train a classifier
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Evaluate the classifier
print(nltk.classify.accuracy(classifier, test_set))

# Test with a new name
print(classifier.classify(gender_features('Emily')))



