# Load pickled data
import pickle
import matplotlib.pyplot as pyplot
from sklearn.utils import shuffle
from imblearn.under_sampling import NearMiss
import cv2
import numpy as np

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'data/train.p'
testing_file = 'data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

X_train, y_train = shuffle(X_train, y_train)

### Replace each question mark with the appropriate value.

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:4]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = max(y_train) + 1  # zero-based indexing

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

n, bins, patches = pyplot.hist(y_train, bins=np.arange(0,n_classes))
pyplot.show()

near_miss_undersampling = NearMiss(return_indices=True)
X_s = np.random.randn(len(X_train), 1)
_,_, indexes = near_miss_undersampling.fit_sample(X_s, y_train)

X_under = X_train[indexes]
y_under = y_train[indexes]
pyplot.hist(y_under, bins=np.arange(0,n_classes))
pyplot.show()
