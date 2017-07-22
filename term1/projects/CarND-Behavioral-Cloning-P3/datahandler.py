import os
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

DEFAULT_FILENAME = "samples/driving_log.csv"
DEFAULT_IMG_DIR = "samples/IMG"
DEFAULT_ANGLE_CORRECTION = 0.15


def augment(image, steering_angle):
    reverse_image = np.fliplr(image)
    reverse_angle = -steering_angle
    return reverse_image, reverse_angle


def read_driving_log(filename=DEFAULT_FILENAME):
    samples = []
    with open(filename) as log:
        reader = csv.reader(log)
        for entry in reader:
            samples.append(entry)
    return samples


def get_images(entry):
    images = []
    angles = []

    image_center = cv2.imread(os.path.join(DEFAULT_IMG_DIR, os.path.basename(entry[0])))
    steering_angle = float(entry[3])

    image_left = cv2.imread(os.path.join(DEFAULT_IMG_DIR, os.path.basename(entry[1])))
    steering_angle_left = steering_angle + DEFAULT_ANGLE_CORRECTION

    image_right = cv2.imread(os.path.join(DEFAULT_IMG_DIR, os.path.basename(entry[2])))
    steering_angle_right = steering_angle - DEFAULT_ANGLE_CORRECTION

    reverse_image, reverse_angle = augment(image_center, steering_angle)

    images.extend([image_center / 255, image_left / 255, image_right / 255, reverse_image / 255])
    angles.extend([steering_angle, steering_angle_left, steering_angle_right, reverse_angle])

    return images, angles


def split_data():
    samples = read_driving_log()
    return train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_of_samples = len(samples)

    while 1:
        for offset in range(0, num_of_samples, batch_size):
            batch = samples[offset:offset + batch_size]
            images = []
            angles = []
            for entry in batch:
                entry_images, entry_angles = get_images(entry)
                images.extend(entry_images)
                angles.extend(entry_angles)

            X_train = np.array(images)
            Y_train = np.array(angles)

            yield shuffle(X_train, Y_train)
