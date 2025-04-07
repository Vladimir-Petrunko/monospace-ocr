import cv2
import numpy
import ocr_model
import tensorflow
import utils
from tensorflow.keras import layers, models
from ocr_model import *
from pathlib import Path

DATASET_PARTITIONS = 5

def get_dataset(mode):
    images, labels = [], []
    fonts = Path('fonts')
    for font in fonts.iterdir():
        font_str = str(font).replace('\\', '/').replace('fonts/', '')
        for character in range(33, 127):
            path = Path('./dataset/character/' + str(character) + '/' + font_str)
            for img in path.iterdir():
                image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                _, image = cv2.threshold(image, 127, 255, cv2.THRESH_OTSU)
                images.append(image / 255)
                if mode == 'character':
                    labels.append(CHARACTER_TO_LABEL[character])
                else:
                    labels.append(FONT_TO_LABEL[font_str.split('-')[0]])
    images = numpy.array(images)
    labels = numpy.array(labels)
    shuffle = numpy.random.permutation(len(images))
    return images[shuffle], labels[shuffle]

def create_character_model():
    images, labels = get_dataset(mode = 'character')

    model = models.Sequential()
    model.add(layers.Input(shape = (24, 12)))
    model.add(layers.Conv1D(256, 2, activation = 'relu'))
    model.add(layers.MaxPooling1D(2, strides = 1))
    model.add(layers.Flatten())
    model.add(layers.Dense(94))

    model.summary()

    model.compile(optimizer = 'adam',
                  loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                  metrics = ['accuracy'])

    for split in range(DATASET_PARTITIONS):
        train_images, test_images = utils.train_test_split(images, DATASET_PARTITIONS, split)
        train_labels, test_labels = utils.train_test_split(labels, DATASET_PARTITIONS, split)
        model.fit(train_images, train_labels, epochs = 5, validation_data = (test_images, test_labels))

    model.save('model/character.keras')

    predictions = model.predict(images)
    predictions_result = numpy.zeros((len(CHARACTER_TO_LABEL), 2), dtype = 'int')
    predictions_errors = [{} for _ in range(len(CHARACTER_TO_LABEL))]
    for i in range(len(predictions)):
        verdict = numpy.argsort(predictions[i])[-1]
        if verdict == labels[i]:
            predictions_result[verdict][1] = predictions_result[verdict][1] + 1
        else:
            predictions_result[verdict][0] = predictions_result[verdict][0] + 1
            predictions_errors[verdict][chr(LABEL_TO_CHARACTER[labels[i]])] = predictions_errors[verdict].get(chr(LABEL_TO_CHARACTER[labels[i]]), 0) + 1

    for i in range(len(CHARACTER_TO_LABEL)):
        print(chr(LABEL_TO_CHARACTER[i]), 'correct', predictions_result[i][1], 'wrong', predictions_result[i][0], 'errors', predictions_errors[i])

ocr_model.initialize_mappings()
create_character_model()