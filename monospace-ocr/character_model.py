import os.path


import cv2
import ocr_model
from keras.src.callbacks import EarlyStopping
from tensorflow.keras import layers, models
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from ocr_model import *
from pathlib import Path

DATASET_PARTITIONS = 10

def early_callback():
    return EarlyStopping(
        monitor = 'val_accuracy',
        min_delta = 0.00,
        patience = 1000, # Do not stop
        mode = 'max',
        restore_best_weights = True
    )

def print_prediction_errors(cnt, predictions_errors):
    print('Prediction errors:')
    for i in range(cnt):
        print(i, predictions_errors[i])

def create_cnn_model(classes):
    model = models.Sequential()
    model.add(layers.Input(shape = (12, 12, 1)))
    model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(classes))
    model.summary()

    model.compile(
        optimizer = 'adam',
        loss = SparseCategoricalCrossentropy(from_logits = True),
        metrics = ['accuracy'])

    return model

def get_prediction_errors(model, images, labels, cnt):
    predictions = model.predict(images)
    predictions_errors = [{} for _ in range(cnt)]
    for i in range(len(predictions)):
        verdict = numpy.argsort(predictions[i])[-1]
        if verdict != labels[i]:
            predictions_errors[verdict][int(labels[i])] = predictions_errors[verdict].get(int(labels[i]), 0) + 1

    return predictions_errors

def get_dataset(characters = None, font_sub = ''):
    characters = CHARACTER_TO_LABEL if characters is None else characters
    counts, images, labels = [0 for _ in range(len(characters))], [], []
    fonts = Path('fonts')
    for font in fonts.iterdir():
        font_str = str(font).replace('\\', '/').replace('fonts/', '')
        if font_sub not in font_str:
            continue
        for character in characters:
            character = int(character)
            path = Path('./dataset/character/full/' + str(character) + '/' + font_str)
            if not os.path.exists(path):
                continue
            for img in path.iterdir():
                image = cv2.imread(img, cv2.IMREAD_GRAYSCALE) / 255
                images.append(image.reshape((image.shape[0], image.shape[1], 1)))
                labels.append(characters[character])
                counts[characters[character]] = counts[characters[character]] + 1
    images = numpy.array(images)
    labels = numpy.array(labels)
    shuffle = numpy.random.permutation(len(images))
    return counts, images[shuffle], labels[shuffle]

def create_character_model():
    counts, images, labels = get_dataset()

    cnt = len(LABEL_TO_CHARACTER)
    character_model = create_cnn_model(cnt)

    character_model.fit(images, labels, epochs = 3)

    character_model.save('model/character/model.keras')

    prediction_errors = get_prediction_errors(character_model, images, labels, len(LABEL_TO_CHARACTER))
    print_prediction_errors(len(prediction_errors), prediction_errors)

ocr_model.initialize_mappings()
create_character_model()