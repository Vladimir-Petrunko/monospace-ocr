import os.path

from keras.src.callbacks import EarlyStopping

import ocr_model
import tensorflow

from sklearn.neighbors import KNeighborsClassifier
from numpy import genfromtxt
from tensorflow.keras import layers, models
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from ocr_model import *
from pathlib import Path
from sklearn.model_selection import train_test_split
from constants import *

DATASET_PARTITIONS = 10

def early_callback():
    return EarlyStopping(
        monitor = 'val_accuracy',
        min_delta = 0.00,
        patience = 1000, # Do not stop
        mode = 'max',
        restore_best_weights = True
    )

def print_components(markers, components):
    component_cnt = len(components)
    for i in range(component_cnt):
        print('markers:', markers[i], 'component:', components[i])

def print_prediction_errors(cnt, predictions_errors):
    print('Prediction errors:')
    for i in range(cnt):
        print(i, predictions_errors[i])

def trim(errors, counts):
    new_errors = []
    for i in range(len(errors)):
        res = {}
        for ch in errors[i]:
            error_cnt = errors[i][ch]
            if error_cnt / counts[ch] >= 0.01:
                res[ch] = errors[i][ch]
        new_errors.append(res)
    return new_errors

def create_cnn_model(classes):
    model = models.Sequential()
    model.add(layers.Input(shape = (20, 20, 1)))
    model.add(layers.Conv2D(256, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(512, (3, 3), activation = 'relu'))
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

def create_character_fallback_model(base_path, component, marker, index, level):
    print(component)
    new_mapping = {LABEL_TO_CHARACTER[int(label)]: index for index, label in enumerate(component)}

    if not os.path.exists(base_path):
        os.mkdir(base_path)
    if not os.path.exists(base_path + '/' + str(index)):
        os.mkdir(base_path + '/' + str(index))
    with open(base_path + '/' + str(index) + '/labels.json', 'w') as file:
        file.write(json.dumps(new_mapping))
    with open(base_path + '/' + str(index) + '/markers.json', 'w') as file:
        file.write(json.dumps({'markers': marker}))
    counts, images, labels = get_dataset(characters = new_mapping, font_sub = 'Consolas')

    cnt = len(new_mapping)

    model = create_cnn_model(cnt)
    for i in range(3):
        train_images, test_images = utils.train_test_split(images, DATASET_PARTITIONS, i)
        train_labels, test_labels = utils.train_test_split(labels, DATASET_PARTITIONS, i)
        model.fit(train_images, train_labels, epochs = 3, validation_data = (test_images, test_labels))
    model.fit(images, labels, epochs = 1)
    model.save(base_path + '/' + str(index) + '/model.keras')

    prediction_errors = get_prediction_errors(model, images, labels, len(new_mapping))
    print_prediction_errors(len(prediction_errors), prediction_errors)

def create_character_model():
    counts, images, labels = get_dataset()

    cnt = len(LABEL_TO_CHARACTER)
    character_model = create_cnn_model(cnt)

    for i in range(3):
        train_images, test_images = utils.train_test_split(images, DATASET_PARTITIONS, i)
        train_labels, test_labels = utils.train_test_split(labels, DATASET_PARTITIONS, i)
        character_model.fit(train_images, train_labels, epochs = 3, validation_data = (test_images, test_labels), callbacks = [early_callback()])
    character_model.fit(images, labels, epochs = 1)

    character_model.save('model/character/model.keras')

    prediction_errors = get_prediction_errors(character_model, images, labels, len(LABEL_TO_CHARACTER))
    print_prediction_errors(len(prediction_errors), prediction_errors)

def create_fallback_models():
    counts, images, labels = get_dataset(font_sub = 'Consolas')

    character_model = models.load_model('model/character/model.keras')

    predictions_errors = get_prediction_errors(character_model, images, labels, len(LABEL_TO_CHARACTER))
    predictions_errors = trim(predictions_errors, counts)

    print_prediction_errors(len(LABEL_TO_CHARACTER), predictions_errors)

    markers, components = ocr_model.get_character_connected_components(predictions_errors)
    print_components(markers, components)

    for i, component in enumerate(components):
       create_character_fallback_model('model/character/fallback', component, markers[i], i, 0)

ocr_model.initialize_mappings()
create_character_model()