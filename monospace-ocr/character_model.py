import os.path

import json
import ocr_model
import tensorflow
from tensorflow.keras import layers, models
from ocr_model import *
from pathlib import Path
from sklearn.model_selection import train_test_split

DATASET_PARTITIONS = 10

def get_prediction_errors(model, images, labels, cnt):
    predictions = model.predict(images)
    predictions_errors = [{} for _ in range(cnt)]
    for i in range(len(predictions)):
        verdict = numpy.argsort(predictions[i])[-1]
        if verdict != labels[i]:
            predictions_errors[verdict][int(labels[i])] = predictions_errors[verdict].get(int(labels[i]), 0) + 1

    for i in range(cnt):
        print(i, predictions_errors[i])

    return predictions_errors

def get_dataset(mode, character_type, characters = None):
    characters = CHARACTER_TO_LABEL if characters is None else characters
    counts, images, labels = [0 for _ in range(len(characters))], [], []
    fonts = Path('fonts')
    for font in fonts.iterdir():
        font_str = str(font).replace('\\', '/').replace('fonts/', '')
        for character in characters:
            character = int(character)
            path = Path('./dataset/character/' + character_type + '/' + str(character) + '/' + font_str)
            if not os.path.exists(path):
                continue
            for img in path.iterdir():
                image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                _, image = cv2.threshold(image, 127, 255, cv2.THRESH_OTSU)
                images.append(1 - (image / 255))
                if mode == 'character':
                    labels.append(characters[character])
                    counts[characters[character]] = counts[characters[character]] + 1
                else:
                    labels.append(FONT_TO_LABEL[font_str.split('-')[0]])
    images = numpy.array(images)
    labels = numpy.array(labels)
    shuffle = numpy.random.permutation(len(images))
    return counts, images[shuffle], labels[shuffle]

def create_character_fallback_model(base_path, component, index, level):
    print(component)
    new_mapping = {int(label): int(index) for index, label in enumerate(component)}
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    if not os.path.exists(base_path + '/' + str(index)):
        os.mkdir(base_path + '/' + str(index))
    json_labels = json.dumps(new_mapping)
    with open(base_path + '/' + str(index) + '/labels.json', 'w') as file:
        file.write(json_labels)
    counts, images, labels = get_dataset(mode = 'character', character_type = 'full', characters = new_mapping)

    sz = len(new_mapping)

    model = models.Sequential()
    model.add(layers.Input(shape = (26, 14)))
    model.add(layers.Conv1D(32, 2, activation = 'relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(sz))

    model.summary()

    model.compile(optimizer='adam',
                  loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size = 0.2)
    model.fit(images, labels, epochs = 15, validation_data = (test_images, test_labels))

    model.save(base_path + '/' + str(index) + '/model.keras')

    predictions_errors = get_prediction_errors(model, images, labels, len(component))
    components = ocr_model.get_character_connected_components(sz, counts, predictions_errors)

    print(components)

def create_character_model(reuse = False):
    counts, images, labels = get_dataset(mode = 'character', character_type = 'full')

    if reuse is False:
        model = models.Sequential()
        model.add(layers.Input(shape = (16, 16)))
        model.add(layers.Conv1D(128, 3, activation = 'relu'))
        model.add(layers.MaxPooling1D(2, strides = 1))
        model.add(layers.Conv1D(256, 3, activation = 'relu'))
        model.add(layers.MaxPooling1D(2, strides = 1))
        model.add(layers.Flatten())
        model.add(layers.Dense(94 * 2, activation = 'relu'))
        model.add(layers.Dense(94))

        model.summary()

        model.compile(optimizer = 'adam',
                      loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                      metrics = ['accuracy'])

        for partition in range(DATASET_PARTITIONS):
            train_images, test_images = utils.train_test_split(images, DATASET_PARTITIONS, partition)
            train_labels, test_labels = utils.train_test_split(labels, DATASET_PARTITIONS, partition)
            model.fit(train_images, train_labels, epochs = 5, validation_data = (test_images, test_labels))
        model.fit(images, labels, epochs = 1)

        model.save('model/character/model.keras')
    else:
        model = models.load_model('model/character/model.keras')

    predictions_errors = get_prediction_errors(model, images, labels, len(CHARACTER_TO_LABEL))

    if 2 == 2:
        return

    components = ocr_model.get_character_connected_components(len(LABEL_TO_CHARACTER), counts, predictions_errors)

    print(components)

    for index, component in enumerate(components):
        create_character_fallback_model('model/character/fallback', component, index, 0)

ocr_model.initialize_mappings()
create_character_model(reuse = False)