import os.path
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

DATASET_PARTITIONS = 5

def print_components(components):
    print('Components:')
    for component in components:
        print(component)

def create_cnn_model(classes):
    model = models.Sequential()
    model.add(layers.Input(shape = (20, 20, 1)))
    model.add(layers.Conv2D(256, (2, 2), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(512, (2, 2), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Flatten())
    model.add(layers.Dense(classes * 8, activation = 'relu'))
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

    print('Prediction errors:')
    for i in range(cnt):
        if len(predictions_errors[i]) > 0:
            print(i, predictions_errors[i])

    return predictions_errors

def get_dataset(characters = None):
    characters = CHARACTER_TO_LABEL if characters is None else characters
    counts, images, labels = [0 for _ in range(len(characters))], [], []
    fonts = Path('fonts')
    for font in fonts.iterdir():
        font_str = str(font).replace('\\', '/').replace('fonts/', '')
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

def create_character_fallback_model(base_path, component, index, level):
    new_mapping = {int(label): index for index, label in enumerate(component)}

    if not os.path.exists(base_path):
        os.mkdir(base_path)
    if not os.path.exists(base_path + '/' + str(index)):
        os.mkdir(base_path + '/' + str(index))
    json_labels = json.dumps(new_mapping)
    with open(base_path + '/' + str(index) + '/labels.json', 'w') as file:
        file.write(json_labels)
    counts, images, labels = get_dataset(characters = new_mapping)

    cnt = len(new_mapping)

    model = create_cnn_model(cnt)
    model.fit(images, labels, epochs = 10)
    model.save(base_path + '/' + str(index) + '/model.keras')

    predictions_errors = get_prediction_errors(model, images, labels, len(new_mapping))
    components = ocr_model.get_character_connected_components(len(new_mapping), counts, predictions_errors)

    print(components)

    for ind, comp in enumerate(components):
        if len(comp) < len(component):
            create_character_fallback_model(base_path + '/' + str(index), comp, ind, level + 1)

def create_character_model():
    counts, images, labels = get_dataset()

    cnt = len(LABEL_TO_CHARACTER)
    character_model = create_cnn_model(cnt)

    for i in range(DATASET_PARTITIONS):
        train_images, test_images = utils.train_test_split(images, DATASET_PARTITIONS, i)
        train_labels, test_labels = utils.train_test_split(labels, DATASET_PARTITIONS, i)
        character_model.fit(train_images, train_labels, epochs = 5, validation_data = (test_images, test_labels))
    character_model.fit(images, labels, epochs = 1)
    character_model.save('model/character/model.keras')

    predictions_errors = get_prediction_errors(character_model, images, labels, len(LABEL_TO_CHARACTER))
    components = ocr_model.get_character_connected_components(len(LABEL_TO_CHARACTER), counts, predictions_errors)

    print_components(components)

    # for index, component in enumerate(components):
    #     create_character_fallback_model('model/character/fallback', component, index, 0)

ocr_model.initialize_mappings()
create_character_model()