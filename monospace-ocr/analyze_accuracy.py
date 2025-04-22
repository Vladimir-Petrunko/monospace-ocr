import cv2
import glob
import json
import numpy
import ocr
import time

annotations = glob.glob('D:/codescan/codescan' + '/**/annotations.json', recursive = True)[349:350]

def get_coding_grid_box(annotations, coding_grid_category):
    for annotation in annotations:
        if annotation['category_id'] == int(coding_grid_category):
            box = annotation['bbox']
            return box[1], box[1] + box[3], box[0], box[0] + box[2]

def analyze_iou():
    times = numpy.array([])
    skipped = 0
    total = 0
    for annotation in annotations:
        with open(annotation, 'r') as file:
            data = json.load(file)
            categories = data['categories']
            coding_grid_category = [key for key, value in categories.items() if value == 'coding_grid'][0]
            image_file = 'D:/codescan/codescan/' + data['images'][0].get('file_name') + '/bw.png'
            coding_grid_box = get_coding_grid_box(data['annotations'], coding_grid_category)
            image = cv2.imread(image_file)
            if coding_grid_box is None or image is None:
                skipped = skipped + 1
                continue
            start = time.time()
            row_l, row_r = coding_grid_box[0], coding_grid_box[1]
            col_l, col_r = coding_grid_box[2], coding_grid_box[3]
            ground_truth_bounding_box = image[row_l:row_r, col_l:col_r]
            text = ocr.image_to_text(ground_truth_bounding_box)
            end = time.time()
            total = total + 1
    print('Time (s):', 'mean:', numpy.mean(times), 'std:', numpy.std(times))
    print('Skipped items:', skipped, 'out of', total)

analyze_iou()