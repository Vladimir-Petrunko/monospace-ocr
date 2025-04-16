import cv2
import glob
import json
import numpy
import region_detector
import time
import utils

annotations = glob.glob('D:/codescan/codescan' + '/**/annotations.json', recursive = True)[:20]

def get_coding_grid_box(annotations, coding_grid_category):
    for annotation in annotations:
        if annotation['category_id'] == int(coding_grid_category):
            box = annotation['bbox']
            return box[1], box[1] + box[3], box[0], box[0] + box[2]

def analyze_iou():
    iou = numpy.array([])
    times = numpy.array([])
    skipped = 0
    total = 0
    for annotation in annotations:
        with open(annotation, 'r') as file:
            data = json.load(file)
            categories = data['categories']
            coding_grid_category = [key for key, value in categories.items() if value == 'coding_grid'][0]
            image_file = 'D:/codescan/codescan/' + data['images'][0].get('file_name') + '/img.png'
            coding_grid_box = get_coding_grid_box(data['annotations'], coding_grid_category)
            image = cv2.imread(image_file)
            if coding_grid_box is None or image is None:
                skipped = skipped + 1
                continue
            start = time.time()
            our_boxes = region_detector.detect_regions(image)
            end = time.time()
            best_iou = 0
            for box in our_boxes:
                best_iou = max(best_iou, utils.intersection_over_union(coding_grid_box, box))
            iou = numpy.append(iou, best_iou)
            times = numpy.append(times, end - start)
            total = total + 1
    print('IOU:', 'mean:', numpy.mean(iou), 'std:', numpy.std(iou))
    print('Time (s):', 'mean:', numpy.mean(times), 'std:', numpy.std(times))
    print('Skipped items:', skipped, 'out of', total)

analyze_iou()